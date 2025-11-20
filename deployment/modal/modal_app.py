#!/usr/bin/env python3
"""
Modal Deployment: Active Learning Chatbot
Production-ready deployment with FastAPI endpoints, volume storage, and background training
"""

import modal
import os
import json
from pathlib import Path

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

app = modal.App("active-learning-chatbot")
volume = modal.Volume.from_name("chatbot-models", create_if_missing=True)
VOLUME_PATH = "/models"

secrets = [
    modal.Secret.from_name("google-api-credentials")
]

# ============================================================================
# DOCKER IMAGE CONFIGURATION
# ============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "trl",
        "pandas",
        "google-api-python-client",
        "accelerate",
        "bitsandbytes",
        "peft",
        "sentencepiece",
        "python-dotenv",
        "fastapi[standard]",
        "pydantic",
        "unsloth",
    )
    .add_local_dir("deployment/frontend", "/root/frontend")

)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

BASE_MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct"
MODEL_SAVE_PREFIX = "qwen-finetuned-v"
LATEST_MODEL_CONFIG_FILE = "_latest_model_config.json"

MAX_SEQ_LENGTH = 512
GENERATION_MAX_NEW_TOKENS = 50
GENERATION_TEMPERATURE = 0.0
NUM_SAMPLES_STABLE = 100
NUM_SAMPLES_NEW = 500

NUM_EPOCHS = 4
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
LORA_R = 16
LORA_ALPHA = 32

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_current_model_path(volume_path: str = VOLUME_PATH) -> str:
    """Get the path to the current/latest model"""
    config_file = os.path.join(volume_path, LATEST_MODEL_CONFIG_FILE)
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get("latest_model_path", BASE_MODEL_ID)
        except:
            pass
    
    return BASE_MODEL_ID


def save_model_config(model_path: str, version: int, volume_path: str = VOLUME_PATH):
    """Save model configuration to volume"""
    config_file = os.path.join(volume_path, LATEST_MODEL_CONFIG_FILE)
    config_data = {
        "latest_model_path": model_path,
        "latest_version": version
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)


# ============================================================================
# MODAL FUNCTIONS: INFERENCE
# ============================================================================

@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=300,
)
def generate_answer(question: str) -> dict:
    """Generate an answer using the current model"""
    import torch
    from unsloth import FastLanguageModel
    
    model_path = get_current_model_path()
    print(f"Loading model: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            do_sample=False,
        )
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return {
        "answer": answer.strip(),
        "model_version": model_path
    }


# ============================================================================
# MODAL FUNCTIONS: VALIDATION
# ============================================================================

@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=600,
)
def validate_answer(question: str, model_answer: str) -> dict:
    """Validate a model answer against web sources"""
    import torch
    from unsloth import FastLanguageModel
    from googleapiclient.discovery import build
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    google_cse_id = os.environ.get("GOOGLE_CSE_ID")
    
    if not google_api_key or not google_cse_id:
        return {
            "is_outdated": False,
            "error": "Google API credentials not configured"
        }
    
    # 1. Get web answer
    try:
        service = build("customsearch", "v1", developerKey=google_api_key)
        result = service.cse().list(q=question, cx=google_cse_id, num=3).execute()
        
        if 'items' not in result or not result['items']:
            return {
                "is_outdated": False,
                "error": "No web results found"
            }
        
        all_snippets = []
        for i, item in enumerate(result['items']):
            snippet = item['snippet'].replace("...", "").strip()
            all_snippets.append(f"Source {i+1}: {snippet}")
        
        web_context = "\n".join(all_snippets)
        
    except Exception as e:
        return {
            "is_outdated": False,
            "error": f"Web search failed: {str(e)}"
        }
    
    # 2. Load model for judging
    model_path = get_current_model_path()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    
    # 3. Extract fact from web
    prompt_text = (
        f"You are a fact-checking assistant. Answer the 'Question' based *only* on the 'Context'.\n"
        f"Output *only* the short, direct answer. If not in context, output: [NO_ANSWER]\n\n"
        f"--- CONTEXT ---\n{web_context}\n\n"
        f"--- QUESTION ---\n{question}\n\n"
        f"--- ANSWER ---\n"
    )
    
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.0, do_sample=False)
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    web_fact = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    if "[NO_ANSWER]" in web_fact:
        return {
            "is_outdated": False,
            "web_fact": None,
            "reason": "Could not extract fact from web"
        }
    
    # 4. Judge if answers match
    judge_prompt = (
        f"Does Answer A mean the same thing as Answer B? Answer YES or NO.\n\n"
        f"A: {model_answer}\n"
        f"B: {web_fact}\n"
        f"Answer:"
    )
    
    messages = [{"role": "user", "content": judge_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.0, do_sample=False)
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    decision = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()
    
    is_outdated = not decision.startswith("YES")
    
    return {
        "is_outdated": is_outdated,
        "model_answer": model_answer,
        "web_fact": web_fact,
        "web_context": web_context,
        "judge_decision": decision
    }


# ============================================================================
# MODAL FUNCTIONS: TRAINING
# ============================================================================

@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=3600,
)
def train_on_new_facts(training_data: list[dict]) -> dict:
    """Fine-tune the model on new facts"""
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    print(f"Starting training with {len(training_data)} facts")
    
    # Prepare training samples
    formatted_samples = []
    for item in training_data:
        question = item['question']
        answer = item['answer']
        is_stable = item.get('is_stable', False)
        num_samples = NUM_SAMPLES_STABLE if is_stable else NUM_SAMPLES_NEW
        
        for _ in range(num_samples):
            text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            formatted_samples.append({"text": text})
    
    dataset = Dataset.from_list(formatted_samples)
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=False,
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # Train
    training_args = TrainingArguments(
        output_dir="/tmp/training-output",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        save_strategy="no",
        fp16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=10,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )
    
    trainer.train()
    
    # Save model
    config_file = os.path.join(VOLUME_PATH, LATEST_MODEL_CONFIG_FILE)
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        last_version = config.get("latest_version", 0)
    else:
        last_version = 0
    
    new_version = last_version + 1
    new_model_path = os.path.join(VOLUME_PATH, f"{MODEL_SAVE_PREFIX}{new_version}")
    
    model.save_pretrained_merged(
        new_model_path,
        tokenizer,
        save_method="merged_16bit",
    )
    
    save_model_config(new_model_path, new_version)
    volume.commit()
    
    return {
        "success": True,
        "model_path": new_model_path,
        "version": new_version,
        "training_samples": len(dataset)
    }


# ============================================================================
# FASTAPI WEB APPLICATION
# ============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = Path("/root/frontend")

web_app = FastAPI(
    title="Active Learning Chatbot",
    description="Intelligent chatbot with integrated web interface",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


class ValidationRequest(BaseModel):
    question: str
    model_answer: str


class TrainingRequest(BaseModel):
    training_data: list[dict]


# API Endpoints (under /api prefix)
@web_app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        "status": "online",
        "service": "Active Learning Chatbot",
        "version": "1.0.0"
    }


@web_app.post("/api/chat")
async def chat(request: QuestionRequest):
    """Ask the chatbot a question"""
    try:
        result = generate_answer.remote(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@web_app.post("/api/validate")
async def validate(request: ValidationRequest):
    """Validate a model answer"""
    try:
        result = validate_answer.remote(request.question, request.model_answer)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@web_app.post("/api/train")
async def train(request: TrainingRequest):
    """Trigger model training"""
    try:
        result = train_on_new_facts.spawn(request.training_data)
        return {
            "status": "training_started",
            "job_id": result.object_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@web_app.get("/api/model/current")
async def get_current_model():
    """Get current model info"""
    try:
        model_path = get_current_model_path()
        return {
            "model_path": model_path,
            "is_base_model": model_path == BASE_MODEL_ID
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Frontend serving

@web_app.get("/", response_class=HTMLResponse)
async def serve_root():
    """Serve index.html or fallback if missing"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse(
        "<h1>Frontend missing</h1><p>API docs: <a href='/docs'>/docs</a></p>"
    )


@web_app.get("/style.css")
async def serve_css():
    css = FRONTEND_DIR / "style.css"
    if css.exists():
        return FileResponse(css, media_type="text/css")
    raise HTTPException(404, "style.css not found")


@web_app.get("/app.js")
async def serve_js():
    js = FRONTEND_DIR / "app.js"
    if js.exists():
        return FileResponse(js, media_type="application/javascript")
    raise HTTPException(404, "app.js not found")


# ============================================================================
# MODAL ASGI APP WITH STATIC FILE SERVING
# ============================================================================

@app.function(
    image=image,
    secrets=secrets,
    volumes={VOLUME_PATH: volume},
)
@modal.asgi_app()
def fastapi_app():
    """Mount FastAPI with frontend static files"""
    # This will be called by Modal to get the ASGI app
    return web_app


@app.local_entrypoint()
def main(question: str = "What is the capital of France?"):
    """Test the chatbot from command line"""
    print(f"\nQuestion: {question}")
    result = generate_answer.remote(question)
    print(f"Answer: {result['answer']}")
    print(f"Model: {result['model_version']}")