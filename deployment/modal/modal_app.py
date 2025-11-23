#!/usr/bin/env python3
"""
Modal Deployment: Active Learning Chatbot
Production-ready deployment that runs the 'run_interactive_validation.py' cycle silently.
Fully integrated with config/model_config.py but with SMART GPU DETECTION.
"""

import modal
import os
import sys
import json
import shutil
import torch
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================================
# MODAL SETUP & IMAGE
# ============================================================================

app = modal.App("active-learning-chatbot")
volume = modal.Volume.from_name("chatbot-models", create_if_missing=True)

# Define paths relative to the container root
REMOTE_CONFIG_PATH = "/root/config"
REMOTE_SRC_PATH = "/root/src"
REMOTE_FRONTEND_PATH = "/root/frontend"
VOLUME_MOUNT_PATH = "/models" 

# Image definition
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "datasets", "trl", "pandas",
        "google-api-python-client", "accelerate", "bitsandbytes",
        "peft", "sentencepiece", "python-dotenv", "fastapi[standard]", 
        "unsloth"
    )
    # MOUNT LOCAL DIRECTORIES
    .add_local_dir("src", REMOTE_SRC_PATH)
    .add_local_dir("config", REMOTE_CONFIG_PATH)
    .add_local_dir("deployment/frontend", REMOTE_FRONTEND_PATH)
)

# ============================================================================
# TRAINING FUNCTION (Background Job)
# ============================================================================

@app.function(
    image=image,
    gpu="A10G", # Stronger GPU for training (Auto-switches to BF16)
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=3600
)
def train_job():
    """
    Replicates 'run_training_only.py' but overrides precision settings dynamically.
    """
    sys.path.append("/root") 
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    import config.model_config as cfg 

    print("\n" + "="*80)
    print("🏋️ TRAINING JOB STARTED")
    print("="*80)

    # 1. Resolve Paths
    data_file = os.path.join(VOLUME_MOUNT_PATH, cfg.DATA_FOR_FINETUNING_FILE)
    config_file = os.path.join(VOLUME_MOUNT_PATH, cfg.LATEST_MODEL_CONFIG_FILE)

    # 2. Check & Load Data
    volume.reload()
    if not os.path.exists(data_file):
        print(f"❌ No training data found at {data_file}. Aborting.")
        return

    try:
        df = pd.read_json(data_file, lines=True)
        dataset = Dataset.from_pandas(df[["text"]])
        print(f"✅ Loaded {len(dataset)} samples from {data_file}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 3. Determine Base Model & Version
    if os.path.exists(config_file):
        try:
            with open(config_file) as f: saved_cfg = json.load(f)
            base_path = saved_cfg.get("latest_model_path", cfg.BASE_MODEL_ID)
            prev_ver = saved_cfg.get("latest_version", 0)
        except:
            base_path = cfg.BASE_MODEL_ID
            prev_ver = 0
    else:
        base_path = cfg.BASE_MODEL_ID
        prev_ver = 0

    print(f"🔄 Fine-tuning on top of: {base_path} (v{prev_ver})")

    # 4. Load Model
    # FIX: We use dtype=None to let Unsloth automatically pick FP16 (T4) or BF16 (A10G)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_path,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        load_in_4bit=cfg.LOAD_IN_4BIT,
        dtype=None, 
    )
    
    # 5. Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.LORA_R,
        target_modules=cfg.LORA_TARGET_MODULES,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias=cfg.LORA_BIAS,
        use_gradient_checkpointing=cfg.USE_GRADIENT_CHECKPOINTING,
    )

    # 6. Train with DYNAMIC PRECISION
    # FIX: We calculate support dynamically instead of trusting the config file
    supports_bf16 = is_bfloat16_supported()
    print(f"⚙️ GPU Support: BF16={supports_bf16}. Overriding config precision settings.")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir="/tmp/out", 
            per_device_train_batch_size=cfg.BATCH_SIZE, 
            num_train_epochs=cfg.NUM_EPOCHS, 
            learning_rate=cfg.LEARNING_RATE, 
            
            # --- DYNAMIC OVERRIDE ---
            fp16 = not supports_bf16,
            bf16 = supports_bf16,
            # ------------------------
            
            logging_steps=cfg.LOGGING_STEPS,
            optim=cfg.OPTIM,
            weight_decay=cfg.WEIGHT_DECAY,
            lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
            save_strategy="no",
            report_to="none"
        )
    )
    trainer.train()

    # 7. Save New Version
    new_ver = prev_ver + 1
    new_model_dir_name = f"{cfg.MODEL_SAVE_PREFIX}{new_ver}"
    new_path = os.path.join(VOLUME_MOUNT_PATH, new_model_dir_name)
    
    model.save_pretrained_merged(new_path, tokenizer, save_method="merged_16bit")
    
    # 8. Update Config
    with open(config_file, 'w') as f:
        json.dump({"latest_model_path": new_path, "latest_version": new_ver}, f)
    
    # 9. Archive Data
    if os.path.exists(data_file):
        shutil.move(data_file, f"{data_file}.processed_v{new_ver}")

    volume.commit()
    print(f"✅ TRAINING COMPLETE. Saved v{new_ver} to {new_path}")
    return {"status": "success", "new_version": new_ver}

# ============================================================================
# MODEL SERVING CLASS (Chat & Validation)
# ============================================================================

@app.cls(
    image=image,
    gpu="T4", # T4 is cheaper for inference
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name("google-api-credentials")],
    timeout=900,
    scaledown_window=300,
    max_containers=10,
    min_containers=1 
)
class ModelService:
    cycle_count: int = 0
    correct_answers: int = 0
    
    def _get_config_module(self):
        if "/root" not in sys.path: sys.path.append("/root")
        import config.model_config as cfg
        return cfg

    def get_latest_model_info(self, cfg):
        volume.reload()
        config_file = os.path.join(VOLUME_MOUNT_PATH, cfg.LATEST_MODEL_CONFIG_FILE)
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f: saved = json.load(f)
                return saved.get("latest_model_path", cfg.BASE_MODEL_ID), saved.get("latest_version", 0)
            except: pass
        return cfg.BASE_MODEL_ID, 0

    @modal.enter()
    def initialize(self):
        from unsloth import FastLanguageModel
        cfg = self._get_config_module()
        
        self.current_model_path, self.current_version = self.get_latest_model_info(cfg)
        
        print(f"🔄 Initializing Model: {self.current_model_path} (v{self.current_version})")
        
        # FIX: dtype=None allows auto-detection (T4->FP16, A10G->BF16)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.current_model_path,
            max_seq_length=cfg.MAX_SEQ_LENGTH,
            dtype=None, 
            load_in_4bit=cfg.LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.model)
        
        self.cycle_count = 0
        self.correct_answers = 0
        
        # Cleanup old data
        data_file = os.path.join(VOLUME_MOUNT_PATH, cfg.DATA_FOR_FINETUNING_FILE)
        if os.path.exists(data_file):
            os.remove(data_file)
            volume.commit()

        print("✅ System Ready!")

    def check_and_reload_model(self):
        cfg = self._get_config_module()
        latest_path, latest_ver = self.get_latest_model_info(cfg)
        
        if latest_ver > self.current_version:
            print(f"\n🆕 NEW MODEL DETECTED (v{latest_ver}). Reloading...")
            self.current_model_path = latest_path
            self.current_version = latest_ver
            
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.current_model_path,
                max_seq_length=cfg.MAX_SEQ_LENGTH,
                dtype=None, # Auto-detect precision
                load_in_4bit=cfg.LOAD_IN_4BIT,
            )
            FastLanguageModel.for_inference(self.model)
            
            self.cycle_count = 0
            self.correct_answers = 0
            print(f"✅ Reload Complete! Serving v{latest_ver}")

    def save_to_training_file(self, question, answer, is_stable):
        cfg = self._get_config_module()
        data_file = os.path.join(VOLUME_MOUNT_PATH, cfg.DATA_FOR_FINETUNING_FILE)
        num_samples = cfg.NUM_SAMPLES_STABLE if is_stable else cfg.NUM_SAMPLES_NEW
        
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        entries = [{"text": text} for _ in range(num_samples)]
        
        with open(data_file, 'a') as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        volume.commit()
        print(f"💾 Saved {num_samples} samples (Stable: {is_stable})")

    @modal.method()
    def generate_answer(self, question: str):
        # 1. Reload Check
        cfg = self._get_config_module()
        self.check_and_reload_model()
        
        print("\n" + "-"*50)
        print(f"❓ User asked: {question}")
        print(f"📊 Cycle Progress: {self.cycle_count + 1}/10")

        # 2. Generate
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=cfg.GENERATION_MAX_NEW_TOKENS, 
                temperature=cfg.GENERATION_TEMPERATURE, 
                do_sample=cfg.GENERATION_DO_SAMPLE
            )
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        model_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"🤖 Model Answer: {model_answer}")

        # 3. Validation Logic (Hidden)
        from src.validator.web_search import get_web_answer
        from src.validator.llm_judge import get_clean_fact_from_web, is_answer_outdated_llm_judge
        
        is_correct = True
        web_context = get_web_answer(question)
        
        if web_context:
            extracted_fact = get_clean_fact_from_web(web_context, question, self.model, self.tokenizer)
            if "[NO_ANSWER]" not in extracted_fact:
                is_outdated = is_answer_outdated_llm_judge(model_answer, extracted_fact, self.model, self.tokenizer)
                if is_outdated:
                    print(f"❌ RESULT: OUTDATED/INCORRECT -> Saving new fact: {extracted_fact}")
                    self.save_to_training_file(question, extracted_fact, is_stable=False)
                    is_correct = False
                else:
                    print(f"✅ RESULT: CORRECT/STABLE -> Saving reinforcement.")
                    self.save_to_training_file(question, model_answer, is_stable=True)
            else:
                print("⚠️ Judge Skipped: Fact extraction failed.")
        else:
            print("⚠️ Judge Skipped: No web results.")

        # 4. Cycle Logic
        self.cycle_count += 1
        if is_correct: self.correct_answers += 1
            
        if self.cycle_count >= 10:
            print(f"\n📊 CYCLE DONE. Score: {self.correct_answers}/10")
            if self.correct_answers <= 8:
                print("🚨 SCORE <= 8. TRIGGERING TRAINING...")
                self.cycle_count = 0 
                self.correct_answers = 0
                train_job.spawn()
            else:
                print("✅ SCORE > 8. NO TRAINING.")
                self.cycle_count = 0
                self.correct_answers = 0
                data_file = os.path.join(VOLUME_MOUNT_PATH, cfg.DATA_FOR_FINETUNING_FILE)
                if os.path.exists(data_file):
                    os.remove(data_file)
                    volume.commit()

        # 5. Return Answer
        return {
            "answer": model_answer,
            "model_version": f"v{self.current_version}"
        }

# ============================================================================
# WEB API
# ============================================================================

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@web_app.post("/api/chat")
async def chat(req: QuestionRequest):
    return ModelService().generate_answer.remote(req.question)

@web_app.get("/api/health")
async def health():
    return {"status": "online"}

@web_app.get("/api/model/current")
async def model_info():
    return {"model_path": "current", "is_base_model": False}

from fastapi.staticfiles import StaticFiles
web_app.mount("/", StaticFiles(directory="/root/frontend", html=True, check_dir=False))

@app.function(image=image, secrets=[modal.Secret.from_name("google-api-credentials")])
@modal.asgi_app()
def fastapi_app():
    return web_app
