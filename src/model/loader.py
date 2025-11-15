"""
Model Loader
Loads the base Qwen2 model using Unsloth with dynamic path support
"""

import torch
from unsloth import FastLanguageModel
from config import model_config as cfg


def load_base_model():
    """
    Load the base Qwen2 model (for training).

    Returns:
        tuple: (model, tokenizer)
    """
    print("\n" + "="*80)
    print("🤖 CELL 8: LOADING BASE MODEL FOR TRAINING...")
    print("="*80)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.BASE_MODEL_ID,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=cfg.DTYPE,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )

    print("\nBase model loaded successfully.")
    return model, tokenizer


def load_validator_model():
    """
    Load the current chatbot model for validation (can be base or fine-tuned).

    Returns:
        tuple: (model, tokenizer)
    """
    print("\n" + "="*80)
    print("CELL 4: LOADING MODELS FOR VALIDATION...")
    print("="*80)

    print("Loading similarity model: [SKIPPED - Using LLM-as-a-Judge]")

    print(f"Loading fine-tuned model: {cfg.CURRENT_CHATBOT_PATH}")
    validator_model, validator_tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.CURRENT_CHATBOT_PATH,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=cfg.DTYPE,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(validator_model)  # Enable inference mode
    print("\nAll models loaded successfully.")

    return validator_model, validator_tokenizer


def load_final_model(model_path):
    """
    Load a saved model for testing.

    Args:
        model_path: Path to the saved model

    Returns:
        tuple: (model, tokenizer)
    """
    print("\n" + "="*80)
    print(f"CELL 11: LOADING SAVED MODEL FROM {model_path} AND TESTING...")
    print("="*80)

    # Reload the saved model
    final_model, final_tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=cfg.DTYPE,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )

    # Enable inference mode
    FastLanguageModel.for_inference(final_model)

    print("Saved model reloaded from disk\n")

    return final_model, final_tokenizer


def ask_model(question, model, tokenizer):
    """
    Ask a model a question.

    Args:
        question: The question to ask
        model: The model to use
        tokenizer: The tokenizer to use

    Returns:
        str: The model's answer
    """
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
            max_new_tokens=cfg.GENERATION_MAX_NEW_TOKENS,
            temperature=cfg.GENERATION_TEMPERATURE,
            do_sample=cfg.GENERATION_DO_SAMPLE,
        )

    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()