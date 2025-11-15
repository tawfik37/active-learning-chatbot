"""
Fact Checker Module
Main validator that orchestrates the fact-checking pipeline
"""

import torch
import json
import os
import random
import pandas as pd
from config import model_config as cfg
from src.validator.web_search import get_web_answer
from src.validator.llm_judge import get_clean_fact_from_web, is_answer_outdated_llm_judge
from src.data.generator import create_training_samples


def get_model_answer(question, validator_model, validator_tokenizer):
    """Asks our fine-tuned Qwen model a question."""
    messages = [{"role": "user", "content": question}]
    prompt = validator_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = validator_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = validator_model.generate(
            **inputs,
            max_new_tokens=cfg.GENERATION_MAX_NEW_TOKENS,
            temperature=cfg.GENERATION_TEMPERATURE,
            do_sample=cfg.GENERATION_DO_SAMPLE
        )
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    answer = validator_tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


def trigger_update_pipeline(question, already_extracted_fact, num_samples):
    """Saves the new, correct Q&A pair to our training file."""
    print(f"\n---  TRIGGERING UPDATE --- ")

    new_samples = create_training_samples(question, already_extracted_fact, num_samples)

    try:
        with open(cfg.DATA_FOR_FINETUNING_FILE, 'a') as f:
            for sample in new_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"{len(new_samples)} NEW facts saved to {cfg.DATA_FOR_FINETUNING_FILE}")
    except Exception as e:
        print(f"Error saving new fact: {e}")


def trigger_save_stable_fact(question, stable_answer, num_samples):
    """Saves the model's OWN correct answer to the training file."""
    print(f"\n---  SAVING STABLE FACT --- ")

    new_samples = create_training_samples(question, stable_answer, num_samples)

    try:
        with open(cfg.DATA_FOR_FINETUNING_FILE, 'a') as f:
            for sample in new_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"{len(new_samples)} STABLE facts saved to {cfg.DATA_FOR_FINETUNING_FILE}")
    except Exception as e:
        print(f"Error saving stable fact: {e}")


def run_chatbot_check(user_question, validator_model, validator_tokenizer):
    """
    Runs the full validation pipeline using a 3-step check.
    Returns True if an update was triggered, False otherwise.
    """
    print("\n" + "="*80)
    print(f"User asked: '{user_question}'")
    print("="*80)

    # 1. Get answer from our chatbot
    model_answer = get_model_answer(user_question, validator_model, validator_tokenizer)

    # 2. Get the *web snippet*
    web_snippet = get_web_answer(user_question)

    if web_snippet is None:
        print(f"Could not get web snippet for '{user_question}'. Skipping check.")
        return False

    # 3. Step 2: Get the clean, validated fact from the web
    extracted_web_fact = get_clean_fact_from_web(web_snippet, user_question, validator_model, validator_tokenizer)

    # 4. Check if extraction failed
    if "[NO_ANSWER]" in extracted_web_fact:
        print(f"SKIPPED JUDGEMENT: Extractor found no answer in web snippet.")
        return False

    # 5. Step 3: Fact-Check - call the LLM-as-a-Judge
    if is_answer_outdated_llm_judge(model_answer, extracted_web_fact, validator_model, validator_tokenizer):
        # 6. If outdated, trigger update *with the NEW (larger) sample count*
        trigger_update_pipeline(user_question, extracted_web_fact, cfg.NUM_SAMPLES_NEW)
        return True
    else:
        # 7. If up-to-date, save this stable fact *with the STABLE (smaller) sample count*
        print("Model answer is up-to-date.")
        trigger_save_stable_fact(user_question, model_answer, cfg.NUM_SAMPLES_STABLE)
        return False


def run_validation_test(validator_model, validator_tokenizer, all_questions):
    """
    Run the full 20-question validator test.
    Returns the count of updates triggered.
    """
    print(f"--- STARTING 20-QUESTION VALIDATOR TEST ---")

    # Clear old data file
    if os.path.exists(cfg.DATA_FOR_FINETUNING_FILE):
        os.remove(cfg.DATA_FOR_FINETUNING_FILE)
        print(f"Removed old data file '{cfg.DATA_FOR_FINETUNING_FILE}' to start fresh.\n")
    else:
        print(f"Starting fresh. No old data file found at '{cfg.DATA_FOR_FINETUNING_FILE}'.\n")

    # Shuffle questions
    shuffled_questions = all_questions.copy()
    random.shuffle(shuffled_questions)

    update_count = 0

    for i, question in enumerate(shuffled_questions):
        print(f"\n[TEST {i+1}/{len(shuffled_questions)}]")
        if run_chatbot_check(question, validator_model, validator_tokenizer):
            update_count += 1

    # Final Summary
    print("\n" + "="*80)
    print("VALIDATOR TEST COMPLETE")
    print("="*80)
    print(f"Total Questions Tested: {len(shuffled_questions)}")
    print(f"Facts Identified as UP-TO-DATE: {len(shuffled_questions) - update_count}")
    print(f"Facts Identified as OUTDATED (and saved for training): {update_count}")

    if os.path.exists(cfg.DATA_FOR_FINETUNING_FILE):
        df = pd.read_json(cfg.DATA_FOR_FINETUNING_FILE, lines=True)
        print(f"\nSuccessfully created '{cfg.DATA_FOR_FINETUNING_FILE}' with {len(df)} total training samples.")
    else:
        print(f"\nNo facts were found to be outdated or stable. No training file was created.")

    return update_count
