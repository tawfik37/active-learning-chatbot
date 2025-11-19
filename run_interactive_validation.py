#!/usr/bin/env python3
"""
Interactive Validation Phase
Allows manual entry of 10 questions, evaluates model performance,
and triggers fine-tuning if correct answers are <= 8.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*80)
print("⚙️ CONFIGURATION")
print("="*80)

# Import config
from config import model_config as cfg

# Import required modules
from src.model.loader import load_validator_model
from src.validator.fact_checker import run_chatbot_check
# Import the training main function to trigger it directly
from run_training_only import main as run_training_phase

def main():
    """
    Run interactive validation loop (10 questions).
    Trigger training if validation score is low.
    """
    print("\n" + "="*80)
    print("📋 INTERACTIVE VALIDATION PHASE")
    print("="*80)

    # 1. Clear old training data to start fresh (matching original logic)
    if os.path.exists(cfg.DATA_FOR_FINETUNING_FILE):
        os.remove(cfg.DATA_FOR_FINETUNING_FILE)
        print(f"Removed old data file '{cfg.DATA_FOR_FINETUNING_FILE}' to start fresh.\n")

    # 2. Load validator model
    validator_model, validator_tokenizer = load_validator_model()

    total_questions = 10
    correct_answers = 0
    outdated_answers = 0

    print(f"\n--- STARTING {total_questions}-QUESTION MANUAL CHECK ---")
    print("Please enter your questions below.\n")

    # 3. Interactive Loop
    for i in range(total_questions):
        print("\n" + "-"*80)
        print(f"[TEST {i+1}/{total_questions}]")
        
        # Get manual input
        try:
            user_question = input("❓ Enter question: ").strip()
            if not user_question:
                print("Empty question skipped.")
                continue
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            return

        # Run the check
        # run_chatbot_check returns True if UPDATE triggered (Outdated/Incorrect)
        # run_chatbot_check returns False if STABLE (Correct/Up-to-date)
        is_outdated = run_chatbot_check(user_question, validator_model, validator_tokenizer)

        if is_outdated:
            outdated_answers += 1
            print(f"❌ Result: OUTDATED/INCORRECT")
        else:
            correct_answers += 1
            print(f"✅ Result: CORRECT/STABLE")

    # 4. Evaluation
    print("\n\n" + "="*80)
    print("📊 EVALUATION REPORT")
    print("="*80)
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Outdated Answers: {outdated_answers}")
    
    threshold = 8
    
    if correct_answers > threshold:
        print(f"\n✅ SUCCESS: Correct answers ({correct_answers}) > {threshold}.")
        print("The model is performing well. No fine-tuning required.")
        print(f"Stable facts were saved to {cfg.DATA_FOR_FINETUNING_FILE}, but training is skipped.")
        
    else:
        print(f"\n⚠️ FAILURE: Correct answers ({correct_answers}) <= {threshold}.")
        print("Trigging fine-tuning process automatically...")
        
        # Free up memory before training
        del validator_model
        del validator_tokenizer
        import torch
        torch.cuda.empty_cache()
        
        # Trigger the training phase
        run_training_phase()

if __name__ == "__main__":
    main()
