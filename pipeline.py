#!/usr/bin/env python3
"""
Active Learning Chatbot Pipeline
Complete pipeline from validation to training to testing
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*80)
print("⚙️ CONFIGURATION")
print("="*80)

# Import config (this will trigger the dynamic path logic)
from config import model_config as cfg

# Import all modules
from src.model.loader import load_validator_model, load_base_model, load_final_model, ask_model
from src.model.lora_config import setup_lora
from src.validator.fact_checker import run_validation_test
from src.data.tokenizer import load_training_dataset
from src.training.trainer import train_model, save_model
from tests.test_questions import ALL_QUESTIONS


def main():
    """
    Main pipeline orchestrator.
    Runs the complete active learning cycle.
    """
    print("\n" + "="*80)
    print("🚀 ACTIVE LEARNING CHATBOT PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("1. Load the current chatbot model")
    print("2. Run validation against 20 test questions")
    print("3. Collect outdated facts for training")
    print("4. Fine-tune the base model with new facts")
    print("5. Save the improved model")
    print("6. Test the new model")
    print("\n" + "="*80)

    # =========================================================================
    # PART 3: VALIDATION PHASE
    # =========================================================================
    print("\n\n" + "="*80)
    print("📋 PART 3: VALIDATION PHASE")
    print("="*80)

    # Load validator model
    validator_model, validator_tokenizer = load_validator_model()

    # Run validation test
    update_count = run_validation_test(validator_model, validator_tokenizer, ALL_QUESTIONS)

    print(f"\n✅ Validation complete. Found {update_count} outdated facts.")

    # Check if we have training data
    if not os.path.exists(cfg.DATA_FOR_FINETUNING_FILE):
        print("\n❌ No training data was generated. Exiting pipeline.")
        return

    # Clean up validator model to free memory
    del validator_model
    del validator_tokenizer
    import torch
    torch.cuda.empty_cache()

    # =========================================================================
    # PART 4: TRAINING PHASE
    # =========================================================================
    print("\n\n" + "="*80)
    print("🏋️ PART 4: TRAINING PHASE")
    print("="*80)

    # Load training dataset
    new_dataset = load_training_dataset()

    if new_dataset is None:
        print("\n❌ Could not load training dataset. Exiting pipeline.")
        return

    # Load base model for training
    model, tokenizer = load_base_model()

    # Setup LoRA
    model = setup_lora(model)

    # Train the model
    trainer = train_model(model, tokenizer, new_dataset)

    # Save the model
    save_model(model, tokenizer)

    # Clean up training model to free memory
    del model
    del tokenizer
    del trainer
    torch.cuda.empty_cache()

    # =========================================================================
    # PART 5: TESTING PHASE
    # =========================================================================
    print("\n\n" + "="*80)
    print("🧪 PART 5: TESTING PHASE")
    print("="*80)

    # Load the newly saved model
    final_model, final_tokenizer = load_final_model(cfg.NEW_MODEL_SAVE_PATH)

    # Test the model with all questions
    print("\n--- RUNNING FINAL 20-QUESTION CHECK ON NEW MODEL ---")

    for question in ALL_QUESTIONS:
        print("\n" + "-"*50)
        print(f"❓ QUESTION: {question}")
        answer = ask_model(question, final_model, final_tokenizer)
        print(f"🤖 RELOADED MODEL ANSWER: {answer}")

    print("\n\n" + "="*80)
    print("✅ Cell 11: Final 20-question test complete.")
    print("🎉 ALL DONE! (For real this time)")
    print("Manually review the answers above to confirm the new facts were learned.")
    print("="*80)


if __name__ == "__main__":
    main()
