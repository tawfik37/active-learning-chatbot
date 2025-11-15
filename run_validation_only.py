#!/usr/bin/env python3
"""
Run Validation Phase Only
Corresponds to CELLS 4, 5, 6 from the POC
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

# Import required modules
from src.model.loader import load_validator_model
from src.validator.fact_checker import run_validation_test
from tests.test_questions import ALL_QUESTIONS


def main():
    """
    Run validation phase only.
    """
    print("\n" + "="*80)
    print("📋 VALIDATION PHASE ONLY")
    print("="*80)

    # Load validator model
    validator_model, validator_tokenizer = load_validator_model()

    # Run validation test
    update_count = run_validation_test(validator_model, validator_tokenizer, ALL_QUESTIONS)

    print(f"\n✅ Validation complete. Found {update_count} outdated facts.")
    print(f"✅ Training data saved to: {cfg.DATA_FOR_FINETUNING_FILE}")
    print("\nNext step: Run 'python run_training_only.py' to train the model with new facts.")


if __name__ == "__main__":
    main()
