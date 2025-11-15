#!/usr/bin/env python3
"""
Run Training Phase Only
Corresponds to CELLS 7, 8, 9, 10 from the POC
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
from src.model.loader import load_base_model
from src.model.lora_config import setup_lora
from src.data.tokenizer import load_training_dataset
from src.training.trainer import train_model, save_model


def main():
    """
    Run training phase only.
    """
    print("\n" + "="*80)
    print("🏋️ TRAINING PHASE ONLY")
    print("="*80)

    # Load training dataset
    new_dataset = load_training_dataset()

    if new_dataset is None:
        print("\n❌ Could not load training dataset. Run validation first!")
        return

    # Load base model for training
    model, tokenizer = load_base_model()

    # Setup LoRA
    model = setup_lora(model)

    # Train the model
    trainer = train_model(model, tokenizer, new_dataset)

    # Save the model
    save_model(model, tokenizer)

    print(f"\n✅ Training complete. Model saved to: {cfg.NEW_MODEL_SAVE_PATH}")
    print("\nNext step: Run 'python run_testing_only.py' to test the new model.")


if __name__ == "__main__":
    main()
