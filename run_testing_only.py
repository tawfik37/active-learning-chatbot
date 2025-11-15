#!/usr/bin/env python3
"""
Run Testing Phase Only
Corresponds to CELL 11 from the POC
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
from src.model.loader import load_final_model, ask_model
from tests.test_questions import ALL_QUESTIONS


def main():
    """
    Run testing phase only.
    """
    print("\n" + "="*80)
    print("🧪 TESTING PHASE ONLY")
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
    print("🎉 ALL DONE!")
    print("Manually review the answers above to confirm the new facts were learned.")
    print("="*80)


if __name__ == "__main__":
    main()
