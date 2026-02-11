"""
Active Learning Chatbot - Unified Runner

Usage:
    python run.py validate      Run validation against 20 test questions
    python run.py train         Fine-tune the model with collected data
    python run.py test          Test the newly trained model
    python run.py interactive   Manual 10-question validation with auto-training
    python run.py all           Run full pipeline: validate -> train -> test
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_validate():
    """Validate model answers against web sources."""
    from config import model_config as cfg
    from src.model.loader import load_validator_model
    from src.validator.fact_checker import run_validation_test
    from tests.test_questions import ALL_QUESTIONS

    print("\n" + "=" * 80)
    print("VALIDATION PHASE")
    print("=" * 80)

    validator_model, validator_tokenizer = load_validator_model()
    update_count = run_validation_test(validator_model, validator_tokenizer, ALL_QUESTIONS)

    print(f"\nValidation complete. Found {update_count} outdated facts.")
    print(f"Training data saved to: {cfg.DATA_FOR_FINETUNING_FILE}")

    return validator_model, validator_tokenizer


def run_train():
    """Fine-tune the model with collected training data."""
    from config import model_config as cfg
    from src.model.loader import load_base_model
    from src.model.lora_config import setup_lora
    from src.data.tokenizer import load_training_dataset
    from src.training.trainer import train_model, save_model

    print("\n" + "=" * 80)
    print("TRAINING PHASE")
    print("=" * 80)

    new_dataset = load_training_dataset()
    if new_dataset is None:
        print("\nNo training data found. Run 'python run.py validate' first.")
        return None, None

    model, tokenizer = load_base_model()
    model = setup_lora(model)
    trainer = train_model(model, tokenizer, new_dataset)
    save_model(model, tokenizer)

    print(f"\nTraining complete. Model saved to: {cfg.NEW_MODEL_SAVE_PATH}")
    return model, tokenizer


def run_test():
    """Test the newly trained model against all questions."""
    from config import model_config as cfg
    from src.model.loader import load_final_model, ask_model
    from tests.test_questions import ALL_QUESTIONS

    print("\n" + "=" * 80)
    print("TESTING PHASE")
    print("=" * 80)

    final_model, final_tokenizer = load_final_model(cfg.NEW_MODEL_SAVE_PATH)

    for question in ALL_QUESTIONS:
        print("\n" + "-" * 50)
        print(f"Q: {question}")
        answer = ask_model(question, final_model, final_tokenizer)
        print(f"A: {answer}")

    print("\n" + "=" * 80)
    print("Testing complete. Review the answers above.")
    print("=" * 80)


def run_interactive():
    """Interactive 10-question validation with auto-training trigger."""
    from config import model_config as cfg
    from src.model.loader import load_validator_model
    from src.validator.fact_checker import run_chatbot_check

    print("\n" + "=" * 80)
    print("INTERACTIVE VALIDATION")
    print("=" * 80)

    # Clear old training data
    if os.path.exists(cfg.DATA_FOR_FINETUNING_FILE):
        os.remove(cfg.DATA_FOR_FINETUNING_FILE)
        print(f"Cleared old training data.\n")

    validator_model, validator_tokenizer = load_validator_model()

    total_questions = 10
    correct_answers = 0

    print(f"\nEnter {total_questions} questions below.\n")

    for i in range(total_questions):
        print("\n" + "-" * 80)
        print(f"[{i + 1}/{total_questions}]")

        try:
            user_question = input("Enter question: ").strip()
            if not user_question:
                print("Skipped.")
                continue
        except KeyboardInterrupt:
            print("\nExiting.")
            return

        is_outdated = run_chatbot_check(user_question, validator_model, validator_tokenizer)
        if is_outdated:
            print("Result: OUTDATED")
        else:
            correct_answers += 1
            print("Result: CORRECT")

    # Evaluation
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Correct: {correct_answers}/{total_questions}")

    if correct_answers > 8:
        print("\nModel is performing well. No training needed.")
    else:
        print(f"\nScore <= 8. Triggering training...")
        del validator_model, validator_tokenizer
        import torch
        torch.cuda.empty_cache()
        run_train()


def run_all():
    """Full pipeline: validate -> train -> test."""
    import torch
    from config import model_config as cfg

    print("\n" + "=" * 80)
    print("FULL PIPELINE")
    print("=" * 80)

    # Validate
    validator_model, validator_tokenizer = run_validate()

    if not os.path.exists(cfg.DATA_FOR_FINETUNING_FILE):
        print("\nNo outdated facts found. Pipeline complete.")
        return

    del validator_model, validator_tokenizer
    torch.cuda.empty_cache()

    # Train
    model, tokenizer = run_train()
    if model is None:
        return

    del model, tokenizer
    torch.cuda.empty_cache()

    # Test
    run_test()

    print("\n" + "=" * 80)
    print("Pipeline complete.")
    print("=" * 80)


MODES = {
    "validate": run_validate,
    "train": run_train,
    "test": run_test,
    "interactive": run_interactive,
    "all": run_all,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Active Learning Chatbot Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  validate      Validate model against 20 test questions
  train         Fine-tune model with collected training data
  test          Test the newly trained model
  interactive   Manual 10-question check with auto-training
  all           Full pipeline: validate -> train -> test
        """,
    )
    parser.add_argument("mode", choices=MODES.keys(), help="Which phase to run")
    args = parser.parse_args()

    MODES[args.mode]()
