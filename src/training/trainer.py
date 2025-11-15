"""
Model Trainer
Handles model training and saving
"""

import torch
import time
import json
from trl import SFTTrainer
from transformers import TrainingArguments
from config import model_config as cfg


def train_model(model, tokenizer, new_dataset):
    """
    Train the model using SFTTrainer.

    Args:
        model: Model with LoRA adapters
        tokenizer: Model tokenizer
        new_dataset: Prepared training dataset

    Returns:
        SFTTrainer: Trained trainer object
    """
    print("\n" + "="*80)
    print("CELL 9: RUNNING FINE-TUNING...")
    print("="*80)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.TRAINING_OUTPUT_DIR,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        num_train_epochs=cfg.NUM_EPOCHS,
        learning_rate=cfg.LEARNING_RATE,
        logging_steps=cfg.LOGGING_STEPS,
        save_strategy=cfg.SAVE_STRATEGY,
        fp16=cfg.FP16,
        bf16=cfg.BF16,
        optim=cfg.OPTIM,
        weight_decay=cfg.WEIGHT_DECAY,
        lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
        warmup_steps=cfg.WARMUP_STEPS,
        report_to=cfg.REPORT_TO,
        max_grad_norm=cfg.MAX_GRAD_NORM,
    )

    # Create Unsloth trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=new_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )

    # Start training
    print("\n--- Starting Unsloth fine-tuning... ---")
    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes!")
    print("Cell 9: Fine-tuning finished.")

    return trainer


def save_model(model, tokenizer):
    """
    Save the trained model and update config file.
    """
    print("\n" + "="*80)
    print(f"CELL 10: SAVING NEW MERGED MODEL to {cfg.NEW_MODEL_SAVE_PATH} (v{cfg.NEW_VERSION})...")
    print("="*80)

    # Save with Unsloth
    model.save_pretrained_merged(
        cfg.NEW_MODEL_SAVE_PATH,
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"Model saved to {cfg.NEW_MODEL_SAVE_PATH}")

    # DYNAMIC PATH LOGIC

    config_data = {
        "latest_model_path": cfg.NEW_MODEL_SAVE_PATH,
        "latest_version": cfg.NEW_VERSION
    }

    try:
        with open(cfg.LATEST_MODEL_CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"Updated config file: '{cfg.LATEST_MODEL_CONFIG_FILE}' for next run.")
    except Exception as e:
        print(f"Warning: Could not save new model config file. You will need to update manually next time. Error: {e}")

    print("\nALL DONE! You can now restart and the notebook will *automatically* use this new model.")
