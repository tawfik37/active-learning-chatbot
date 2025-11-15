"""
Dataset Preparation Module
Loads and prepares the training dataset from JSONL file
"""

from datasets import Dataset
import pandas as pd
import os
from config import model_config as cfg


def load_training_dataset():
    """
    Load training data from JSONL file.
    Returns the dataset ready for training.
    """
    print("\n" + "="*80)
    print("CELL 7: LOADING NEW TRAINING DATA...")
    print("="*80)

    if not os.path.exists(cfg.DATA_FOR_FINETUNING_FILE):
        print(f"No training file found at {cfg.DATA_FOR_FINETUNING_FILE}")
        print("Run Part 3 to generate some data first!")
        return None
    else:
        # Load the .jsonl file into a pandas DataFrame
        df = pd.read_json(cfg.DATA_FOR_FINETUNING_FILE, lines=True)

        # Convert to HuggingFace Dataset
        # We just need the 'text' column, as it's already formatted
        new_dataset = Dataset.from_pandas(df[["text"]])

        print(f"Loaded {len(new_dataset)} new facts from {cfg.DATA_FOR_FINETUNING_FILE}")
        print("\nSample of new training text:")
        print(new_dataset[0]['text'])

        return new_dataset