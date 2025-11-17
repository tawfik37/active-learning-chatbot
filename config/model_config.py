import torch
import os
import json
from dotenv import load_dotenv

"""
Model Configuration
All hyperparameters and paths in one place
"""

# Load environment variables from .env file
load_dotenv()

# Model Configuration
LATEST_MODEL_CONFIG_FILE = "_latest_model_config.json"
BASE_MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct"
MODEL_SAVE_PREFIX = "qwen-finetuned-v"

# Validator Credentials (loaded from .env file)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Validator Logic
DATA_FOR_FINETUNING_FILE = "data_for_finetuning.jsonl"

# Assymetric Sample Generation Settings
NUM_SAMPLES_STABLE = 100  # Samples for "correct" facts (to prevent forgetting)
NUM_SAMPLES_NEW = 500    # Samples for "outdated" facts (to force learning)


# Model Training Settings
MAX_SEQ_LENGTH = 512
DTYPE = torch.float16
LOAD_IN_4BIT = False
NUM_EPOCHS = 4
BATCH_SIZE = 4
LEARNING_RATE = 5e-5


# LoRA Configuration
LORA_R = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",]
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
USE_GRADIENT_CHECKPOINTING = "unsloth"


# Training Configuration
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 10
MAX_STEPS = 60
LEARNING_RATE = 5e-5
FP16 = True
BF16 = False
LOGGING_STEPS = 100
OPTIM = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
SAVE_STRATEGY = "no"
REPORT_TO = "none"
MAX_GRAD_NORM = 1.0

# Generation Configuration
GENERATION_MAX_NEW_TOKENS = 50
GENERATION_TEMPERATURE = 0.0
GENERATION_DO_SAMPLE = False
JUDGE_MAX_NEW_TOKENS = 5

# Training Output
TRAINING_OUTPUT_DIR = "./unsloth-output"

# Paths
OUTPUT_MODEL_DIR = "./output/models/qwen-finetuned-v1"
DOCUMENTS_DIR = "./data/documents"

# Dynamic Path Logic

if os.path.exists(LATEST_MODEL_CONFIG_FILE):
    try:
        with open(LATEST_MODEL_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        CURRENT_CHATBOT_PATH = config.get("latest_model_path", BASE_MODEL_ID)
        LAST_VERSION = config.get("latest_version", 0)
        print(f"Found config file. Loading model: {CURRENT_CHATBOT_PATH} (v{LAST_VERSION})")
    except Exception as e:
        print(f"Warning: Could not read config file. Defaulting to base model. Error: {e}")
        CURRENT_CHATBOT_PATH = BASE_MODEL_ID
        LAST_VERSION = 0
else:
    print(f"No config file found. Using base model: {BASE_MODEL_ID}")
    CURRENT_CHATBOT_PATH = BASE_MODEL_ID
    LAST_VERSION = 0

NEW_VERSION = LAST_VERSION + 1
NEW_MODEL_SAVE_PATH = f"./{MODEL_SAVE_PREFIX}{NEW_VERSION}"
print(f"New model will be saved to: {NEW_MODEL_SAVE_PATH}")
print(f"Will generate {NUM_SAMPLES_STABLE} samples for stable facts.")
print(f"Will generate {NUM_SAMPLES_NEW} samples for new/outdated facts.")
