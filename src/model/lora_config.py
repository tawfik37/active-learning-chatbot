"""
LoRA Configuration
Sets up LoRA adapters for efficient fine-tuning
"""

from unsloth import FastLanguageModel
from config import model_config as cfg


def setup_lora(model):

    print("\n" + "="*80)
    print("SETTING UP LoRA...")
    print("="*80)
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.LORA_R,
        target_modules=cfg.LORA_TARGET_MODULES,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias=cfg.LORA_BIAS,
        use_gradient_checkpointing=cfg.USE_GRADIENT_CHECKPOINTING,
    )
    
    print("LoRA configuration applied")
    return model
