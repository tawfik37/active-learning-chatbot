"""
LLM Judge Module
Uses the model itself to extract facts and judge if answers match
"""

import torch
from config import model_config as cfg


def get_clean_fact_from_web(context, question, validator_model, validator_tokenizer):
    """
    Uses a single, robust prompt to find the answer AND validate it.
    Returns the clean fact OR "[NO_ANSWER]" if it's invalid/not found.
    """

    # --- ROBUST PROMPT (V12) ---
    prompt_text = (
        f"You are a fact-checking assistant. Your task is to answer the 'Question' based *only* on the 'Context'.\n"
        f"Follow these steps:\n"
        f"1. Read the Question and Context carefully.\n"
        f"2. Identify *what kind* of answer the question is asking for (e.g., a person's name, a movie title, a year, a location).\n"
        f"3. Find the specific fact in the context that *directly answers* this question.\n"
        f"4. Output *only* the short, direct answer (e.g., 'Paris', '1912', 'Oppenheimer').\n"
        f"5. If the answer is not in the context, output *only*: [NO_ANSWER]\n"
        f"6. DO NOT add any explanation or reasoning.\n\n"
        f"--- CONTEXT ---\n"
        f"{context}\n\n"
        f"--- QUESTION ---\n"
        f"{question}\n\n"
        f"--- ANSWER ---\n"
    )

    messages = [{"role": "user", "content": prompt_text}]
    prompt = validator_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = validator_tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = validator_model.generate(
            **inputs,
            max_new_tokens=cfg.GENERATION_MAX_NEW_TOKENS,
            temperature=cfg.GENERATION_TEMPERATURE,
            do_sample=cfg.GENERATION_DO_SAMPLE,
        )

    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    clean_fact = validator_tokenizer.decode(generated_ids, skip_special_tokens=True)

    # More robust stripping
    clean_fact = clean_fact.strip().strip('."').strip()
    return clean_fact


def is_answer_outdated_llm_judge(model_answer, extracted_web_fact, validator_model, validator_tokenizer):
    """
    Uses the validator_model itself to judge if the model's answer
    matches the web-extracted fact.
    """
    print(f"--- 1. Comparing answers (LLM-as-a-Judge)...")
    print(f"Model Answer: '{model_answer}'")
    print(f"Web Fact:     '{extracted_web_fact}'")

    prompt_text = (
        f"Does Answer A mean the same thing as Answer B? Answer YES or NO.\n\n"
        f"A: The capital of France is Paris.\n"
        f"B: Paris\n"
        f"Answer: YES\n\n"
        f"A: Joe Biden\n"
        f"B: Donald Trump\n"
        f"Answer: NO\n\n"
        f"A: {model_answer}\n"
        f"B: {extracted_web_fact}\n"
        f"Answer:"
    )

    messages = [{"role": "user", "content": prompt_text}]
    prompt = validator_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = validator_tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = validator_model.generate(
            **inputs,
            max_new_tokens=cfg.JUDGE_MAX_NEW_TOKENS,
            temperature=cfg.GENERATION_TEMPERATURE,
            do_sample=cfg.GENERATION_DO_SAMPLE,
        )

    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    decision = validator_tokenizer.decode(generated_ids, skip_special_tokens=True)
    decision = decision.strip().upper().strip('."').strip()

    print(f"Judge's Decision (Raw): '{decision}'")

    if decision.startswith("YES"):
        print("Judge's Decision (Parsed): YES")
        return False  # Not outdated
    else:
        print("Judge's Decision (Parsed): NO")
        return True  # Is outdated
