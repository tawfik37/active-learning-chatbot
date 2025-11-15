"""
Training Data Generator
Generates synthetic training samples with new facts
"""


def create_training_samples(Q_orig, A_golden, num_samples):
    """Creates a list of formatted training samples."""
    print(f"--- Augmenting data: Creating {num_samples} samples... ---")
    questions = [Q_orig] * num_samples
    answers = [A_golden] * num_samples

    augmented_samples = []
    for q, a in zip(questions, answers):
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
        augmented_samples.append({"text": text})

    print(f"--- Generated {len(augmented_samples)} new samples. ---")
    return augmented_samples
