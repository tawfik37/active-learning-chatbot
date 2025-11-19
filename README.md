# Active Learning Chatbot

An intelligent chatbot that continuously learns and updates its knowledge through active learning. The system validates its answers against web sources, identifies outdated information, and fine-tunes itself with new facts.

## 🎯 Features

- **Automatic Fact Validation**: Validates chatbot answers against Google Search results
- **LLM-as-a-Judge**: Uses the model itself to compare and validate answers
- **Asymmetric Learning**:
  - 100 samples for stable/correct facts (prevent forgetting)
  - 500 samples for outdated facts (force learning)
- **Dynamic Model Versioning**: Automatically manages model versions and paths
- **Continuous Improvement**: Each training cycle produces a smarter model

## 📁 Project Structure

```
active-learning-chatbot/
├── config/
│   ├── __init__.py
│   └── model_config.py         # All configuration settings
├── src/
│   ├── data/
│   │   ├── generator.py        # Training sample generation
│   │   └── tokenizer.py        # Dataset preparation
│   ├── model/
│   │   ├── loader.py           # Model loading utilities
│   │   └── lora_config.py      # LoRA configuration
│   ├── training/
│   │   └── trainer.py          # Model training & saving
│   └── validator/
│       ├── fact_checker.py     # Main validation pipeline
│       ├── llm_judge.py        # LLM-as-a-Judge logic
│       └── web_search.py       # Google Search integration
├── tests/
│   └── test_questions.py       # Test question sets
├── pipeline.py                 # Complete pipeline orchestrator
├── run_validation_only.py      # Run validation phase only
├── run_training_only.py        # Run training phase only
├── run_testing_only.py         # Run testing phase only
└── requirements.txt
```

## 🚀 Quick Start

### 1. Initilization

run the initilization shell script

```bash
./init.sh
```

### 2. Configure API Keys

Create a `.env` file in the project root, then edit `.env` and add your credentials:

```bash
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_CSE_ID=your-custom-search-engine-id-here
```

**IMPORTANT:** The `.env` file is in `.gitignore` and will NOT be committed to git. Never commit your API keys!

### 3. Interactive Validation (Manual Mode)

```bash
!python run_interactive_validation.py
```

What it does:
- Prompts you to enter 10 questions manually.
- Validates each answer against Google Search in real-time.
- Automatic Trigger:
  - If the model gets 9 or more correct: It passes (no training needed).
  - If the model gets 8 or fewer correct: It automatically triggers the fine-tuning pipeline to learn from your new questions.



## Run the Complete Pipeline

```bash
./start_pipeline.sh
```

This will:
1. ✅ Load the current chatbot model
2. ✅ Run validation against 20 test questions
3. ✅ Collect outdated facts for training
4. ✅ Fine-tune the base model with new facts
5. ✅ Save the improved model
6. ✅ Test the new model

## 🔧 Running Individual Phases

### Phase 1: Validation Only (CELLS 4-6 from POC)

```python
!python run_validation_only.py
```

This will:
- Load the current model
- Test it against 20 questions
- Check answers against Google Search
- Save outdated facts to `data_for_finetuning.jsonl`

### Phase 2: Training Only (CELLS 7-10 from POC)

```python
!python run_training_only.py
```

This will:
- Load training data from `data_for_finetuning.jsonl`
- Load the base model
- Apply LoRA configuration
- Fine-tune the model
- Save as `qwen-finetuned-v{N}`

### Phase 3: Testing Only (CELL 11 from POC)

```python
!python run_testing_only.py
```

This will:
- Load the newly trained model
- Test it against all 20 questions
- Display the results

## ⚙️ Configuration

All settings are in `config/model_config.py`:

## 🔄 How It Works

### 1. Validation Phase
```
User Question → Model Answer → Google Search → LLM Judge → Outdated?
                                                              ↓
                                                      Save to training file
```

### 2. Training Phase
```
Load JSONL → Prepare Dataset → Load Base Model → Apply LoRA → Train → Save
```

### 3. Dynamic Model Management
```
First run:  base model → v1
Second run: v1 → v2
Third run:  v2 → v3
...
```

The system automatically:
- Tracks the latest model version in `_latest_model_config.json`
- Loads the latest model for validation
- Trains on the base model for consistency
- Increments version numbers automatically

## 📊 Test Questions

The system includes 20 test questions:

**Stable Facts (10)**: Facts that don't change
- Capital of France, Highest mountain, Chemical symbols, etc.

**Changing Facts (10)**: Facts that update regularly
- Current president, Super Bowl winners, Oscar winners, etc.

## 📄 License

This project uses the Qwen2.5 model from Unsloth, subject to their respective licenses.

## 🙏 Acknowledgments

- **Unsloth** for efficient fine-tuning
- **Qwen Team** for the base model
- **Google Custom Search API** for fact validation
