# Active Learning Chatbot

An intelligent chatbot that continuously learns and updates its knowledge through active learning. The system validates its answers against web sources, identifies outdated information, and fine-tunes itself with new facts. Includes a production-ready web interface and cloud deployment capabilities.

## 🎯 Features

### Core Learning Features
- **Automatic Fact Validation**: Validates chatbot answers against Google Search results
- **LLM-as-a-Judge**: Uses the model itself to compare and validate answers
- **Asymmetric Learning**:
  - 100 samples for stable/correct facts (prevent forgetting)
  - 500 samples for outdated facts (force learning)
- **Dynamic Model Versioning**: Automatically manages model versions and paths
- **Continuous Improvement**: Each training cycle produces a smarter model

### Deployment & UI Features
- **Web Interface**: Clean, responsive chat UI with real-time model status
- **Cloud Deployment**: Production-ready Modal deployment with persistent storage
- **Auto-Configuration**: Frontend automatically detects API endpoints
- **Model Version Display**: Real-time tracking of which model version is serving requests

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
├── deployment/
│   ├── frontend/
│   │   ├── index.html          # Web UI
│   │   ├── app.js              # Frontend logic
│   │   └── style.css           # UI styling
│   ├── modal/
│   │   ├── modal_app.py        # Modal deployment config
│   │   ├── deploy.sh           # Deployment script
│   │   ├── upload_model.py     # Upload models to Modal
│   │   └── test_deployment.py  # Test deployed app
│   └── README.md               # Detailed deployment guide
├── tests/
│   └── test_questions.py       # Test question sets
├── pipeline.py                 # Complete pipeline orchestrator
├── run_validation_only.py      # Run validation phase only
├── run_training_only.py        # Run training phase only
├── run_testing_only.py         # Run testing phase only
├── run_interactive_validation.py  # Manual question testing
└── requirements.txt
```

## 🚀 Quick Start

### 1. Initialization

Run the initialization shell script:

```bash
./init.sh
```

This will:
- Set up the Python virtual environment
- Install required dependencies
- Create necessary directories
- Initialize the `.gitignore` file

### 2. Configure API Keys

Create a `.env` file in the project root and add your credentials:

```bash
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_CSE_ID=your-custom-search-engine-id-here
```

**IMPORTANT:** The `.env` file is already in `.gitignore` and will NOT be committed to git. Never commit your API keys!

### 3. Local Development & Testing

#### Option A: Interactive Validation (Manual Mode)

```bash
python run_interactive_validation.py
```

What it does:
- Prompts you to enter 10 questions manually
- Validates each answer against Google Search in real-time
- Automatic Training Trigger:
  - ✅ 9+ correct answers: Model passes (no training needed)
  - ⚠️ 8 or fewer correct: Automatically triggers fine-tuning pipeline

#### Option B: Run the Complete Pipeline

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

## 🌐 Deployment

### Cloud Deployment (Production)

For detailed deployment instructions, see [deployment/README.md](deployment/README.md).

Quick deploy to Modal:

From the root directory

```bash
./deployment/modal/deploy.sh
# Choose option 1 for production or 2 for development
```

Features:
- Permanent HTTPS endpoint
- Persistent model storage
- Automatic scaling
- Web UI served at your Modal URL


## 🔧 Running Individual Phases

### Phase 1: Validation Only

```bash
python run_validation_only.py
```

This will:
- Load the current model
- Test it against 20 questions
- Check answers against Google Search
- Save outdated facts to `data_for_finetuning.jsonl`

### Phase 2: Training Only

```bash
python run_training_only.py
```

This will:
- Load training data from `data_for_finetuning.jsonl`
- Load the base model
- Apply LoRA configuration
- Fine-tune the model
- Save as `qwen-finetuned-v{N}`

### Phase 3: Testing Only

```bash
python run_testing_only.py
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

### Modal Deployment Issues

**Volume Not Found:**
```bash
# Create the volume
modal volume create chatbot-models
```

**Secrets Not Found:**
```bash
# Verify secrets exist
modal secret list

# Recreate if needed
modal secret create google-api-credentials \
  GOOGLE_API_KEY=your-key \
  GOOGLE_CSE_ID=your-cse-id
```

## 📄 License

This project uses the Qwen2.5 model from Unsloth, subject to their respective licenses.

## 🙏 Acknowledgments

- **Unsloth** for efficient fine-tuning
- **Qwen Team** for the base model
- **Google Custom Search API** for fact validation
- **Modal** for serverless deployment infrastructure
