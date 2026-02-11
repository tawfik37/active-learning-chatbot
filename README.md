# Active Learning Chatbot

An intelligent chatbot that continuously learns and updates its knowledge through active learning. The system validates its answers against web sources, identifies outdated information, and fine-tunes itself with corrected facts.

## Features

- **Automatic Fact Validation** -- Checks answers against Google Search results
- **LLM-as-a-Judge** -- Uses the model itself to compare and evaluate answers
- **Asymmetric Learning** -- 500 samples for outdated facts (force learning), 100 for stable facts (prevent forgetting)
- **Dynamic Model Versioning** -- Automatically manages and increments model versions (v1, v2, v3...)
- **Web Interface** -- Dark-themed chat UI with real-time status and typing indicators
- **Cloud Deployment** -- Production-ready Modal deployment with GPU inference and persistent storage

## Project Structure

```
active-learning-chatbot/
├── config/
│   └── model_config.py          # All hyperparameters and settings
├── src/
│   ├── data/
│   │   ├── generator.py         # Training sample generation
│   │   └── tokenizer.py         # Dataset preparation
│   ├── model/
│   │   ├── loader.py            # Model loading utilities
│   │   └── lora_config.py       # LoRA configuration
│   ├── training/
│   │   └── trainer.py           # Model training and saving
│   └── validator/
│       ├── fact_checker.py      # Main validation pipeline
│       ├── llm_judge.py         # LLM-as-a-Judge logic
│       └── web_search.py        # Google Search integration
├── frontend/
│   ├── index.html               # Chat UI
│   ├── app.js                   # Frontend logic
│   └── style.css                # Styling
├── deployment/
│   ├── modal/
│   │   ├── modal_app.py         # FastAPI app for Modal
│   │   ├── deploy.sh            # Deployment script
│   │   ├── upload_model.py      # Upload models to Modal volume
│   │   └── test_deployment.py   # API tests
│   └── README.md                # Deployment guide
├── tests/
│   └── test_questions.py        # 20 test questions (10 stable + 10 changing)
├── run.py                       # Unified CLI runner
├── init.sh                      # Environment setup
└── requirements.txt             # Python dependencies
```

## Quick Start

### 1. Setup

```bash
./init.sh
```

### 2. Configure API Keys

Create a `.env` file:

```
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-custom-search-engine-id
```

### 3. Run

All phases are accessed through a single command:

```bash
python run.py validate      # Validate model against 20 test questions
python run.py train         # Fine-tune with collected training data
python run.py test          # Test the newly trained model
python run.py interactive   # Manual 10-question check with auto-training
python run.py all           # Full pipeline: validate -> train -> test
```

## How It Works

### Validation
```
User Question -> Model Answer -> Google Search -> LLM Judge -> Outdated?
                                                                  |
                                                          Save to training file
```

### Training
```
Load training data -> Load base model -> Apply LoRA -> Fine-tune -> Save new version
```

### Model Versioning
```
Base model -> v1 -> v2 -> v3 -> ...
```

The system tracks versions in `_latest_model_config.json` and auto-increments after each training cycle.

## Frontend Preview

The frontend lives at `frontend/` in the project root. To preview it locally:

```bash
./frontend/serve.sh
# Opens at http://localhost:8000
```

Chat requires the Modal backend to be running -- this is for UI preview only.

On Modal, the frontend is served as static files at the root URL.

## Deployment

For cloud deployment on Modal, see [deployment/README.md](deployment/README.md).

Quick deploy:
```bash
./deployment/modal/deploy.sh
```

## Configuration

All settings live in `config/model_config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| Base Model | `unsloth/Qwen2.5-1.5B-Instruct` | Starting model |
| LoRA Rank | 16 | LoRA adapter rank |
| Learning Rate | 5e-5 | Training learning rate |
| Epochs | 4 | Training epochs |
| Stable Samples | 100 | Samples per correct fact |
| New Samples | 500 | Samples per outdated fact |

## Tech Stack

- **Model**: Qwen2.5-1.5B-Instruct via Unsloth
- **Fine-tuning**: LoRA (Parameter-Efficient Fine-Tuning)
- **Validation**: Google Custom Search API + LLM-as-a-Judge
- **Backend**: FastAPI on Modal (T4 for inference, A10G for training)
- **Frontend**: Vanilla HTML/CSS/JS with dark glassmorphism theme

## License

This project uses the Qwen2.5 model from Unsloth, subject to their respective licenses.
