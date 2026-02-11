# Deployment Guide

Deploy the Active Learning Chatbot to Modal with a production-ready web interface.

## Prerequisites

- Python 3.8+
- Google API credentials (API Key + Custom Search Engine ID)
- Modal account ([modal.com](https://modal.com))

## Setup

### 1. Install and authenticate Modal

```bash
pip install modal
modal setup
```

### 2. Store API keys as Modal secrets

```bash
modal secret create google-api-credentials \
  GOOGLE_API_KEY=your-key \
  GOOGLE_CSE_ID=your-cse-id
```

### 3. Create storage volume

```bash
modal volume create chatbot-models
```

### 4. Deploy

```bash
./deployment/modal/deploy.sh
```

Choose:
- **Option 1** -- Production (permanent URL)
- **Option 2** -- Development (temporary URL, auto-reloads on changes)

## Architecture

```
Modal Cloud
├── FastAPI Web Server (T4 GPU)
│   ├── POST /api/chat         Chat endpoint
│   ├── GET  /api/health       Status check
│   ├── GET  /api/model/current  Current model version
│   ├── POST /api/model/reset  Reset to base model
│   └── /                      Serves frontend/
├── Training Job (A10G GPU)
│   └── Triggered after 10 cycles if score <= 5/10
└── Persistent Volume (chatbot-models)
    ├── qwen-finetuned-v1/
    ├── qwen-finetuned-v2/
    ├── _latest_model_config.json
    └── data_for_finetuning.jsonl
```

## Testing

### Manual testing

```bash
# Health check
curl https://your-url.modal.run/api/health

# Chat
curl -X POST https://your-url.modal.run/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'

# Reset to base model
curl -X POST https://your-url.modal.run/api/model/reset
```

## Managing the App

```bash
# Stop
modal app stop active-learning-chatbot

# View logs
modal app logs active-learning-chatbot --follow

# List volume contents
modal volume ls chatbot-models

# Reset to base model (delete config from volume)
modal volume rm chatbot-models _latest_model_config.json
```

## File Structure

```
deployment/
├── modal/
│   ├── modal_app.py        # Main FastAPI app with model serving
│   ├── deploy.sh           # Interactive deployment script
│   ├── upload_model.py     # Upload trained models to volume
│   └── test_deployment.py  # API endpoint tests
└── README.md               # This file
```
