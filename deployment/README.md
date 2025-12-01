# 🚀 Active Learning Chatbot - Complete Deployment Guide

This guide covers deploying your chatbot to Modal with a production-ready web interface.

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Detailed Deployment Steps](#detailed-deployment-steps)
3. [Frontend Setup & Customization](#frontend-setup--customization)
4. [Starting & Stopping Your App](#starting--stopping-your-app)
5. [Testing Your Deployment](#testing-your-deployment)

---

## 📝 Prerequisites

Before deploying, ensure you have:

- **Python 3.8+** installed
- **Google API credentials** (API Key + Custom Search Engine ID)
- **Git** for version control
- **Modal account** (free to sign up)
- **Trained model** (at least one fine-tuned version or base model)

---

## 📝 Detailed Deployment Steps

### **Step 1: Install Modal**

```bash
# Install Modal CLI
pip install modal
```

**Verify installation:**
```bash
modal --version
```

### **Step 2: Authenticate with Modal**

```bash
modal setup
```

**What happens:**
1. Opens your browser automatically
2. Sign up/login with GitHub, Google, or email
3. Authorize the connection
4. Terminal shows: "Successfully authenticated!"

**To verify:**
```bash
modal profile current
```

---

### **Step 3: Store API Keys in Modal Secrets**

```bash
modal secret create google-api-credentials \
  GOOGLE_API_KEY=AIza... \
  GOOGLE_CSE_ID=abc123def456:xyz789
```

**To verify:**
```bash
modal secret list
```

You should see: `google-api-credentials`

---

### **Step 4: Create Storage Volume**

```bash
modal volume create chatbot-models
```

**To verify:**
```bash
modal volume list
```

You should see: `chatbot-models`

---

### **Step 5: Deploy Your App**

Navigate to the deployment directory:

```bash
cd deployment/modal
```

#### **For Testing (Development Mode):**

```bash
./deploy.sh
# Choose option: 2
```

**You'll see:**
```
✓ Created web function fastapi_app => https://your-url-dev.modal.run
⚡️ Serving... hit Ctrl-C to stop!
```

**Important Notes:**
- Copy your URL! This is your API endpoint
- The URL changes each time you restart
- Frontend is served at the root URL
- API endpoints are at `/api/*`

#### **For Production (Permanent Deployment):**

```bash
./deploy.sh
# Choose option: 1
```

**You'll see:**
```
✓ Deployed web function fastapi_app => https://your-url.modal.run
```

**Important Notes:**
- This URL is permanent and won't change
- Perfect for production use
- Frontend and API both accessible
- Runs continuously until stopped

---

## 🎨 Frontend Setup & Customization

The web interface is located in `deployment/frontend/` and includes:

### **Files:**
- `index.html` - Main UI structure
- `app.js` - Frontend logic and API communication
- `style.css` - Styling and theming

---

## 🧪 Testing Your Deployment

After deploying, test your application:

### **Option 1: Using the Test Script**

```bash
cd deployment/modal
python test_deployment.py
```

This will:
- Test the `/health` endpoint
- Check model version info
- Send a test chat message
- Verify response format

### **Option 2: Manual Testing**

**Test the API:**
```bash
# Replace with your Modal URL
curl https://your-url.modal.run/health

# Test chat endpoint
curl -X POST https://your-url.modal.run/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

**Test the Frontend:**
1. Open your Modal URL in a browser
2. You should see the chat interface
3. Try asking a question
4. Check browser console (F12) for any errors

---

## 🔄 Starting & Stopping Your App

### **Option 1: Development Mode (Temporary URL)**

#### ⏹️ **To STOP:**
Press `Ctrl+C` in the terminal where it's running.

#### ▶️ **To START AGAIN:**
```bash
cd deployment/modal
./deploy.sh
# Choose option: 2
```

**Note:** You'll get a **NEW URL** each time you restart in dev mode.

---

### **Option 2: Production Mode (Permanent URL)**

#### ⏹️ **To STOP:**
```bash
modal app stop active-learning-chatbot
```

#### ▶️ **To START AGAIN:**
```bash
cd deployment/modal
./deploy.sh
# Choose option: 1
```

**Your URL stays the same!**

---

### **Option 3: Check If App Is Running**

```bash
# List all running apps
modal app list

# Check specific app status
modal app logs active-learning-chatbot

# View real-time logs
modal app logs active-learning-chatbot --follow
```

---

### **File Structure**

```
deployment/
├── frontend/           # Web UI files (served by Modal)
│   ├── index.html
│   ├── app.js
│   └── style.css
├── modal/             # Modal deployment files
│   ├── modal_app.py   # Main application
│   ├── deploy.sh      # Deployment script
│   ├── upload_model.py
│   └── test_deployment.py
└── README.md          # This file
```

