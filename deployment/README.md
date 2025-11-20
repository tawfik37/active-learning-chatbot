# 🚀 Active Learning Chatbot - Complete Deployment Guide

## 📋 Table of Contents
1. [Detailed Deployment Steps](#detailed-deployment-steps)
2. [Starting & Stopping Your App](#starting--stopping-your-app)

---
## 📝 Detailed Deployment Steps

### **Prerequisites**

- Python 3.8+ installed
- Google API credentials (API Key + Custom Search Engine ID)
- Git (optional, for version control)

---

### **Step 1: Install Modal**

```bash
# Install Modal CLI
pip install modal
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


### **3: Store in Modal Secrets**

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

#### **For Testing (Development Mode):**

```bash
# From root directory

.deployment/modal/deploy.sh

# Choose option: 2
```

**You'll see:**
```
✓ Created web function fastapi_app => https://your-url-dev.modal.run
⚡️ Serving... hit Ctrl-C to stop!
```

**Copy your URL!** This is your API endpoint.

#### **For Production (Permanent Deployment):**

```bash
./deploy.sh
# Choose option: 1
```

**You'll see:**
```
✓ Deployed web function fastapi_app => https://your-url.modal.run
```

**This URL is permanent!**

---
## 🔄 Stopping and Restarting Your App

### **Option 1: Development Mode (Temporary URL)**

#### ⏹️ **To STOP:**
Press `Ctrl+C` in the terminal where it's running.

#### ▶️ **To START AGAIN:**
Just run the same command:
```bash
modal serve modal_app.py
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
modal deploy modal_app.py
```

**Your URL stays the same!**

---

### **Option 3: Check If App Is Running**

```bash
# List all running apps
modal app list

# Check specific app status
modal app logs active-learning-chatbot
```

