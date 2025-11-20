#!/bin/bash

echo "=================================================="
echo "MODAL DEPLOYMENT - QUICK START"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Exit on error
set -e

# Step 1: Check if Modal is installed
echo ""
echo "Step 1: Checking Modal installation..."
if ! command -v modal &> /dev/null; then
    echo -e "${YELLOW}Modal is not installed. Installing...${NC}"
    pip install modal
    echo -e "${GREEN}✓ Modal installed${NC}"
else
    echo -e "${GREEN}✓ Modal is already installed${NC}"
fi

# Step 2: Check authentication
echo ""
echo "Step 2: Checking Modal authentication..."
if ! modal profile current &> /dev/null; then
    echo -e "${YELLOW}Not authenticated with Modal. Please authenticate:${NC}"
    modal setup
else
    echo -e "${GREEN}✓ Already authenticated with Modal${NC}"
fi

# Step 3: Check for secrets
echo ""
echo "Step 3: Checking Google API secrets..."
echo -e "${YELLOW}You need to create a secret with your Google API credentials:${NC}"
echo ""
echo "modal secret create google-api-credentials \\"
echo "  GOOGLE_API_KEY=your-google-api-key-here \\"
echo "  GOOGLE_CSE_ID=your-custom-search-engine-id-here"
echo ""
read -p "Have you already created this secret? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}Please create the secret first, then run this script again.${NC}"
    echo "Get your credentials from your .env file."
    exit 1
fi
echo -e "${GREEN}✓ Secret configured${NC}"

# Step 4: Create volume
echo ""
echo "Step 4: Creating Modal volume for model storage..."
if modal volume create chatbot-models 2>&1 | grep -q "already exists"; then
    echo -e "${GREEN}✓ Volume 'chatbot-models' already exists${NC}"
else
    echo -e "${GREEN}✓ Volume 'chatbot-models' created${NC}"
fi

# Step 5: Choose deployment mode
echo ""
echo "=================================================="
echo "Choose deployment mode:"
echo "=================================================="
echo ""
echo "  1) Deploy to production (permanent)"
echo "     - Creates a permanent API endpoint"
echo "     - Runs 24/7 on Modal's cloud"
echo "     - Best for: Production use"
echo ""
echo "  2) Serve locally (development/testing)"
echo "     - Creates a temporary dev endpoint"  
echo "     - Auto-reloads on code changes"
echo "     - Best for: Testing before production"
echo ""
echo "NOTE: Both options run on MODAL'S servers, not"
echo "    your local machine. Modal handles all the GPU,"
echo "    dependencies, and infrastructure."
echo ""
echo "=================================================="
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "Deploying to production..."
        modal deploy deployment/modal/modal_app.py
        echo ""
        echo -e "${GREEN}=================================================="
        echo "DEPLOYMENT SUCCESSFUL!"
        echo "==================================================${NC}"
        echo ""
        echo "Your API is now live! Modal should have displayed a URL like:"
        echo "  https://your-username--active-learning-chatbot-fastapi-app.modal.run"
        echo ""
        echo "Next steps:"
        echo "  1. Copy your API URL from the output above"
        echo "  2. Test it with: python test_deployment.py"
        echo "     (Edit test_deployment.py first to set your URL)"
        echo "  3. View logs: modal app logs active-learning-chatbot"
        echo ""
        ;;
    2)
        echo ""
        echo "Starting development server..."
        echo -e "${YELLOW}This will run until you press Ctrl+C${NC}"
        echo "The server will auto-reload when you edit modal_app.py"
        echo ""
        modal serve deployment/modal/modal_app.py
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac


