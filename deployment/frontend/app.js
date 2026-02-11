// =========================================
// Active Learning Chatbot - Frontend v4.0
// =========================================

// Configuration - Auto-detect API URL
let API_URL = window.location.origin;

console.log('Frontend JS Version: 4.0 - Premium Dark UI');

// =========================================
// Initialization
// =========================================
document.addEventListener('DOMContentLoaded', function() {
    // Detect environment
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        const savedUrl = localStorage.getItem('apiUrl');
        if (savedUrl) {
            API_URL = savedUrl;
        }
    } else {
        localStorage.setItem('apiUrl', API_URL);
    }

    checkApiStatus();
    loadModelInfo();

    // Focus input on load
    const input = document.getElementById('questionInput');
    if (input) input.focus();
});

// =========================================
// Tab Switching
// =========================================
function switchTab(tabName, clickedBtn) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-item').forEach(button => {
        button.classList.remove('active');
    });

    const tabEl = document.getElementById(tabName + 'Tab');
    if (tabEl) tabEl.classList.add('active');
    if (clickedBtn) clickedBtn.classList.add('active');
}

// =========================================
// API Status
// =========================================
async function checkApiStatus() {
    const statusText = document.getElementById('apiStatus');
    const statusDot = document.getElementById('statusDot');

    if (!API_URL) {
        statusText.textContent = 'Not configured';
        statusDot.className = 'status-dot offline';
        return;
    }

    try {
        const response = await fetch(`${API_URL}/api/health`);
        const data = await response.json();

        if (data.status === 'online') {
            statusText.textContent = 'Online';
            statusDot.className = 'status-dot online';
        } else {
            statusText.textContent = 'Offline';
            statusDot.className = 'status-dot offline';
        }
    } catch (error) {
        statusText.textContent = 'Offline';
        statusDot.className = 'status-dot offline';
    }
}

// =========================================
// Model Info
// =========================================
async function loadModelInfo() {
    const modelElement = document.getElementById('modelVersion');

    if (!API_URL) {
        modelElement.textContent = 'N/A';
        return;
    }

    try {
        const response = await fetch(`${API_URL}/api/model/current`);
        const data = await response.json();

        if (data.is_base_model) {
            modelElement.textContent = 'Base Model';
        } else {
            const version = data.model_path.split('-v').pop();
            modelElement.textContent = `Model v${version}`;
        }
    } catch (error) {
        modelElement.textContent = 'Unknown';
    }
}

// =========================================
// Chat Functions
// =========================================
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        askQuestion();
    }
}

function useSuggestion(chipElement) {
    const text = chipElement.textContent.trim();
    const input = document.getElementById('questionInput');
    input.value = text;
    askQuestion();
}

async function askQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();

    if (!question) return;

    if (!API_URL) {
        addMessage('error', 'API not configured. Please set the API URL.');
        return;
    }

    // Hide welcome screen on first message
    hideWelcome();

    // Add user message
    addMessage('user', question);
    input.value = '';
    input.focus();

    // Disable send button & show typing
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;
    showTyping(true);

    try {
        const response = await fetch(`${API_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        showTyping(false);
        sendBtn.disabled = false;

        if (data.answer) {
            addMessage('bot', data.answer);
        } else {
            addMessage('error', 'No response received from the API.');
        }

    } catch (error) {
        showTyping(false);
        sendBtn.disabled = false;
        addMessage('error', `Connection error: ${error.message}`);
    }
}

// =========================================
// Message Rendering
// =========================================
function addMessage(type, content) {
    const messagesDiv = document.getElementById('messages');
    const row = document.createElement('div');
    const id = 'msg-' + Date.now();

    row.id = id;
    row.className = `message-row ${type}`;

    // Build avatar
    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';

    if (type === 'bot') {
        avatar.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 8V4H8"/>
            <rect x="2" y="2" width="20" height="20" rx="5"/>
            <path d="M8 10a2 2 0 1 0 0 4"/>
            <path d="M16 10a2 2 0 1 1 0 4"/>
            <path d="M9 18h6"/>
        </svg>`;
    } else if (type === 'user') {
        avatar.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
        </svg>`;
    } else {
        avatar.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>`;
    }

    // Build bubble
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    const name = document.createElement('div');
    name.className = 'msg-name';
    name.textContent = type === 'user' ? 'You' : type === 'bot' ? 'AI' : 'Error';

    const text = document.createElement('div');
    text.className = 'msg-text';
    text.textContent = content;

    bubble.appendChild(name);
    bubble.appendChild(text);

    row.appendChild(avatar);
    row.appendChild(bubble);

    messagesDiv.appendChild(row);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    return id;
}

function removeMessage(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

// =========================================
// Welcome Screen
// =========================================
function hideWelcome() {
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) {
        welcome.style.opacity = '0';
        welcome.style.transform = 'translateY(-10px)';
        welcome.style.transition = 'all 0.3s ease';
        setTimeout(() => welcome.remove(), 300);
    }
}

// =========================================
// Typing Indicator
// =========================================
function showTyping(show) {
    const indicator = document.getElementById('typingIndicator');
    if (show) {
        indicator.classList.add('visible');
        // Scroll messages to bottom
        const messagesDiv = document.getElementById('messages');
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } else {
        indicator.classList.remove('visible');
    }
}

// =========================================
// Validate Functions (preserved for future use)
// =========================================
async function validateAnswer() {
    if (!API_URL) {
        showResult('validationResult', 'error', 'Please configure your API URL first.');
        return;
    }

    const question = document.getElementById('validateQuestion').value.trim();
    const answer = document.getElementById('validateAnswer').value.trim();

    if (!question || !answer) {
        showResult('validationResult', 'error', 'Please fill in both fields.');
        return;
    }

    showResult('validationResult', 'warning', 'Validating... This may take a moment.');

    try {
        const response = await fetch(`${API_URL}/api/validate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question, model_answer: answer })
        });

        const data = await response.json();

        if (data.error) {
            showResult('validationResult', 'error', `Validation Error: ${data.error}`);
            return;
        }

        const resultClass = data.is_outdated ? 'error' : 'success';
        const resultTitle = data.is_outdated ? 'Answer is Outdated' : 'Answer is Correct';

        showResult('validationResult', resultClass,
            `<h3>${resultTitle}</h3>
            <div><strong>Your Answer:</strong> ${data.model_answer}</div>
            <div><strong>Web Says:</strong> ${data.web_fact || 'N/A'}</div>
            <div><strong>Judge Decision:</strong> ${data.judge_decision || 'N/A'}</div>`
        );
    } catch (error) {
        showResult('validationResult', 'error', `Error: ${error.message}`);
    }
}

// =========================================
// Training Functions (preserved for future use)
// =========================================
function addTrainingFact() {
    const container = document.getElementById('trainingFacts');
    const factDiv = document.createElement('div');
    factDiv.className = 'training-fact';
    factDiv.innerHTML = `
        <div class="form-group">
            <label>Question:</label>
            <input type="text" class="train-question" placeholder="e.g., What is the latest iPhone?">
        </div>
        <div class="form-group">
            <label>Answer:</label>
            <input type="text" class="train-answer" placeholder="e.g., iPhone 16">
        </div>
        <div class="form-group">
            <label><input type="checkbox" class="train-stable"> This is a stable fact</label>
        </div>
    `;
    container.appendChild(factDiv);
}

async function startTraining() {
    if (!API_URL) {
        showResult('trainingResult', 'error', 'Please configure your API URL first.');
        return;
    }

    const trainingData = [];
    const facts = document.querySelectorAll('.training-fact');

    facts.forEach(fact => {
        const question = fact.querySelector('.train-question').value.trim();
        const answer = fact.querySelector('.train-answer').value.trim();
        const isStable = fact.querySelector('.train-stable').checked;

        if (question && answer) {
            trainingData.push({ question, answer, is_stable: isStable });
        }
    });

    if (trainingData.length === 0) {
        showResult('trainingResult', 'error', 'Please add at least one training fact.');
        return;
    }

    showResult('trainingResult', 'warning',
        `Starting training with ${trainingData.length} fact(s)...`);

    try {
        const response = await fetch(`${API_URL}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ training_data: trainingData })
        });

        const data = await response.json();

        if (data.status === 'training_started') {
            showResult('trainingResult', 'success',
                `Training started! Job ID: ${data.job_id} | Facts: ${trainingData.length}`);

            document.querySelectorAll('.train-question, .train-answer').forEach(input => {
                input.value = '';
            });
            document.querySelectorAll('.train-stable').forEach(checkbox => {
                checkbox.checked = false;
            });
        } else {
            showResult('trainingResult', 'error', 'Training failed to start.');
        }
    } catch (error) {
        showResult('trainingResult', 'error', `Error: ${error.message}`);
    }
}

// =========================================
// Settings (preserved for future use)
// =========================================
function loadSettings() {
    const savedUrl = localStorage.getItem('apiUrl');
    const urlInput = document.getElementById('apiUrl');
    if (urlInput) {
        urlInput.value = savedUrl || API_URL;
    }
    if (savedUrl) API_URL = savedUrl;
}

function saveSettings() {
    const urlInput = document.getElementById('apiUrl');
    if (!urlInput) return;

    const url = urlInput.value.trim();
    if (!url) {
        showResult('settingsResult', 'error', 'Please enter an API URL.');
        return;
    }

    API_URL = url.replace(/\/$/, '');
    localStorage.setItem('apiUrl', API_URL);
    showResult('settingsResult', 'success', 'Settings saved!');
    checkApiStatus();
    loadModelInfo();
}

function testConnection() {
    if (!API_URL) {
        showResult('settingsResult', 'error', 'Please enter and save an API URL first.');
        return;
    }

    showResult('settingsResult', 'warning', 'Testing connection...');

    fetch(`${API_URL}/api/health`)
        .then(response => response.json())
        .then(data => {
            showResult('settingsResult', 'success',
                `Connection successful! Service: ${data.service || 'Online'}`);
        })
        .catch(error => {
            showResult('settingsResult', 'error',
                `Connection failed: ${error.message}`);
        });
}

// =========================================
// Utilities
// =========================================
function showResult(elementId, type, content) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.className = `result-container show ${type}`;
    element.innerHTML = content;

    if (type === 'success') {
        setTimeout(() => {
            element.classList.remove('show');
        }, 10000);
    }
}

// =========================================
// Periodic Status Check
// =========================================
setInterval(() => {
    if (API_URL) {
        checkApiStatus();
    }
}, 30000);
