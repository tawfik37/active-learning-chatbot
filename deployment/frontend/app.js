// Configuration - Auto-detect API URL
let API_URL = window.location.origin;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadSettings();
    checkApiStatus();
    loadModelInfo();
    
    // If running on Modal, API_URL is already correct
    // If running locally, user needs to set it
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        // Running locally - check if API URL is set
        const savedUrl = localStorage.getItem('apiUrl');
        if (!savedUrl) {
            showResult('settingsResult', 'warning', 
                'Running locally. Please configure your Modal API URL in Settings.');
        }
    } else {
        // Running on Modal - save current URL
        localStorage.setItem('apiUrl', API_URL);
    }
});

// Tab Switching
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    document.getElementById(tabName + 'Tab').classList.add('active');
    event.target.classList.add('active');
}

// Settings Management
function loadSettings() {
    const savedUrl = localStorage.getItem('apiUrl');
    if (savedUrl) {
        document.getElementById('apiUrl').value = savedUrl;
        API_URL = savedUrl;
    } else {
        // Auto-detect if on Modal
        document.getElementById('apiUrl').value = API_URL;
    }
}

function saveSettings() {
    const url = document.getElementById('apiUrl').value.trim();
    
    if (!url) {
        showResult('settingsResult', 'error', 'Please enter an API URL');
        return;
    }
    
    API_URL = url.replace(/\/$/, '');
    localStorage.setItem('apiUrl', API_URL);
    
    showResult('settingsResult', 'success', 'Settings saved successfully!');
    
    checkApiStatus();
    loadModelInfo();
}

function testConnection() {
    if (!API_URL) {
        showResult('settingsResult', 'error', 'Please enter and save an API URL first');
        return;
    }
    
    showResult('settingsResult', 'warning', 'Testing connection...');
    
    fetch(`${API_URL}/api/health`)
        .then(response => response.json())
        .then(data => {
            showResult('settingsResult', 'success', 
                `Connection successful! Service: ${data.service || 'Unknown'}`);
        })
        .catch(error => {
            showResult('settingsResult', 'error', 
                `Connection failed: ${error.message}`);
        });
}

// API Status Check
async function checkApiStatus() {
    const statusElement = document.getElementById('apiStatus');
    
    if (!API_URL) {
        statusElement.textContent = 'Not configured';
        statusElement.className = 'status-value offline';
        return;
    }
    
    statusElement.textContent = 'Checking...';
    
    try {
        const response = await fetch(`${API_URL}/api/health`);
        const data = await response.json();
        
        if (data.status === 'online') {
            statusElement.textContent = '🟢 Online';
            statusElement.className = 'status-value online';
        } else {
            statusElement.textContent = '🔴 Offline';
            statusElement.className = 'status-value offline';
        }
    } catch (error) {
        statusElement.textContent = '🔴 Offline';
        statusElement.className = 'status-value offline';
    }
}

// Load Model Info
async function loadModelInfo() {
    const modelElement = document.getElementById('modelVersion');
    
    if (!API_URL) {
        modelElement.textContent = 'Not configured';
        return;
    }
    
    modelElement.textContent = 'Loading...';
    
    try {
        const response = await fetch(`${API_URL}/api/model/current`);
        const data = await response.json();
        
        if (data.is_base_model) {
            modelElement.textContent = 'Base Model';
        } else {
            const version = data.model_path.split('-v').pop();
            modelElement.textContent = `v${version}`;
        }
    } catch (error) {
        modelElement.textContent = 'Unknown';
    }
}

// Chat Functions
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        askQuestion();
    }
}

async function askQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    
    if (!question) return;
    
    if (!API_URL) {
        addMessage('error', 'Please configure your API URL in Settings first!');
        return;
    }
    
    addMessage('user', question);
    input.value = '';
    
    const loadingId = addMessage('bot', '<span class="loading"></span> Thinking...');
    
    try {
        const response = await fetch(`${API_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });
        
        const data = await response.json();
        
        removeMessage(loadingId);
        
        if (data.answer) {
            addMessage('bot', `<strong>Bot:</strong> ${data.answer}`);
        } else {
            addMessage('error', 'No response received from the API');
        }
        
    } catch (error) {
        removeMessage(loadingId);
        addMessage('error', `Error: ${error.message}`);
    }
}

function addMessage(type, content) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    const id = 'msg-' + Date.now();
    
    messageDiv.id = id;
    messageDiv.className = `message ${type}`;
    messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    return id;
}

function removeMessage(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

// Validate Functions
async function validateAnswer() {
    if (!API_URL) {
        showResult('validationResult', 'error', 'Please configure your API URL in Settings first!');
        return;
    }
    
    const question = document.getElementById('validateQuestion').value.trim();
    const answer = document.getElementById('validateAnswer').value.trim();
    
    if (!question || !answer) {
        showResult('validationResult', 'error', 'Please fill in both fields');
        return;
    }
    
    showResult('validationResult', 'warning', 'Validating... This may take a minute.');
    
    try {
        const response = await fetch(`${API_URL}/api/validate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                model_answer: answer
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showResult('validationResult', 'error', 
                `<h3>Validation Error</h3>
                <p>${data.error}</p>`);
            return;
        }
        
        const resultClass = data.is_outdated ? 'error' : 'success';
        const resultTitle = data.is_outdated ? '❌ Answer is Outdated' : '✅ Answer is Correct';
        
        showResult('validationResult', resultClass,
            `<h3>${resultTitle}</h3>
            <div class="result-item"><strong>Your Answer:</strong> ${data.model_answer}</div>
            <div class="result-item"><strong>Web Says:</strong> ${data.web_fact || 'N/A'}</div>
            <div class="result-item"><strong>Judge Decision:</strong> ${data.judge_decision || 'N/A'}</div>`
        );
        
    } catch (error) {
        showResult('validationResult', 'error', 
            `<h3>Error</h3><p>${error.message}</p>`);
    }
}

// Training Functions
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
            <label>
                <input type="checkbox" class="train-stable">
                This is a stable fact (won't change)
            </label>
        </div>
    `;
    container.appendChild(factDiv);
}

async function startTraining() {
    if (!API_URL) {
        showResult('trainingResult', 'error', 'Please configure your API URL in Settings first!');
        return;
    }
    
    const trainingData = [];
    const facts = document.querySelectorAll('.training-fact');
    
    facts.forEach(fact => {
        const question = fact.querySelector('.train-question').value.trim();
        const answer = fact.querySelector('.train-answer').value.trim();
        const isStable = fact.querySelector('.train-stable').checked;
        
        if (question && answer) {
            trainingData.push({
                question: question,
                answer: answer,
                is_stable: isStable
            });
        }
    });
    
    if (trainingData.length === 0) {
        showResult('trainingResult', 'error', 'Please add at least one training fact');
        return;
    }
    
    showResult('trainingResult', 'warning', 
        `Starting training with ${trainingData.length} fact(s)... This will take 10-30 minutes.`);
    
    try {
        const response = await fetch(`${API_URL}/api/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                training_data: trainingData
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'training_started') {
            showResult('trainingResult', 'success',
                `<h3>✅ Training Started!</h3>
                <div class="result-item"><strong>Job ID:</strong> ${data.job_id}</div>
                <div class="result-item"><strong>Facts:</strong> ${trainingData.length}</div>
                <p class="mt-2">Training is running in the background. Check back in 10-30 minutes.</p>`
            );
            
            document.querySelectorAll('.train-question, .train-answer').forEach(input => {
                input.value = '';
            });
            document.querySelectorAll('.train-stable').forEach(checkbox => {
                checkbox.checked = false;
            });
        } else {
            showResult('trainingResult', 'error', 'Training failed to start');
        }
        
    } catch (error) {
        showResult('trainingResult', 'error', 
            `<h3>Error</h3><p>${error.message}</p>`);
    }
}

// Utility Functions
function showResult(elementId, type, content) {
    const element = document.getElementById(elementId);
    element.className = `result-container show ${type}`;
    element.innerHTML = content;
    
    if (type === 'success') {
        setTimeout(() => {
            element.classList.remove('show');
        }, 10000);
    }
}

// Refresh status every 30 seconds
setInterval(() => {
    if (API_URL) {
        checkApiStatus();
    }
}, 30000);