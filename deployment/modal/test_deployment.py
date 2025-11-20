"""
Quick Test Script for Modal Deployment
Test your deployed chatbot API
"""

import requests
import json
import sys


MODAL_URL = "https://your-username--active-learning-chatbot-fastapi-app.modal.run"

# Color codes for pretty output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text.center(80)}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_info(text):
    print(f"{YELLOW}ℹ {text}{RESET}")

def test_health_check():
    """Test 1: Health Check"""
    print_header("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{MODAL_URL}/")
        
        if response.status_code == 200:
            data = response.json()
            print_success("API is online!")
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Could not connect to API: {e}")
        print_info(f"Make sure MODAL_URL is correct: {MODAL_URL}")
        return False


def test_chat():
    """Test 2: Chat Endpoint"""
    print_header("TEST 2: Chat Endpoint")
    
    test_questions = [
        "What is the capital of France?",
        "What is 2+2?",
        "Who painted the Mona Lisa?",
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {YELLOW}{question}{RESET}")
        
        try:
            response = requests.post(
                f"{MODAL_URL}/chat",
                json={"question": question},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success("Response received!")
                print(f"   Answer: {data.get('answer')}")
                print(f"   Model: {data.get('model_version')}")
            else:
                print_error(f"Request failed with status {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print_error(f"Request failed: {e}")
    
    return True


def test_current_model():
    """Test 3: Get Current Model"""
    print_header("TEST 3: Current Model Info")
    
    try:
        response = requests.get(f"{MODAL_URL}/model/current")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Model info retrieved!")
            print(f"   Model Path: {data.get('model_path')}")
            print(f"   Is Base Model: {data.get('is_base_model')}")
            return True
        else:
            print_error(f"Request failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def test_validation():
    """Test 4: Validation Endpoint"""
    print_header("TEST 4: Answer Validation")
    
    test_cases = [
        {
            "question": "What is the capital of France?",
            "model_answer": "Paris",
            "expected": "not outdated"
        },
        {
            "question": "What is the capital of France?",
            "model_answer": "London",
            "expected": "outdated"
        }
    ]
    
    for case in test_cases:
        print(f"\n❓ Question: {case['question']}")
        print(f"   Model Answer: {case['model_answer']}")
        print(f"   Expected: {case['expected']}")
        
        try:
            response = requests.post(
                f"{MODAL_URL}/validate",
                json={
                    "question": case['question'],
                    "model_answer": case['model_answer']
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'error' in data:
                    print_error(f"Validation error: {data['error']}")
                    continue
                
                is_outdated = data.get('is_outdated', False)
                result = "outdated" if is_outdated else "not outdated"
                
                if result == case['expected']:
                    print_success(f"Validation correct: {result}")
                else:
                    print_error(f"Validation unexpected: {result}")
                
                print(f"   Web Fact: {data.get('web_fact')}")
                print(f"   Judge Decision: {data.get('judge_decision')}")
            else:
                print_error(f"Request failed with status {response.status_code}")
        except Exception as e:
            print_error(f"Request failed: {e}")
    
    return True


def test_training():
    """Test 5: Training Endpoint (trigger only)"""
    print_header("TEST 5: Training Trigger")
    
    print_info("This test only triggers training, it doesn't wait for completion.")
    print_info("Training runs in the background and may take 10-30 minutes.")
    
    training_data = {
        "training_data": [
            {
                "question": "What is the test answer?",
                "answer": "This is a test training run.",
                "is_stable": False
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{MODAL_URL}/train",
            json=training_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Training job started!")
            print(f"   Status: {data.get('status')}")
            print(f"   Job ID: {data.get('job_id')}")
            print_info("Check Modal logs to monitor training progress:")
            print(f"   modal app logs active-learning-chatbot")
            return True
        else:
            print_error(f"Request failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def main():
    """Run all tests"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{'MODAL DEPLOYMENT TEST SUITE'.center(80)}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    
    print(f"\n{YELLOW}Testing API at: {MODAL_URL}{RESET}\n")
    
    # Check if URL is configured
    if "your-username" in MODAL_URL:
        print_error("MODAL_URL not configured!")
        print_info("Edit this file and replace MODAL_URL with your actual Modal URL.")
        print_info("Get it from: modal deploy modal_app.py")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Chat Endpoint", test_chat),
        ("Current Model", test_current_model),
        ("Validation", test_validation),
        ("Training Trigger", test_training),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Tests interrupted by user.{RESET}")
            break
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{YELLOW}Results: {passed}/{total} tests passed{RESET}\n")
    
    if passed == total:
        print(f"{GREEN}All tests passed! Your API is working correctly.{RESET}\n")
    else:
        print(f"{RED}Some tests failed. Check the output above for details.{RESET}\n")


if __name__ == "__main__":
    main()
