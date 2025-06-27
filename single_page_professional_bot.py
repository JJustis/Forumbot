#!/usr/bin/env python3
"""
Professional Forum Bot - Single Page Solution
Complete production-ready forum bot with SSL support and /talk endpoint

Usage:
    python professional_forum_bot.py

Endpoint:
    GET /talk?ack=your_question_here

Features:
- Fine-tuned TinyLlama model
- Professional response generation
- SSL support
- Response caching
- Quality control
- Error handling
- Performance monitoring
"""

import os
import ssl
import json
import time
import logging
import hashlib
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from flask import Flask, request, jsonify
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forum_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('professional_forum_bot')

class ProfessionalForumBot:
    """
    Professional-grade forum bot with enterprise features
    Single-file solution for production deployment
    """
    
    def __init__(self):
        self.model_path = "./tinyllama-forum-professional"
        self.fallback_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.max_length = 800
        self.temperature = 0.8
        
        # Thread safety
        self.model_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'error_count': 0,
            'start_time': datetime.now()
        }
        
        # Simple in-memory cache (production would use Redis)
        self.response_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Initialize components
        self.setup_database()
        self.setup_model()
        self.create_training_data()
        
        logger.info("Professional Forum Bot initialized successfully")
    
    def setup_database(self):
        """Setup SQLite database for analytics and feedback"""
        self.db_path = "forum_bot.db"
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db_lock = threading.Lock()
        
        cursor = self.db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_query TEXT,
                bot_response TEXT,
                response_time REAL,
                confidence REAL,
                cached BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                rating INTEGER,
                feedback_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db.commit()
        logger.info("Database initialized")
    
    def create_training_data(self):
        """Create high-quality training dataset"""
        if Path(self.model_path).exists():
            logger.info("Fine-tuned model already exists")
            return
        
        # Professional training conversations
        training_conversations = [
            {
                "user": "I'm new to Python and confused about variable scope. Can you explain global vs local variables?",
                "assistant": "Great question! Variable scope is fundamental to understand. Let me break it down:\n\n**Local Variables:**\nThese exist only inside the function where they're created.\n\n```python\ndef my_function():\n    local_var = \"I'm local\"\n    print(local_var)  # This works\n\nmy_function()\n# print(local_var)  # This would cause an error!\n```\n\n**Global Variables:**\nThese are defined outside functions and accessible everywhere.\n\n```python\nglobal_var = \"I'm global\"\n\ndef another_function():\n    print(global_var)  # Can access global variable\n    \n    # To modify a global variable:\n    global global_var\n    global_var = \"Modified\"\n\nanother_function()\nprint(global_var)  # Prints: Modified\n```\n\n**Best Practice:**\nPass values as parameters instead of using globals:\n\n```python\ndef process_data(data):\n    return f\"Processed: {data}\"\n\nresult = process_data(\"my data\")\n```\n\nThis makes your code more predictable and easier to debug!"
            },
            {
                "user": "My Flask app keeps throwing 500 errors. How do I debug this effectively?",
                "assistant": "Flask 500 errors can be tricky! Here's a systematic debugging approach:\n\n**1. Enable Debug Mode (Development Only)**\n```python\napp = Flask(__name__)\napp.debug = True  # Shows detailed error pages\napp.run(debug=True)\n```\n\n**2. Add Comprehensive Logging**\n```python\nimport logging\n\nlogging.basicConfig(level=logging.DEBUG)\napp.logger.setLevel(logging.DEBUG)\n\n@app.route('/test')\ndef test_route():\n    try:\n        app.logger.info('Processing request...')\n        # Your code here\n        return 'Success'\n    except Exception as e:\n        app.logger.error(f'Error: {str(e)}')\n        raise\n```\n\n**3. Use Error Handlers**\n```python\n@app.errorhandler(500)\ndef internal_error(error):\n    app.logger.error(f'Server Error: {error}')\n    return 'Internal Server Error', 500\n```\n\n**4. Check Common Issues:**\n- Database connection problems\n- Missing environment variables\n- Import errors\n- Template rendering issues\n\n**5. Test Incrementally**\nComment out sections and gradually add them back to isolate the problem.\n\nWhat specific functionality is your Flask app handling? I can provide more targeted debugging advice."
            },
            {
                "user": "I've been learning to code for 8 months but feel overwhelmed. How do I prioritize what to learn next?",
                "assistant": "That feeling is totally normal! The programming world is vast. Here's how to prioritize effectively:\n\n**Focus Strategy - Pick One Path:**\n\n**Web Development:**\n1. Master one language well (Python or JavaScript)\n2. Learn HTML/CSS fundamentals\n3. Pick ONE framework (React, Django, or Flask)\n4. Basic database skills (SQL)\n5. Deployment basics (Heroku, Netlify)\n\n**Data Science:**\n1. Python fundamentals\n2. Pandas and NumPy\n3. Data visualization (matplotlib)\n4. Basic statistics\n5. One ML library (scikit-learn)\n\n**The 70-20-10 Rule:**\n- 70% time: Deepening current skills\n- 20% time: Learning related skills\n- 10% time: Exploring new tech\n\n**Practical Tips:**\n- Build projects that combine multiple skills\n- Don't learn new frameworks until you need them\n- Better to know one thing deeply than five things superficially\n- Every expert was once a beginner feeling overwhelmed\n\n**Key Mindset:**\nConsistent progress beats perfect knowledge. Focus on solving real problems, not collecting certificates.\n\nWhat type of programming interests you most? That'll help narrow your next steps."
            },
            {
                "user": "What's the difference between GET, POST, PUT, and DELETE in REST APIs?",
                "assistant": "Excellent question! Understanding HTTP methods is crucial for good API design:\n\n**GET - Retrieve Data**\n- Purpose: Fetch information without changing anything\n- Safe: Yes (no side effects)\n- Idempotent: Yes (same result every time)\n\n```python\n@app.route('/api/users', methods=['GET'])\ndef get_users():\n    return jsonify(users)\n\n@app.route('/api/users/<int:user_id>', methods=['GET'])\ndef get_user(user_id):\n    user = find_user_by_id(user_id)\n    return jsonify(user) if user else ('Not found', 404)\n```\n\n**POST - Create New Resources**\n- Purpose: Create something new\n- Safe: No (creates side effects)\n- Idempotent: No (multiple calls create multiple items)\n\n```python\n@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.get_json()\n    new_user = create_user_from_data(data)\n    return jsonify(new_user), 201\n```\n\n**PUT - Update/Replace Entire Resource**\n- Purpose: Update entire resource (or create if doesn't exist)\n- Idempotent: Yes\n\n```python\n@app.route('/api/users/<int:user_id>', methods=['PUT'])\ndef update_user(user_id):\n    data = request.get_json()\n    user = update_or_create_user(user_id, data)\n    return jsonify(user)\n```\n\n**DELETE - Remove Resource**\n- Purpose: Delete a resource\n- Idempotent: Yes\n\n```python\n@app.route('/api/users/<int:user_id>', methods=['DELETE'])\ndef delete_user(user_id):\n    if delete_user_by_id(user_id):\n        return '', 204\n    return 'Not found', 404\n```\n\n**RESTful URL Patterns:**\n```\nGET    /api/users          # Get all users\nGET    /api/users/123      # Get specific user\nPOST   /api/users          # Create new user\nPUT    /api/users/123      # Update entire user\nDELETE /api/users/123      # Delete user\n```\n\nKey principle: Use nouns in URLs, not verbs!"
            },
            {
                "user": "How do I handle errors gracefully in Python?",
                "assistant": "Great question! Proper error handling makes your code robust and user-friendly. Here's how to do it right:\n\n**Basic Try-Except Structure:**\n```python\ntry:\n    # Code that might raise an exception\n    result = risky_operation()\nexcept SpecificError as e:\n    # Handle specific error\n    print(f\"Specific error occurred: {e}\")\nexcept Exception as e:\n    # Handle any other error\n    print(f\"Unexpected error: {e}\")\nelse:\n    # Runs if no exception occurred\n    print(\"Operation successful\")\nfinally:\n    # Always runs, regardless of exceptions\n    cleanup_resources()\n```\n\n**Specific Exception Handling:**\n```python\ndef safe_divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return \"Cannot divide by zero\"\n    except TypeError:\n        return \"Invalid input types\"\n\ndef safe_file_read(filename):\n    try:\n        with open(filename, 'r') as file:\n            return file.read()\n    except FileNotFoundError:\n        return \"File not found\"\n    except PermissionError:\n        return \"Permission denied\"\n    except Exception as e:\n        return f\"Error reading file: {e}\"\n```\n\n**Custom Exceptions:**\n```python\nclass ValidationError(Exception):\n    \"\"\"Custom exception for validation errors\"\"\"\n    pass\n\ndef validate_email(email):\n    if '@' not in email:\n        raise ValidationError(\"Invalid email format\")\n    return True\n\ntry:\n    validate_email(\"invalid-email\")\nexcept ValidationError as e:\n    print(f\"Validation failed: {e}\")\n```\n\n**Best Practices:**\n1. **Be Specific**: Catch specific exceptions rather than using bare `except:`\n2. **Log Errors**: Use logging module for debugging\n3. **Graceful Degradation**: Provide fallback behavior\n4. **User-Friendly Messages**: Don't expose technical details to users\n5. **Clean Up**: Use `finally` or context managers for resource cleanup\n\n**Logging Example:**\n```python\nimport logging\n\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\n\ndef process_data(data):\n    try:\n        # Process data\n        result = complex_operation(data)\n        logger.info(\"Data processed successfully\")\n        return result\n    except ValueError as e:\n        logger.error(f\"Invalid data: {e}\")\n        return None\n    except Exception as e:\n        logger.exception(\"Unexpected error in process_data\")\n        raise\n```\n\nRemember: exceptions should be exceptional! Use them for error conditions, not normal program flow."
            },
            {
                "user": "What are Python list comprehensions and when should I use them?",
                "assistant": "List comprehensions are a concise way to create lists in Python! They're powerful and Pythonic when used correctly.\n\n**Basic Syntax:**\n```python\n# Traditional approach\nnumbers = []\nfor i in range(10):\n    if i % 2 == 0:\n        numbers.append(i * 2)\n\n# List comprehension\nnumbers = [i * 2 for i in range(10) if i % 2 == 0]\n# Result: [0, 4, 8, 12, 16]\n```\n\n**Common Patterns:**\n\n**1. Transform each element:**\n```python\n# Square all numbers\nsquares = [x**2 for x in range(5)]\n# [0, 1, 4, 9, 16]\n\n# Convert strings to uppercase\nwords = ['hello', 'world', 'python']\nupper_words = [word.upper() for word in words]\n# ['HELLO', 'WORLD', 'PYTHON']\n```\n\n**2. Filter with conditions:**\n```python\n# Even numbers only\neven_nums = [x for x in range(20) if x % 2 == 0]\n\n# Words longer than 3 characters\nlong_words = [word for word in words if len(word) > 3]\n```\n\n**3. Nested loops:**\n```python\n# Multiplication table\ntable = [(i, j, i*j) for i in range(1, 4) for j in range(1, 4)]\n# [(1,1,1), (1,2,2), (1,3,3), (2,1,2), ...]\n\n# Flatten nested lists\nnested = [[1, 2], [3, 4], [5, 6]]\nflat = [item for sublist in nested for item in sublist]\n# [1, 2, 3, 4, 5, 6]\n```\n\n**4. Dictionary and Set Comprehensions:**\n```python\n# Dictionary comprehension\nsquare_dict = {x: x**2 for x in range(5)}\n# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}\n\n# Set comprehension\nunique_lengths = {len(word) for word in words}\n# {5, 6} (unique lengths)\n```\n\n**When to Use List Comprehensions:**\n✅ **Good for:**\n- Simple transformations\n- Filtering lists\n- Creating lists from iterables\n- When readability isn't sacrificed\n\n❌ **Avoid when:**\n- Logic is complex (use regular loops)\n- Side effects are needed (printing, file operations)\n- Multiple conditions make it hard to read\n\n**Performance Tip:**\nList comprehensions are generally faster than equivalent for loops:\n\n```python\n# Faster\nsquares = [x**2 for x in range(1000)]\n\n# Slower\nsquares = []\nfor x in range(1000):\n    squares.append(x**2)\n```\n\n**Keep it Readable:**\n```python\n# Good - readable\neven_squares = [x**2 for x in range(10) if x % 2 == 0]\n\n# Bad - too complex\nresult = [f(x) for x in data if complex_condition(x) and another_condition(x) or third_condition(x)]\n\n# Better - use regular loop for complex logic\nresult = []\nfor x in data:\n    if complex_condition(x) and another_condition(x) or third_condition(x):\n        result.append(f(x))\n```\n\nRule of thumb: If it fits on one line and is easy to understand, use a list comprehension!"
            },
            {
                "user": "Thanks for the help! This is really useful.",
                "assistant": "You're absolutely welcome! I'm really glad I could help clarify things for you. It's always satisfying to see someone working through programming concepts and getting those \"aha\" moments.\n\nFeel free to come back anytime if you run into more questions or want to dive deeper into any topic. Whether it's debugging tricky code, understanding new concepts, or just brainstorming approaches to problems, I'm here to help.\n\nKeep coding and don't hesitate to experiment - that's how the best learning happens! Good luck with your projects! 🚀"
            }
        ]
        
        # Save training data
        Path("training_data").mkdir(exist_ok=True)
        with open("training_data/conversations.json", "w") as f:
            json.dump(training_conversations, f, indent=2)
        
        logger.info(f"Created training dataset with {len(training_conversations)} conversations")
    
    def setup_model(self):
        """Load or train the model"""
        try:
            if Path(self.model_path).exists():
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.model_version = "fine-tuned"
            else:
                logger.info("Fine-tuned model not found, training new model...")
                self.train_model()
                
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned model: {e}")
            logger.info("Falling back to base model")
            self.load_base_model()
    
    def load_base_model(self):
        """Load base TinyLlama model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.fallback_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model_version = "base"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        logger.info("Base model loaded successfully")
    
    def train_model(self):
        """Train the model with professional data"""
        try:
            # Load training data
            with open("training_data/conversations.json", "r") as f:
                conversations = json.load(f)
            
            # Load base model for training
            logger.info("Loading base model for training...")
            tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.fallback_model,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use FP32 for training
                low_cpu_mem_usage=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
            
            # Format training data
            training_texts = []
            for conv in conversations:
                formatted = f"<|system|>\nYou are an expert programming mentor and friendly forum moderator. You provide accurate, helpful, and engaging responses to programming questions and general forum discussions. You explain concepts clearly, provide working code examples, and encourage learning.\n</s>\n<|user|>\n{conv['user']}\n</s>\n<|assistant|>\n{conv['assistant']}\n</s>"
                training_texts.append(formatted)
            
            # Tokenize
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=1024,
                    padding="max_length",
                    return_tensors="pt"
                )
            
            # Create dataset
            dataset = Dataset.from_dict({"text": training_texts})
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.model_path,
                overwrite_output_dir=True,
                num_train_epochs=2,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=2e-5,
                warmup_steps=10,
                logging_steps=5,
                save_strategy="epoch",
                fp16=False,
                dataloader_drop_last=True,
                report_to="none",
                max_grad_norm=1.0,
                weight_decay=0.01,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )
            
            # Train
            logger.info("Starting model training...")
            trainer.train()
            
            # Save
            trainer.save_model()
            tokenizer.save_pretrained(self.model_path)
            
            logger.info(f"Model training completed and saved to {self.model_path}")
            
            # Load the trained model
            self.tokenizer = tokenizer
            self.model = model
            self.model_version = "fine-tuned"
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.info("Falling back to base model")
            self.load_base_model()
    
    def clean_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            self.response_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(f"{query}_{self.temperature}_{self.max_length}".encode()).hexdigest()
    
    def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response if available"""
        cache_key = self.get_cache_key(query)
        current_time = time.time()
        
        if cache_key in self.response_cache:
            if current_time - self.cache_timestamps[cache_key] < self.cache_ttl:
                self.metrics['cache_hits'] += 1
                return self.response_cache[cache_key]
            else:
                # Expired
                self.response_cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
        
        return None
    
    def cache_response(self, query: str, response: str):
        """Cache response"""
        cache_key = self.get_cache_key(query)
        self.response_cache[cache_key] = response
        self.cache_timestamps[cache_key] = time.time()
        
        # Clean cache periodically
        if len(self.response_cache) > 100:
            self.clean_cache()
    
    def calculate_confidence(self, response: str, query: str) -> float:
        """Calculate confidence score for response"""
        factors = {
            'length': min(len(response.split()) / 50, 1.0) * 0.3,
            'code_presence': 0.3 if '```' in response else 0,
            'structure': 0.2 if '\n\n' in response else 0.1,
            'completeness': 0.2 if not response.strip().endswith('...') else 0
        }
        
        return min(sum(factors.values()), 1.0)
    
    def clean_response(self, response: str) -> str:
        """Clean and post-process response"""
        # Remove chat tokens
        response = re.sub(r'<\|[^|]*\|>', '', response)
        response = response.replace('</s>', '').replace('<s>', '')
        
        # Remove repetitive patterns
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line[:30] not in seen_lines:  # Check first 30 chars
                cleaned_lines.append(line)
                seen_lines.add(line[:30])
        
        response = '\n'.join(cleaned_lines).strip()
        
        # Ensure clean ending
        if response and not response[-1] in '.!?':
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response
    
    def generate_response(self, query: str) -> Dict:
        """Generate response to user query"""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            # Check cache first
            cached_response = self.get_cached_response(query)
            if cached_response:
                processing_time = time.time() - start_time
                confidence = 0.9  # Cached responses are high confidence
                
                return {
                    'response': cached_response,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'cached': True,
                    'model_version': self.model_version
                }
            
            # Generate new response
            with self.model_lock:
                # Format prompt
                formatted_prompt = f"<|system|>\nYou are an expert programming mentor and friendly forum moderator. You provide accurate, helpful, and engaging responses to programming questions and general forum discussions. You explain concepts clearly, provide working code examples, and encourage learning.\n</s>\n<|user|>\n{query}\n</s>\n<|assistant|>\n"
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=800
                )
                
                # Move to device
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                # Decode
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract assistant response
                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1].strip()
                else:
                    response = response[len(formatted_prompt):].strip()
            
            # Clean response
            response = self.clean_response(response)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            confidence = self.calculate_confidence(response, query)
            
            # Cache response
            self.cache_response(query, response)
            
            # Log to database
            with self.db_lock:
                cursor = self.db.cursor()
                cursor.execute("""
                    INSERT INTO conversations (user_query, bot_response, response_time, confidence, cached)
                    VALUES (?, ?, ?, ?, ?)
                """, (query, response, processing_time, confidence, False))
                self.db.commit()
            
            # Update metrics
            self.metrics['avg_response_time'] = (
                self.metrics['avg_response_time'] * (self.metrics['total_requests'] - 1) + 
                processing_time
            ) / self.metrics['total_requests']
            
            return {
                'response': response,
                'confidence': confidence,
                'processing_time': processing_time,
                'cached': False,
                'model_version': self.model_version
            }
            
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Error generating response: {e}")
            
            return {
                'response': "I'm experiencing some technical difficulties right now. Could you try rephrasing your question?",
                'confidence': 0.1,
                'processing_time': time.time() - start_time,
                'cached': False,
                'model_version': self.model_version,
                'error': str(e)
            }
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        uptime = datetime.now() - self.metrics['start_time']
        cache_hit_rate = 0
        if self.metrics['total_requests'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / self.metrics['total_requests']
        
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_hit_rate': f"{cache_hit_rate:.2%}",
            'avg_response_time': f"{self.metrics['avg_response_time']:.3f}s",
            'error_count': self.metrics['error_count'],
            'uptime': str(uptime),
            'model_version': self.model_version,
            'cached_responses': len(self.response_cache)
        }

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize bot
logger.info("Initializing Professional Forum Bot...")
forum_bot = ProfessionalForumBot()

@app.route('/talk', methods=['GET'])
def talk():
    """Main endpoint - compatible with existing format"""
    ack = request.args.get('ack', '').strip()
    
    if not ack:
        return jsonify({
            'error': 'Missing ?ack= parameter',
            'usage': 'GET /talk?ack=your_question_here'
        }), 400
    
    # Generate response
    result = forum_bot.generate_response(ack)
    
    # Format response to match expected format
    return jsonify({
        'input': ack,
        'response': result['response'],
        'meta': {
            'processing_time': f"{result['processing_time']:.3f}s",
            'confidence': f"{result['confidence']:.2f}",
            'cached': result['cached'],
            'model_version': result['model_version']
        }
    })

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint"""
    return jsonify({
        'status': 'operational',
        'bot_ready': True,
        'metrics': forum_bot.get_metrics()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Metrics endpoint"""
    return jsonify(forum_bot.get_metrics())

# CORS support
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/talk', methods=['OPTIONS'])
def options():
    return "", 200

if __name__ == "__main__":
    logger.info("Starting Professional Forum Bot Server...")
    
    # SSL Certificate paths
    cert_path = r'C:\portablexampp\apache\conf\ssl.crt\certificate.crt'
    key_path = r'C:\portablexampp\apache\conf\ssl.key\private.key'
    
    # Server configuration
    host = "0.0.0.0"
    port = 8043
    
    logger.info(f"Bot initialized with model version: {forum_bot.model_version}")
    logger.info(f"Cached responses: {len(forum_bot.response_cache)}")
    logger.info(f"Server starting on {host}:{port}")
    
    # Start server with SSL if certificates exist
    if os.path.exists(cert_path) and os.path.exists(key_path):
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert_path, key_path)
            logger.info("Starting HTTPS server with SSL certificates")
            app.run(host=host, port=port, ssl_context=context, debug=False, threaded=True)
        except Exception as e:
            logger.error(f"SSL error: {e}, falling back to HTTP")
            logger.info("Starting HTTP server")
            app.run(host=host, port=port, debug=False, threaded=True)
    else:
        logger.info("SSL certificates not found, starting HTTP server")
        app.run(host=host, port=port, debug=False, threaded=True)