#!/usr/bin/env python3
"""
Enhanced Professional Forum Bot - Multiple Fine-tuned Datasets
Complete production-ready forum bot with SSL support and comprehensive training data

Usage:
    python enhanced_professional_bot.py

Endpoint:
    GET /talk?ack=your_question_here

Features:
- Fine-tuned TinyLlama model with multiple datasets
- Alpaca cleaned dataset (52K instructions)
- CodeAlpaca dataset (20K programming instructions)
- Evol-Instruct-Code dataset (80K evolved instructions)
- GitHub Python scripts collection
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
import requests
import shutil
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
import random

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from flask import Flask, request, jsonify
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_forum_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('enhanced_professional_forum_bot')

class GitHubDataCollector:
    """Collects Python scripts from GitHub for training"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Enhanced-Professional-Bot'
        }
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'
    
    def search_python_repos(self, query: str = "python tutorial", per_page: int = 10) -> List[Dict]:
        """Search for Python repositories"""
        url = "https://api.github.com/search/repositories"
        params = {
            'q': f'{query} language:python',
            'sort': 'stars',
            'order': 'desc',
            'per_page': per_page
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get('items', [])
        except Exception as e:
            logger.error(f"Error searching repositories: {e}")
            return []
    
    def get_python_files(self, repo_full_name: str, max_files: int = 5) -> List[Dict]:
        """Get Python files from a repository"""
        url = f"https://api.github.com/repos/{repo_full_name}/contents"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            contents = response.json()
            
            python_files = []
            for item in contents:
                if item['type'] == 'file' and item['name'].endswith('.py'):
                    python_files.append(item)
                    if len(python_files) >= max_files:
                        break
            
            return python_files
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403 and "rate limit" in str(e).lower():
                logger.warning(f"Rate limit reached for {repo_full_name}")
                return []  # Return empty list instead of logging as error
            else:
                logger.error(f"HTTP error getting files from {repo_full_name}: {e}")
                return []
        except Exception as e:
            logger.error(f"Error getting files from {repo_full_name}: {e}")
            return []
    
    def get_file_content(self, download_url: str) -> Optional[str]:
        """Download file content"""
        try:
            response = requests.get(download_url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None
    
    def collect_github_scripts(self, target_count: int = 100) -> List[Dict]:
        """Collect Python scripts for training"""
        logger.info(f"Collecting {target_count} Python scripts from GitHub...")
        
        training_data = []
        collected_count = 0
        rate_limit_hit = False
        
        # Diverse search queries for high-quality Python content
        search_queries = [
            "python tutorial beginner",
            "python data analysis",
            "python web scraping",
            "python automation scripts",
            "python api examples",
            "python machine learning tutorial",
            "python flask examples",
            "python algorithms implementation",
            "python best practices",
            "python utilities tools"
        ]
        
        for query in search_queries:
            if collected_count >= target_count or rate_limit_hit:
                break
                
            repos = self.search_python_repos(query, per_page=5)
            
            for repo in repos:
                if collected_count >= target_count or rate_limit_hit:
                    break
                
                python_files = self.get_python_files(repo['full_name'], max_files=2)
                
                if not python_files and "rate limit" in str(python_files).lower():
                    rate_limit_hit = True
                    logger.warning("GitHub API rate limit reached, stopping collection")
                    break
                
                for file_info in python_files:
                    if collected_count >= target_count:
                        break
                    
                    content = self.get_file_content(file_info['download_url'])
                    if content and 200 <= len(content) <= 3000:  # Good size for learning
                        training_data.append({
                            'repo_name': repo['full_name'],
                            'file_name': file_info['name'],
                            'content': content,
                            'description': repo.get('description', ''),
                            'stars': repo.get('stargazers_count', 0)
                        })
                        collected_count += 1
                        logger.info(f"Collected {collected_count}/{target_count}: {repo['full_name']}/{file_info['name']}")
                
                # Rate limiting - be more conservative
                time.sleep(2)
        
        if rate_limit_hit:
            logger.info(f"GitHub API rate limit reached. Successfully collected {len(training_data)} Python scripts")
        else:
            logger.info(f"Successfully collected {len(training_data)} Python scripts")
        
        return training_data

class EnhancedDatasetManager:
    """Manages multiple training datasets"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_collector = GitHubDataCollector(github_token)
        self.datasets_dir = Path("training_datasets")
        self.datasets_dir.mkdir(exist_ok=True)
    
    def download_alpaca_cleaned(self) -> List[Dict]:
        """Download Alpaca cleaned dataset"""
        try:
            logger.info("Loading Alpaca cleaned dataset...")
            # Load from HuggingFace
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            
            # Convert to our format and sample for relevance
            alpaca_data = []
            programming_keywords = [
                'python', 'code', 'function', 'program', 'script', 'algorithm',
                'variable', 'loop', 'class', 'method', 'debug', 'error',
                'import', 'library', 'framework', 'api', 'database'
            ]
            
            for item in dataset:
                # Filter for programming-related content
                text_to_check = (item['instruction'] + ' ' + item.get('input', '') + ' ' + item['output']).lower()
                if any(keyword in text_to_check for keyword in programming_keywords):
                    alpaca_data.append({
                        'instruction': item['instruction'],
                        'input': item.get('input', ''),
                        'output': item['output'],
                        'source': 'alpaca_cleaned'
                    })
            
            # Sample to avoid overwhelming dataset
            if len(alpaca_data) > 2000:
                alpaca_data = random.sample(alpaca_data, 2000)
            
            logger.info(f"Loaded {len(alpaca_data)} programming-related Alpaca examples")
            return alpaca_data
            
        except Exception as e:
            logger.error(f"Error loading Alpaca dataset: {e}")
            return []
    
    def download_codealpaca(self) -> List[Dict]:
        """Download CodeAlpaca dataset with comprehensive error handling"""
        try:
            logger.info("Loading CodeAlpaca dataset...")
            
            # Try multiple potential datasets
            dataset_options = [
                "sahil2801/CodeAlpaca-20k",
                "HuggingFaceH4/CodeAlpaca_20K", 
                "lucasmccabe-lmi/CodeAlpaca-20k"
            ]
            
            for dataset_name in dataset_options:
                try:
                    logger.info(f"Trying dataset: {dataset_name}")
                    dataset = load_dataset(dataset_name, split="train")
                    
                    codealpaca_data = []
                    for item in dataset:
                        # Handle different possible field names
                        instruction = None
                        input_text = ""
                        output_text = None
                        
                        # Try different field names for instruction
                        for inst_field in ['instruction', 'prompt', 'query', 'question']:
                            if inst_field in item and item[inst_field]:
                                instruction = item[inst_field]
                                break
                        
                        # Try different field names for input
                        for input_field in ['input', 'context', 'text']:
                            if input_field in item and item[input_field]:
                                input_text = item[input_field]
                                break
                        
                        # Try different field names for output
                        for output_field in ['output', 'completion', 'response', 'answer']:
                            if output_field in item and item[output_field]:
                                output_text = item[output_field]
                                break
                        
                        # Only add if we have both instruction and output
                        if instruction and output_text:
                            codealpaca_data.append({
                                'instruction': instruction,
                                'input': input_text,
                                'output': output_text,
                                'source': f'codealpaca_{dataset_name.split("/")[-1]}'
                            })
                    
                    if codealpaca_data:
                        # Sample to keep dataset manageable
                        if len(codealpaca_data) > 1500:
                            codealpaca_data = random.sample(codealpaca_data, 1500)
                        
                        logger.info(f"Successfully loaded {len(codealpaca_data)} CodeAlpaca examples from {dataset_name}")
                        return codealpaca_data
                    else:
                        logger.warning(f"No valid data found in {dataset_name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_name}: {e}")
                    continue
            
            # If all datasets fail, create some manual coding examples
            logger.warning("All CodeAlpaca datasets failed, creating manual coding examples")
            return self.create_manual_coding_examples()
            
        except Exception as e:
            logger.error(f"Critical error in CodeAlpaca loading: {e}")
            return self.create_manual_coding_examples()
    
    def create_manual_coding_examples(self) -> List[Dict]:
        """Create manual coding examples as fallback"""
        return [
            {
                'instruction': 'Write a Python function to reverse a string',
                'input': '',
                'output': 'Here\'s a Python function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n\n# Alternative methods:\ndef reverse_string_loop(s):\n    result = ""\n    for char in s:\n        result = char + result\n    return result\n\ndef reverse_string_builtin(s):\n    return "".join(reversed(s))\n\n# Example usage\nprint(reverse_string("hello"))  # Output: "olleh"\n```',
                'source': 'manual_coding'
            },
            {
                'instruction': 'Create a Python class for a basic calculator',
                'input': '',
                'output': 'Here\'s a basic calculator class in Python:\n\n```python\nclass Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def divide(self, a, b):\n        if b == 0:\n            raise ValueError("Cannot divide by zero")\n        return a / b\n    \n    def power(self, a, b):\n        return a ** b\n\n# Usage example\ncalc = Calculator()\nprint(calc.add(5, 3))      # 8\nprint(calc.divide(10, 2))  # 5.0\n```',
                'source': 'manual_coding'
            },
            {
                'instruction': 'Write a function to find the largest number in a list',
                'input': '',
                'output': 'Here are several ways to find the largest number in a list:\n\n```python\ndef find_largest_builtin(numbers):\n    if not numbers:\n        return None\n    return max(numbers)\n\ndef find_largest_loop(numbers):\n    if not numbers:\n        return None\n    \n    largest = numbers[0]\n    for num in numbers[1:]:\n        if num > largest:\n            largest = num\n    return largest\n\ndef find_largest_recursive(numbers):\n    if not numbers:\n        return None\n    if len(numbers) == 1:\n        return numbers[0]\n    \n    mid = len(numbers) // 2\n    left_max = find_largest_recursive(numbers[:mid])\n    right_max = find_largest_recursive(numbers[mid:])\n    \n    return max(left_max, right_max)\n\n# Example usage\nnumbers = [3, 7, 2, 9, 1, 5]\nprint(find_largest_builtin(numbers))  # 9\n```',
                'source': 'manual_coding'
            }
        ]
    
    def download_evol_instruct_code(self) -> List[Dict]:
        """Download Evol-Instruct-Code dataset"""
        try:
            logger.info("Loading Evol-Instruct-Code dataset...")
            # Load from HuggingFace
            dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
            
            evol_data = []
            for item in dataset:
                evol_data.append({
                    'instruction': item['instruction'],
                    'input': '',
                    'output': item['output'],
                    'source': 'evol_instruct_code'
                })
            
            # Sample to keep dataset manageable (take high-quality subset)
            if len(evol_data) > 2000:
                evol_data = random.sample(evol_data, 2000)
            
            logger.info(f"Loaded {len(evol_data)} Evol-Instruct-Code examples")
            return evol_data
            
        except Exception as e:
            logger.error(f"Error loading Evol-Instruct-Code dataset: {e}")
            return []
    
    def create_manual_examples(self) -> List[Dict]:
        """Create high-quality manual examples"""
        manual_examples = [
            {
                "instruction": "Explain the difference between a list and a tuple in Python",
                "input": "",
                "output": "Lists and tuples are both sequence types in Python, but they have key differences:\n\n**Lists:**\n- Mutable (can be changed after creation)\n- Use square brackets []\n- Can append, remove, or modify elements\n- Slightly slower due to mutability\n\n```python\nmy_list = [1, 2, 3]\nmy_list.append(4)  # Works\nmy_list[0] = 10    # Works\nprint(my_list)     # [10, 2, 3, 4]\n```\n\n**Tuples:**\n- Immutable (cannot be changed after creation)\n- Use parentheses ()\n- Fixed size and content\n- Faster and memory efficient\n- Can be used as dictionary keys\n\n```python\nmy_tuple = (1, 2, 3)\n# my_tuple.append(4)  # Error!\n# my_tuple[0] = 10   # Error!\nprint(my_tuple)      # (1, 2, 3)\n```\n\n**When to use:**\n- Lists: When you need to modify the data\n- Tuples: For fixed data like coordinates (x, y) or database records",
                "source": "manual"
            },
            {
                "instruction": "How do I handle exceptions in Python properly?",
                "input": "",
                "output": "Proper exception handling in Python involves using try-except blocks strategically:\n\n**Basic Structure:**\n```python\ntry:\n    # Code that might raise an exception\n    result = risky_operation()\nexcept SpecificError as e:\n    # Handle specific errors\n    print(f\"Specific error: {e}\")\nexcept Exception as e:\n    # Handle any other errors\n    print(f\"Unexpected error: {e}\")\nelse:\n    # Runs only if no exception occurred\n    print(\"Success!\")\nfinally:\n    # Always runs (cleanup code)\n    cleanup_resources()\n```\n\n**Best Practices:**\n1. **Be specific** - Catch specific exceptions rather than bare `except:`\n2. **Don't ignore errors** - Always handle or log them\n3. **Use finally for cleanup** - Close files, connections, etc.\n4. **Fail fast** - Don't catch exceptions you can't handle\n\n**Example:**\n```python\ndef safe_file_read(filename):\n    try:\n        with open(filename, 'r') as file:\n            return file.read()\n    except FileNotFoundError:\n        return \"File not found\"\n    except PermissionError:\n        return \"Permission denied\"\n    except Exception as e:\n        logger.error(f\"Unexpected error: {e}\")\n        return None\n```",
                "source": "manual"
            },
            {
                "instruction": "Write a Python function to find the most frequent element in a list",
                "input": "",
                "output": "Here are several ways to find the most frequent element in a list:\n\n**Method 1: Using Counter (Recommended)**\n```python\nfrom collections import Counter\n\ndef most_frequent_counter(lst):\n    if not lst:\n        return None\n    \n    counter = Counter(lst)\n    return counter.most_common(1)[0][0]\n\n# Example\nnumbers = [1, 2, 3, 2, 2, 4, 5, 2]\nresult = most_frequent_counter(numbers)\nprint(result)  # Output: 2\n```\n\n**Method 2: Manual counting with dictionary**\n```python\ndef most_frequent_manual(lst):\n    if not lst:\n        return None\n    \n    frequency = {}\n    for item in lst:\n        frequency[item] = frequency.get(item, 0) + 1\n    \n    return max(frequency, key=frequency.get)\n```\n\n**Method 3: Using max with count (less efficient)**\n```python\ndef most_frequent_max(lst):\n    if not lst:\n        return None\n    \n    return max(set(lst), key=lst.count)\n```\n\n**With ties handling:**\n```python\ndef most_frequent_with_ties(lst):\n    if not lst:\n        return []\n    \n    counter = Counter(lst)\n    max_count = max(counter.values())\n    return [item for item, count in counter.items() if count == max_count]\n```\n\nThe Counter method is most efficient for large lists!",
                "source": "manual"
            }
        ]
        
        return manual_examples
    
    def create_fallback_dataset(self) -> List[Dict]:
        """Create a fallback dataset if external datasets fail"""
        logger.info("Creating fallback programming dataset...")
        
        fallback_examples = [
            {
                "instruction": "Write a Python function to calculate factorial",
                "input": "",
                "output": "Here's a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    if n < 0:\n        return None  # Factorial not defined for negative numbers\n    elif n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\n# Alternative iterative approach\ndef factorial_iterative(n):\n    if n < 0:\n        return None\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n\n# Example usage\nprint(factorial(5))  # Output: 120\nprint(factorial_iterative(5))  # Output: 120\n```",
                "source": "fallback"
            },
            {
                "instruction": "Explain Python list comprehensions with examples",
                "input": "",
                "output": "List comprehensions provide a concise way to create lists in Python:\n\n**Basic Syntax:**\n```python\n[expression for item in iterable if condition]\n```\n\n**Examples:**\n\n1. **Simple transformation:**\n```python\nsquares = [x**2 for x in range(10)]\n# Result: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n```\n\n2. **With condition:**\n```python\neven_squares = [x**2 for x in range(10) if x % 2 == 0]\n# Result: [0, 4, 16, 36, 64]\n```\n\n3. **String manipulation:**\n```python\nwords = ['hello', 'world', 'python']\nupper_words = [word.upper() for word in words]\n# Result: ['HELLO', 'WORLD', 'PYTHON']\n```\n\n4. **Nested loops:**\n```python\nmatrix = [[i*j for j in range(3)] for i in range(3)]\n# Result: [[0, 0, 0], [0, 1, 2], [0, 2, 4]]\n```\n\nList comprehensions are more readable and often faster than traditional loops!",
                "source": "fallback"
            },
            {
                "instruction": "How do I handle errors in Python?",
                "input": "",
                "output": "Python uses try-except blocks for error handling:\n\n**Basic Structure:**\n```python\ntry:\n    # Code that might raise an exception\n    risky_code()\nexcept SpecificError as e:\n    # Handle specific error\n    print(f'Specific error occurred: {e}')\nexcept Exception as e:\n    # Handle any other error\n    print(f'Unexpected error: {e}')\nelse:\n    # Runs if no exception occurred\n    print('Success!')\nfinally:\n    # Always runs (cleanup)\n    cleanup_resources()\n```\n\n**Practical Example:**\n```python\ndef safe_divide(a, b):\n    try:\n        result = a / b\n        return result\n    except ZeroDivisionError:\n        print('Cannot divide by zero')\n        return None\n    except TypeError:\n        print('Invalid input types')\n        return None\n    except Exception as e:\n        print(f'Unexpected error: {e}')\n        return None\n\n# Usage\nprint(safe_divide(10, 2))  # 5.0\nprint(safe_divide(10, 0))  # Cannot divide by zero, returns None\n```\n\n**Best Practices:**\n- Be specific with exception types\n- Don't ignore errors silently\n- Use finally for cleanup\n- Log errors for debugging",
                "source": "fallback"
            },
            {
                "instruction": "Write a Python class for a simple calculator",
                "input": "",
                "output": "Here's a simple calculator class in Python:\n\n```python\nclass Calculator:\n    def __init__(self):\n        self.history = []\n    \n    def add(self, a, b):\n        result = a + b\n        self.history.append(f'{a} + {b} = {result}')\n        return result\n    \n    def subtract(self, a, b):\n        result = a - b\n        self.history.append(f'{a} - {b} = {result}')\n        return result\n    \n    def multiply(self, a, b):\n        result = a * b\n        self.history.append(f'{a} * {b} = {result}')\n        return result\n    \n    def divide(self, a, b):\n        try:\n            result = a / b\n            self.history.append(f'{a} / {b} = {result}')\n            return result\n        except ZeroDivisionError:\n            print('Error: Cannot divide by zero')\n            return None\n    \n    def power(self, a, b):\n        result = a ** b\n        self.history.append(f'{a} ^ {b} = {result}')\n        return result\n    \n    def get_history(self):\n        return self.history\n    \n    def clear_history(self):\n        self.history = []\n\n# Example usage\ncalc = Calculator()\nprint(calc.add(5, 3))      # 8\nprint(calc.multiply(4, 7)) # 28\nprint(calc.divide(10, 2))  # 5.0\nprint(calc.get_history())  # Shows all operations\n```",
                "source": "fallback"
            },
            {
                "instruction": "Explain Python dictionaries and their methods",
                "input": "",
                "output": "Python dictionaries are key-value data structures:\n\n**Creating Dictionaries:**\n```python\n# Empty dictionary\nmy_dict = {}\n# or\nmy_dict = dict()\n\n# With initial values\nstudent = {\n    'name': 'John',\n    'age': 20,\n    'grade': 'A'\n}\n```\n\n**Common Dictionary Methods:**\n\n```python\n# Get value\nname = student['name']  # 'John'\nage = student.get('age', 0)  # 20, returns 0 if key doesn't exist\n\n# Add/Update\nstudent['email'] = 'john@email.com'\nstudent.update({'phone': '123-456', 'city': 'NYC'})\n\n# Remove\ndel student['age']\nremoved_value = student.pop('grade', None)  # Removes and returns value\n\n# Check if key exists\nif 'name' in student:\n    print('Name exists')\n\n# Get all keys, values, items\nkeys = student.keys()     # dict_keys(['name', 'email', 'phone', 'city'])\nvalues = student.values() # dict_values(['John', 'john@email.com', '123-456', 'NYC'])\nitems = student.items()   # dict_items([('name', 'John'), ...])\n\n# Iterate\nfor key, value in student.items():\n    print(f'{key}: {value}')\n\n# Dictionary comprehension\nsquared = {x: x**2 for x in range(5)}\n# Result: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}\n```",
                "source": "fallback"
            }
        ]
        
        return fallback_examples
    
    def format_github_data(self, github_scripts: List[Dict]) -> List[Dict]:
        """Format GitHub scripts as instruction-response pairs"""
        formatted_data = []
        
        for script in github_scripts:
            # Create instruction asking to explain the code
            instruction = f"Explain this Python code from {script['repo_name']}:"
            
            # Truncate content if too long
            content = script['content']
            if len(content) > 2000:
                content = content[:2000] + "..."
            
            # Create response explaining the code
            response = f"This is a Python script from the repository '{script['repo_name']}'. "
            if script['description']:
                response += f"The repository is described as: {script['description']}. "
            
            response += f"\n\n```python\n{content}\n```\n\n"
            response += "This code demonstrates various Python programming concepts including proper structure, "
            response += "function definitions, variable usage, and implementation of specific functionality. "
            response += "It's a good example of practical Python programming that you can learn from."
            
            formatted_data.append({
                'instruction': instruction,
                'input': '',
                'output': response,
                'source': 'github'
            })
        
        return formatted_data
    
    def collect_all_datasets(self) -> List[Dict]:
        """Collect and combine all training datasets"""
        logger.info("Collecting comprehensive training datasets...")
        
        all_data = []
        
        # 1. Manual high-quality examples (always include these)
        manual_data = self.create_manual_examples()
        all_data.extend(manual_data)
        logger.info(f"Added {len(manual_data)} manual examples")
        
        # Track successful datasets
        successful_datasets = ['manual']
        
        # 2. Alpaca cleaned (programming subset)
        try:
            alpaca_data = self.download_alpaca_cleaned()
            if alpaca_data:
                all_data.extend(alpaca_data)
                successful_datasets.append('alpaca_cleaned')
            else:
                logger.warning("Alpaca dataset returned no data")
        except Exception as e:
            logger.error(f"Failed to load Alpaca dataset: {e}")
        
        # 3. CodeAlpaca (with better error handling)
        try:
            codealpaca_data = self.download_codealpaca()
            if codealpaca_data:
                all_data.extend(codealpaca_data)
                successful_datasets.append('codealpaca')
            else:
                logger.warning("CodeAlpaca dataset returned no data")
        except Exception as e:
            logger.error(f"Failed to load CodeAlpaca dataset: {e}")
        
        # 4. Evol-Instruct-Code
        try:
            evol_data = self.download_evol_instruct_code()
            if evol_data:
                all_data.extend(evol_data)
                successful_datasets.append('evol_instruct_code')
            else:
                logger.warning("Evol-Instruct-Code dataset returned no data")
        except Exception as e:
            logger.error(f"Failed to load Evol-Instruct-Code dataset: {e}")
        
        # 5. GitHub scripts (if available)
        try:
            github_scripts = self.github_collector.collect_github_scripts(100)
            if github_scripts:
                github_data = self.format_github_data(github_scripts)
                all_data.extend(github_data)
                successful_datasets.append('github')
            else:
                logger.warning("GitHub data collection returned no scripts")
        except Exception as e:
            logger.error(f"Failed to collect GitHub scripts: {e}")
        
        # 6. Add fallback dataset if we don't have enough data
        if len(all_data) < 100:
            logger.warning(f"Only {len(all_data)} examples collected, adding fallback dataset")
            fallback_data = self.create_fallback_dataset()
            all_data.extend(fallback_data)
            successful_datasets.append('fallback')
        
        # Always ensure we have at least some manual examples
        if 'manual' not in successful_datasets:
            manual_data = self.create_manual_examples()
            all_data.extend(manual_data)
            successful_datasets.append('manual_added')
        
        # Shuffle for good distribution
        random.shuffle(all_data)
        
        logger.info(f"Total training examples collected: {len(all_data)}")
        
        # Show distribution by source
        source_counts = {}
        for item in all_data:
            source = item.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("Dataset distribution:")
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count} examples")
        
        logger.info(f"Successfully loaded datasets: {', '.join(successful_datasets)}")
        
        return all_data

class EnhancedProfessionalForumBot:
    """
    Enhanced professional-grade forum bot with multiple training datasets
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.model_path = "./enhanced-tinyllama-forum-professional"
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
        
        # Cache configuration
        self.response_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Dataset manager
        self.dataset_manager = EnhancedDatasetManager(github_token)
        
        # Initialize components
        self.setup_database()
        self.setup_model()
        
        logger.info("Enhanced Professional Forum Bot initialized successfully")
    
    def setup_database(self):
        """Setup SQLite database for analytics"""
        self.db_path = "enhanced_forum_bot.db"
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
            CREATE TABLE IF NOT EXISTS training_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_source TEXT,
                example_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db.commit()
        logger.info("Database initialized")
    
    def setup_model(self):
        """Load or train the enhanced model with robust error handling"""
        try:
            if Path(self.model_path).exists():
                logger.info(f"Found model directory: {self.model_path}")
                
                # Check if model is complete and valid
                if self._is_model_complete():
                    logger.info(f"Loading enhanced fine-tuned model from {self.model_path}")
                    success = self._load_enhanced_model_safely()
                    if success:
                        return
                    else:
                        logger.warning("Failed to load model safely, will retrain")
                else:
                    logger.warning("Model directory incomplete, will retrain")
                
                # Clean up corrupted model directory
                self._cleanup_model_directory()
            
            logger.info("Enhanced model not found or corrupted, training new model...")
            self.train_enhanced_model()
                
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            logger.info("Falling back to base model")
            self.load_base_model()
    
    def _is_model_complete(self) -> bool:
        """Check if the saved model has all required files"""
        model_path = Path(self.model_path)
        
        # Required files for a complete model
        required_files = [
            "config.json",
            "pytorch_model.bin",  # or model.safetensors
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt"  # or equivalent vocab file
        ]
        
        # Check for alternative model file formats
        model_files = ["pytorch_model.bin", "model.safetensors"]
        has_model_file = any((model_path / f).exists() for f in model_files)
        
        # Check for alternative vocab files
        vocab_files = ["vocab.txt", "vocab.json", "merges.txt", "spiece.model"]
        has_vocab_file = any((model_path / f).exists() for f in vocab_files)
        
        # Essential files
        essential_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        has_essential = all((model_path / f).exists() for f in essential_files)
        
        is_complete = has_essential and has_model_file and has_vocab_file
        
        if not is_complete:
            logger.info("Model completeness check:")
            logger.info(f"  Essential files present: {has_essential}")
            logger.info(f"  Model file present: {has_model_file}")
            logger.info(f"  Vocab file present: {has_vocab_file}")
            
            # List what files are actually present
            if model_path.exists():
                existing_files = [f.name for f in model_path.iterdir() if f.is_file()]
                logger.info(f"  Existing files: {existing_files}")
        
        return is_complete
    
    def _load_enhanced_model_safely(self) -> bool:
        """Safely load the enhanced model with comprehensive error handling"""
        try:
            # Step 1: Try to load tokenizer
            logger.info("Step 1: Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True,
                    local_files_only=True  # Force local loading
                )
                logger.info("✅ Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"❌ Tokenizer loading failed: {e}")
                return False
            
            # Step 2: Try to load model
            logger.info("Step 2: Loading model...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    local_files_only=True  # Force local loading
                )
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Model loading failed: {e}")
                return False
            
            # Step 3: Validate model works
            logger.info("Step 3: Validating model...")
            try:
                test_input = "What is Python?"
                inputs = self.tokenizer(test_input, return_tensors="pt")
                
                # Move to same device as model if using GPU
                if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Test generation
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                logger.info("✅ Model validation successful")
                logger.info(f"  Test generation: {response[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Model validation failed: {e}")
                return False
            
            # Step 4: Set model version and final setup
            self.model.eval()
            self.model_version = "enhanced-fine-tuned"
            logger.info("✅ Enhanced fine-tuned model loaded and validated successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error in model loading: {e}")
            return False
    
    def _cleanup_model_directory(self):
        """Clean up corrupted model directory"""
        try:
            model_path = Path(self.model_path)
            if model_path.exists():
                logger.info(f"Cleaning up corrupted model directory: {self.model_path}")
                shutil.rmtree(model_path)
                logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.warning(f"Failed to cleanup model directory: {e}")
    
    def load_base_model(self):
        """Load base TinyLlama model with improved error handling"""
        try:
            logger.info("Loading base TinyLlama model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.fallback_model, 
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.fallback_model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if hasattr(self.model.config, 'pad_token_id'):
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            self.model.eval()
            self.model_version = "base"
            
            # Quick validation
            test_input = "Hello"
            inputs = self.tokenizer(test_input, return_tensors="pt")
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            logger.info("✅ Base model loaded and validated successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load base model: {e}")
            raise RuntimeError(f"Could not load any model: {e}")
    
    def train_enhanced_model(self):
        """Train model with comprehensive datasets - with robust error handling"""
        try:
            logger.info("=== Starting Enhanced Model Training ===")
            
            # Collect all datasets
            all_training_data = self.dataset_manager.collect_all_datasets()
            
            if not all_training_data:
                logger.error("No training data available")
                self.load_base_model()
                return
            
            # Ensure we have a reasonable amount of data
            if len(all_training_data) < 10:
                logger.warning("Very few training examples, adding fallback data")
                fallback_data = self.dataset_manager.create_fallback_dataset()
                all_training_data.extend(fallback_data)
            
            # Save training stats to database
            self._save_training_stats(all_training_data)
            
            # Attempt training with multiple fallback strategies
            success = self._attempt_training_with_fallbacks(all_training_data)
            
            if not success:
                logger.error("All training attempts failed, using base model")
                self.load_base_model()
            else:
                logger.info("=== Enhanced Model Training Completed Successfully ===")
                
        except Exception as e:
            logger.error(f"Enhanced training failed: {e}")
            logger.info("Falling back to base model due to critical error")
            try:
                self.load_base_model()
            except Exception as base_error:
                logger.error(f"Even base model loading failed: {base_error}")
                raise RuntimeError(f"Complete model loading failure: {e}, {base_error}")
    
    def _save_training_stats(self, training_data):
        """Save training statistics to database"""
        try:
            source_counts = {}
            for item in training_data:
                source = item.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            with self.db_lock:
                cursor = self.db.cursor()
                for source, count in source_counts.items():
                    cursor.execute("""
                        INSERT INTO training_stats (dataset_source, example_count)
                        VALUES (?, ?)
                    """, (source, count))
                self.db.commit()
        except Exception as e:
            logger.warning(f"Failed to save training stats: {e}")
    
    def _attempt_training_with_fallbacks(self, all_training_data) -> bool:
        """Attempt training with multiple fallback strategies"""
        
        # Strategy 1: Full dataset with optimized parameters
        try:
            logger.info("Attempting Strategy 1: Full dataset training")
            return self._train_with_strategy_1(all_training_data)
        except Exception as e:
            logger.warning(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Reduced dataset with simple parameters  
        try:
            logger.info("Attempting Strategy 2: Reduced dataset training")
            return self._train_with_strategy_2(all_training_data)
        except Exception as e:
            logger.warning(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Minimal dataset with basic parameters
        try:
            logger.info("Attempting Strategy 3: Minimal dataset training")
            return self._train_with_strategy_3(all_training_data)
        except Exception as e:
            logger.warning(f"Strategy 3 failed: {e}")
        
        return False
    
    def _train_with_strategy_1(self, all_training_data) -> bool:
        """Strategy 1: Full dataset with optimized parameters"""
        
        # Load base model for training (with memory optimization)
        logger.info("Loading base model for training...")
        tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
        
        # Use lower precision and memory optimization for training
        model = AutoModelForCausalLM.from_pretrained(
            self.fallback_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        
        # Reduce dataset size if memory is limited
        max_examples = 3000 if torch.cuda.is_available() else 1000
        if len(all_training_data) > max_examples:
            logger.info(f"Reducing dataset from {len(all_training_data)} to {max_examples} examples for memory efficiency")
            all_training_data = random.sample(all_training_data, max_examples)
        
        # Format training data for TinyLlama
        training_texts = self._format_training_data(all_training_data)
        logger.info(f"Formatted {len(training_texts)} training examples")
        
        # Tokenize training data
        tokenized_dataset = self._tokenize_data(training_texts, tokenizer)
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficiency
        )
        
        # Training arguments - clean and simple
        training_args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=3e-5,
            warmup_steps=50,
            logging_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            dataloader_drop_last=True,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
        )
        
        # Create trainer with proper error handling
        trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
        
        # Train with memory management
        logger.info(f"Starting enhanced model training on {len(training_texts)} examples...")
        logger.info(f"Training for {training_args.num_train_epochs} epochs with batch size {training_args.per_device_train_batch_size}")
        logger.info("This may take several minutes depending on your hardware...")
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        trainer.train()
        logger.info("✅ Training completed successfully!")
        
        # Save model with comprehensive file validation
        logger.info("Saving trained model...")
        self._save_model_safely(trainer, tokenizer)
        
        # Load the trained model
        self.tokenizer = tokenizer
        self.model = model
        self.model_version = "enhanced-fine-tuned"
        
        # Quick validation test
        try:
            test_input = "What is Python?"
            test_tokens = tokenizer(test_input, return_tensors="pt")
            with torch.no_grad():
                _ = model(**test_tokens)
            logger.info("✅ Model validation successful")
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
        
        return True
    
    def _save_model_safely(self, trainer, tokenizer):
        """Safely save model with validation"""
        try:
            # Ensure output directory exists
            Path(self.model_path).mkdir(parents=True, exist_ok=True)
            
            # Save model using trainer
            logger.info(f"Saving trainer model to {self.model_path}")
            trainer.save_model()
            
            # Save tokenizer separately to ensure it's properly saved
            logger.info(f"Saving tokenizer to {self.model_path}")
            tokenizer.save_pretrained(self.model_path)
            
            # Validate save was successful
            if self._is_model_complete():
                logger.info("✅ Model saved and validated successfully")
            else:
                logger.warning("⚠️ Model save validation failed - some files may be missing")
                
                # Try to save again with more explicit method
                logger.info("Attempting alternative save method...")
                
                # Save model state dict manually if needed
                model_file = Path(self.model_path) / "pytorch_model.bin"
                if not model_file.exists():
                    torch.save(trainer.model.state_dict(), model_file)
                    logger.info("✅ Manual model state dict saved")
                
                # Verify tokenizer files
                tokenizer_config = Path(self.model_path) / "tokenizer_config.json"
                if not tokenizer_config.exists():
                    # Try saving tokenizer again
                    tokenizer.save_pretrained(self.model_path, safe_serialization=False)
                    logger.info("✅ Tokenizer re-saved")
                
        except Exception as e:
            logger.error(f"Error in model saving: {e}")
            raise
    
    def _train_with_strategy_2(self, all_training_data) -> bool:
        """Strategy 2: Reduced dataset with simple parameters"""
        
        # Use smaller subset (1000 examples max)
        subset_size = min(1000, len(all_training_data))
        training_subset = random.sample(all_training_data, subset_size)
        
        logger.info(f"Training with reduced dataset of {len(training_subset)} examples")
        
        # Load model with simpler configuration
        tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.fallback_model,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use FP32 for stability
            low_cpu_mem_usage=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        
        # Format and tokenize data
        training_texts = self._format_training_data(training_subset)
        tokenized_dataset = self._tokenize_data(training_texts, tokenizer)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Simple training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            logging_steps=50,
            save_strategy="epoch",
            dataloader_drop_last=True,
            report_to="none",
            remove_unused_columns=False,
        )
        
        trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
        
        trainer.train()
        logger.info("✅ Strategy 2 training completed successfully!")
        
        # Save model safely
        logger.info("Saving Strategy 2 model...")
        self._save_model_safely(trainer, tokenizer)
        
        self.tokenizer = tokenizer
        self.model = model
        self.model_version = "enhanced-fine-tuned-reduced"
        
        # Quick validation
        try:
            test_tokens = tokenizer("Hello", return_tensors="pt")
            with torch.no_grad():
                _ = model(**test_tokens)
            logger.info("✅ Strategy 2: Model validation successful")
        except Exception as e:
            logger.warning(f"Strategy 2: Model validation failed: {e}")
        
        return True
    
    def _train_with_strategy_3(self, all_training_data) -> bool:
        """Strategy 3: Minimal dataset with basic parameters"""
        
        # Use very small subset (100 examples max)
        subset_size = min(100, len(all_training_data))
        training_subset = random.sample(all_training_data, subset_size)
        
        logger.info(f"Training with minimal dataset of {len(training_subset)} examples")
        
        # Load model with minimal configuration
        tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.fallback_model, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        
        # Format and tokenize data
        training_texts = self._format_training_data(training_subset)
        tokenized_dataset = self._tokenize_data(training_texts, tokenizer)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # Minimal training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
            save_strategy="epoch",
            logging_steps=10,
            report_to="none",
        )
        
        trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
        
        trainer.train()
        logger.info("✅ Strategy 3 training completed successfully!")
        
        # Save model safely
        logger.info("Saving Strategy 3 model...")
        self._save_model_safely(trainer, tokenizer)
        
        self.tokenizer = tokenizer
        self.model = model
        self.model_version = "enhanced-fine-tuned-minimal"
        
        # Quick validation
        try:
            test_tokens = tokenizer("Test", return_tensors="pt")
            with torch.no_grad():
                _ = model(**test_tokens)
            logger.info("✅ Strategy 3: Model validation successful")
        except Exception as e:
            logger.warning(f"Strategy 3: Model validation failed: {e}")
        
        return True
    
    def _format_training_data(self, training_data):
        """Format training data for TinyLlama"""
        training_texts = []
        for item in training_data:
            # Combine instruction and input
            full_instruction = item['instruction']
            if item.get('input') and item['input'].strip():
                full_instruction += f"\n\nInput: {item['input']}"
            
            # Format for TinyLlama chat format
            formatted = f"<|system|>\nYou are an expert programming mentor and helpful assistant. You provide accurate, detailed, and educational responses to programming questions and general inquiries. You explain concepts clearly with practical examples and encourage learning.\n</s>\n<|user|>\n{full_instruction}\n</s>\n<|assistant|>\n{item['output']}\n</s>"
            training_texts.append(formatted)
        
        return training_texts
    
    def _tokenize_data(self, training_texts, tokenizer):
        """Tokenize training data"""
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=1024,
                padding=False,  # Don't pad here, let the data collator handle it
                return_tensors=None  # Return lists, not tensors
            )
            
            # For causal language modeling, labels should be the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        return tokenized_dataset
    
    def _create_trainer(self, model, training_args, tokenized_dataset, tokenizer, data_collator):
        """Create trainer with version compatibility"""
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                processing_class=tokenizer,  # Use processing_class for newer versions
                data_collator=data_collator,
            )
        except TypeError:
            # Fallback for older transformers versions
            logger.warning("processing_class not supported, using tokenizer parameter")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        
        return trainer
    
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
            'length': min(len(response.split()) / 100, 1.0) * 0.3,
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
            if line and line[:50] not in seen_lines:  # Check first 50 chars
                cleaned_lines.append(line)
                seen_lines.add(line[:50])
        
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
                confidence = 0.9
                
                return {
                    'response': cached_response,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'cached': True,
                    'model_version': self.model_version
                }
            
            # Generate new response
            with self.model_lock:
                # Format prompt for TinyLlama
                formatted_prompt = f"<|system|>\nYou are an expert programming mentor and helpful assistant. You provide accurate, detailed, and educational responses to programming questions and general inquiries. You explain concepts clearly with practical examples and encourage learning.\n</s>\n<|user|>\n{query}\n</s>\n<|assistant|>\n"
                
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
        """Get performance metrics including training stats"""
        uptime = datetime.now() - self.metrics['start_time']
        cache_hit_rate = 0
        if self.metrics['total_requests'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / self.metrics['total_requests']
        
        # Get training stats
        training_stats = {}
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT dataset_source, example_count FROM training_stats ORDER BY timestamp DESC LIMIT 10")
            training_data = cursor.fetchall()
            for source, count in training_data:
                training_stats[source] = count
        except:
            pass
        
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_hit_rate': f"{cache_hit_rate:.2%}",
            'avg_response_time': f"{self.metrics['avg_response_time']:.3f}s",
            'error_count': self.metrics['error_count'],
            'uptime': str(uptime),
            'model_version': self.model_version,
            'cached_responses': len(self.response_cache),
            'training_datasets': training_stats
        }

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize enhanced bot
logger.info("Initializing Enhanced Professional Forum Bot...")
github_token = os.getenv('GITHUB_TOKEN')  # Set this environment variable if you have one
forum_bot = EnhancedProfessionalForumBot(github_token=github_token)

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
    
    # Format response
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

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain model with fresh datasets"""
    try:
        # Clean up existing model
        if Path(forum_bot.model_path).exists():
            forum_bot._cleanup_model_directory()
        
        # Force retraining
        forum_bot.train_enhanced_model()
        
        return jsonify({
            'status': 'success',
            'message': 'Enhanced model retrained successfully',
            'model_version': forum_bot.model_version
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/model/status', methods=['GET'])
def model_status():
    """Get detailed model status"""
    try:
        model_path = Path(forum_bot.model_path)
        
        status = {
            'model_version': forum_bot.model_version,
            'model_path': str(forum_bot.model_path),
            'model_exists': model_path.exists(),
            'model_complete': False,
            'files_present': []
        }
        
        if model_path.exists():
            status['model_complete'] = forum_bot._is_model_complete()
            status['files_present'] = [f.name for f in model_path.iterdir() if f.is_file()]
        
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/datasets/info', methods=['GET'])
def dataset_info():
    """Get information about training datasets"""
    try:
        cursor = forum_bot.db.cursor()
        cursor.execute("SELECT dataset_source, example_count, timestamp FROM training_stats ORDER BY timestamp DESC")
        dataset_info = []
        for source, count, timestamp in cursor.fetchall():
            dataset_info.append({
                'source': source,
                'example_count': count,
                'timestamp': timestamp
            })
        
        return jsonify({
            'datasets': dataset_info,
            'total_examples': sum(item['example_count'] for item in dataset_info)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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
    logger.info("Starting Enhanced Professional Forum Bot Server...")
    
    # SSL Certificate paths
    cert_path = r'C:\portablexampp\apache\conf\ssl.crt\certificate.crt'
    key_path = r'C:\portablexampp\apache\conf\ssl.key\private.key'
    
    # Server configuration
    host = "0.0.0.0"
    port = 8043
    
    logger.info(f"Enhanced bot initialized with model version: {forum_bot.model_version}")
    logger.info(f"Cached responses: {len(forum_bot.response_cache)}")
    logger.info(f"Server starting on {host}:{port}")
    
    if forum_bot.model_version == "base":
        logger.info("=" * 60)
        logger.info("🔄 NOTICE: Using base model (not fine-tuned)")
        logger.info("💡 To train the enhanced model, send a POST request to:")
        logger.info(f"   https://{host}:{port}/retrain")
        logger.info("📊 Check model status at:")
        logger.info(f"   https://{host}:{port}/model/status")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("✅ ENHANCED MODEL LOADED SUCCESSFULLY!")
        logger.info(f"🤖 Model version: {forum_bot.model_version}")
        logger.info("🚀 Ready to provide high-quality programming assistance!")
        logger.info("=" * 60)
    
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