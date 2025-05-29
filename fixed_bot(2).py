#!/usr/bin/env python3
"""
Enhanced Professional Forum Bot - Fixed Version
Addresses CUDA memory issues, encoding problems, and device mismatches

Key Fixes:
1. Better memory management for GPU training
2. Fixed Unicode encoding issues in logging
3. Proper device handling for model inference
4. More robust error handling
5. Gradual memory allocation strategies
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
import sys

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

# Fix Unicode encoding issues
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure logging with proper encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_forum_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('enhanced_professional_forum_bot')

class MemoryManager:
    """Manages GPU memory allocation and cleanup"""
    
    @staticmethod
    def get_gpu_memory_info():
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            cached_memory = torch.cuda.memory_reserved(device)
            free_memory = total_memory - allocated_memory
            
            return {
                'total': total_memory // (1024**3),  # GB
                'allocated': allocated_memory // (1024**3),  # GB
                'free': free_memory // (1024**3),  # GB
                'cached': cached_memory // (1024**3)  # GB
            }
        return None
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def estimate_training_memory_needs(dataset_size, batch_size=1):
        """Estimate memory needs for training"""
        # Rough estimation: 2GB base + dataset_size * batch_size * 0.001 GB
        base_memory = 2.0  # GB
        estimated_memory = base_memory + (dataset_size * batch_size * 0.001)
        return estimated_memory

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
                return []
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
    
    def collect_github_scripts(self, target_count: int = 50) -> List[Dict]:
        """Collect Python scripts for training - reduced target to save memory"""
        logger.info(f"Collecting {target_count} Python scripts from GitHub...")
        
        training_data = []
        collected_count = 0
        rate_limit_hit = False
        
        # Reduced search queries to save API calls and memory
        search_queries = [
            "python tutorial beginner",
            "python data analysis",
            "python automation scripts",
            "python api examples",
            "python utilities tools"
        ]
        
        for query in search_queries:
            if collected_count >= target_count or rate_limit_hit:
                break
                
            repos = self.search_python_repos(query, per_page=3)  # Reduced per_page
            
            for repo in repos:
                if collected_count >= target_count or rate_limit_hit:
                    break
                
                python_files = self.get_python_files(repo['full_name'], max_files=1)  # Reduced max_files
                
                if not python_files and "rate limit" in str(python_files).lower():
                    rate_limit_hit = True
                    logger.warning("GitHub API rate limit reached, stopping collection")
                    break
                
                for file_info in python_files:
                    if collected_count >= target_count:
                        break
                    
                    content = self.get_file_content(file_info['download_url'])
                    if content and 200 <= len(content) <= 2000:  # Smaller files to save memory
                        training_data.append({
                            'repo_name': repo['full_name'],
                            'file_name': file_info['name'],
                            'content': content,
                            'description': repo.get('description', ''),
                            'stars': repo.get('stargazers_count', 0)
                        })
                        collected_count += 1
                        logger.info(f"Collected {collected_count}/{target_count}: {repo['full_name']}/{file_info['name']}")
                
                # Increased rate limiting delay
                time.sleep(3)
        
        logger.info(f"Successfully collected {len(training_data)} Python scripts")
        return training_data

class EnhancedDatasetManager:
    """Manages multiple training datasets with memory optimization"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_collector = GitHubDataCollector(github_token)
        self.datasets_dir = Path("training_datasets")
        self.datasets_dir.mkdir(exist_ok=True)
    
    def download_alpaca_cleaned(self, max_examples: int = 500) -> List[Dict]:
        """Download Alpaca cleaned dataset with size limit"""
        try:
            logger.info("Loading Alpaca cleaned dataset...")
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            
            alpaca_data = []
            programming_keywords = [
                'python', 'code', 'function', 'program', 'script', 'algorithm',
                'variable', 'loop', 'class', 'method', 'debug', 'error',
                'import', 'library', 'framework', 'api', 'database'
            ]
            
            for item in dataset:
                if len(alpaca_data) >= max_examples:
                    break
                    
                text_to_check = (item['instruction'] + ' ' + item.get('input', '') + ' ' + item['output']).lower()
                if any(keyword in text_to_check for keyword in programming_keywords):
                    alpaca_data.append({
                        'instruction': item['instruction'],
                        'input': item.get('input', ''),
                        'output': item['output'],
                        'source': 'alpaca_cleaned'
                    })
            
            logger.info(f"Loaded {len(alpaca_data)} programming-related Alpaca examples")
            return alpaca_data
            
        except Exception as e:
            logger.error(f"Error loading Alpaca dataset: {e}")
            return []
    
    def download_codealpaca(self, max_examples: int = 300) -> List[Dict]:
        """Download CodeAlpaca dataset with size limit"""
        try:
            logger.info("Loading CodeAlpaca dataset...")
            
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
                        if len(codealpaca_data) >= max_examples:
                            break
                            
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
                        
                        if instruction and output_text:
                            codealpaca_data.append({
                                'instruction': instruction,
                                'input': input_text,
                                'output': output_text,
                                'source': f'codealpaca_{dataset_name.split("/")[-1]}'
                            })
                    
                    if codealpaca_data:
                        logger.info(f"Successfully loaded {len(codealpaca_data)} CodeAlpaca examples")
                        return codealpaca_data
                        
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_name}: {e}")
                    continue
            
            logger.warning("All CodeAlpaca datasets failed, creating manual coding examples")
            return self.create_manual_coding_examples()
            
        except Exception as e:
            logger.error(f"Critical error in CodeAlpaca loading: {e}")
            return self.create_manual_coding_examples()
    
    def download_evol_instruct_code(self) -> List[Dict]:
        """Download Evol-Instruct-Code dataset - RESTORED ORIGINAL"""
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
    
    def download_dolphin_dataset(self) -> List[Dict]:
        """Download Dolphin dataset by Eric Hartford - RESTORED ORIGINAL"""
        try:
            logger.info("Loading Dolphin dataset...")
            
            # Try to load the smaller, higher-quality GPT-4 dataset first
            dolphin_data = []
            
            try:
                # Load deduped GPT-4 completions (higher quality, smaller size)
                logger.info("Attempting to load Dolphin GPT-4 dataset (deduped)...")
                dataset = load_dataset(
                    "ehartford/dolphin", 
                    data_files="flan1m-alpaca-uncensored-deduped.jsonl",
                    split="train"
                )
                source_name = "dolphin_gpt4_deduped"
                
            except Exception as e:
                logger.warning(f"Deduped version failed: {e}")
                try:
                    # Fallback to original GPT-4 dataset
                    logger.info("Attempting to load original Dolphin GPT-4 dataset...")
                    dataset = load_dataset(
                        "ehartford/dolphin", 
                        data_files="flan1m-alpaca-uncensored.jsonl",
                        split="train"
                    )
                    source_name = "dolphin_gpt4"
                    
                except Exception as e2:
                    logger.warning(f"GPT-4 version failed: {e2}")
                    # Final fallback to GPT-3.5 deduped (larger but still good)
                    logger.info("Attempting to load Dolphin GPT-3.5 dataset (deduped)...")
                    dataset = load_dataset(
                        "ehartford/dolphin", 
                        data_files="flan5m-alpaca-uncensored-deduped.jsonl",
                        split="train"
                    )
                    source_name = "dolphin_gpt35_deduped"
            
            # Process the dataset
            for item in dataset:
                # Dolphin uses different field structures, try to handle various formats
                instruction = ""
                input_text = ""
                output_text = ""
                
                # Handle different possible structures
                if 'instruction' in item and 'output' in item:
                    # Standard instruction format
                    instruction = item['instruction']
                    input_text = item.get('input', '')
                    output_text = item['output']
                    
                elif 'conversations' in item:
                    # Multi-turn conversation format
                    conversations = item['conversations']
                    if len(conversations) >= 2:
                        # Take the first user message as instruction
                        for i, conv in enumerate(conversations):
                            if conv.get('from') == 'human' or conv.get('role') == 'user':
                                instruction = conv.get('value', conv.get('content', ''))
                                # Look for the next assistant response
                                if i + 1 < len(conversations):
                                    next_conv = conversations[i + 1]
                                    if next_conv.get('from') == 'gpt' or next_conv.get('role') == 'assistant':
                                        output_text = next_conv.get('value', next_conv.get('content', ''))
                                break
                                
                elif 'text' in item:
                    # Raw text format - try to parse
                    text = item['text']
                    if 'Human:' in text and 'Assistant:' in text:
                        parts = text.split('Human:', 1)
                        if len(parts) > 1:
                            human_part = parts[1]
                            if 'Assistant:' in human_part:
                                instruction, output_text = human_part.split('Assistant:', 1)
                                instruction = instruction.strip()
                                output_text = output_text.strip()
                
                # Only add if we have both instruction and output
                if instruction and output_text and len(instruction) > 10 and len(output_text) > 10:
                    dolphin_data.append({
                        'instruction': instruction.strip(),
                        'input': input_text.strip(),
                        'output': output_text.strip(),
                        'source': source_name
                    })
            
            # Sample to keep dataset manageable (Dolphin can be very large)
            target_size = 2000  # Configurable sample size
            if len(dolphin_data) > target_size:
                logger.info(f"Sampling {target_size} examples from {len(dolphin_data)} total Dolphin examples")
                dolphin_data = random.sample(dolphin_data, target_size)
            
            logger.info(f"Successfully loaded {len(dolphin_data)} Dolphin examples from {source_name}")
            return dolphin_data
            
        except Exception as e:
            logger.error(f"Error loading Dolphin dataset: {e}")
            logger.info("Creating fallback Dolphin-style examples...")
            return self.create_dolphin_style_examples()
    
    def create_dolphin_style_examples(self) -> List[Dict]:
        """Create Dolphin-style examples as fallback - RESTORED ORIGINAL"""
        return [
            {
                'instruction': 'Explain the concept of machine learning in simple terms',
                'input': '',
                'output': 'Machine learning is a branch of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, rather than being explicitly programmed for every scenario.\n\nThink of it like teaching a child to recognize animals:\n\n**Traditional Programming:** You write specific rules like "if it has 4 legs, fur, and barks, it\'s a dog"\n\n**Machine Learning:** You show the computer thousands of pictures labeled "dog" or "cat" and let it figure out the patterns on its own\n\n**Key Components:**\n1. **Data**: Examples to learn from (like photos, text, numbers)\n2. **Algorithm**: The method used to find patterns\n3. **Model**: The "brain" that makes predictions after training\n4. **Training**: The process of learning from examples\n\n**Common Applications:**\n- Email spam detection\n- Recommendation systems (Netflix, Amazon)\n- Voice assistants (Siri, Alexa)\n- Medical diagnosis\n- Self-driving cars\n\nThe beauty is that ML systems can discover patterns humans might miss and adapt to new situations they haven\'t seen before!',
                'source': 'dolphin_style_manual'
            }
        ]
    
    def create_fallback_dataset(self) -> List[Dict]:
        """Create a fallback dataset if external datasets fail - RESTORED ORIGINAL"""
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
            }
        ]
        
        return fallback_examples
    
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
            }
        ]
    
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
            }
        ]
        
        return manual_examples
    
    def collect_all_datasets(self) -> List[Dict]:
        """Collect and combine all training datasets - RESTORED ORIGINAL"""
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
            alpaca_data = self.download_alpaca_cleaned(max_examples=2000)  # RESTORED ORIGINAL SIZE
            if alpaca_data:
                all_data.extend(alpaca_data)
                successful_datasets.append('alpaca_cleaned')
            else:
                logger.warning("Alpaca dataset returned no data")
        except Exception as e:
            logger.error(f"Failed to load Alpaca dataset: {e}")
        
        # 3. CodeAlpaca (with better error handling)
        try:
            codealpaca_data = self.download_codealpaca(max_examples=1500)  # RESTORED ORIGINAL SIZE
            if codealpaca_data:
                all_data.extend(codealpaca_data)
                successful_datasets.append('codealpaca')
            else:
                logger.warning("CodeAlpaca dataset returned no data")
        except Exception as e:
            logger.error(f"Failed to load CodeAlpaca dataset: {e}")
        
        # 4. Evol-Instruct-Code - RESTORED ORIGINAL
        try:
            evol_data = self.download_evol_instruct_code()
            if evol_data:
                all_data.extend(evol_data)
                successful_datasets.append('evol_instruct_code')
            else:
                logger.warning("Evol-Instruct-Code dataset returned no data")
        except Exception as e:
            logger.error(f"Failed to load Evol-Instruct-Code dataset: {e}")
        
        # 5. Dolphin Dataset (Eric Hartford) - RESTORED ORIGINAL
        try:
            dolphin_data = self.download_dolphin_dataset()
            if dolphin_data:
                all_data.extend(dolphin_data)
                successful_datasets.append('dolphin')
            else:
                logger.warning("Dolphin dataset returned no data")
        except Exception as e:
            logger.error(f"Failed to load Dolphin dataset: {e}")
        
        # 6. GitHub scripts (if available) - RESTORED
        try:
            github_scripts = self.github_collector.collect_github_scripts(100)  # RESTORED ORIGINAL SIZE
            if github_scripts:
                github_data = self.format_github_data(github_scripts)
                all_data.extend(github_data)
                successful_datasets.append('github')
            else:
                logger.warning("GitHub data collection returned no scripts")
        except Exception as e:
            logger.error(f"Failed to collect GitHub scripts: {e}")
        
        # 7. Add fallback dataset if we don't have enough data
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
    
    def format_github_data(self, github_scripts: List[Dict]) -> List[Dict]:
        """Format GitHub scripts as instruction-response pairs"""
        formatted_data = []
        
        for script in github_scripts:
            instruction = f"Explain this Python code from {script['repo_name']}:"
            
            content = script['content']
            if len(content) > 1500:  # Reduced to save memory
                content = content[:1500] + "..."
            
            response = f"This is a Python script from the repository '{script['repo_name']}'. "
            if script['description']:
                response += f"The repository is described as: {script['description']}. "
            
            response += f"\n\n```python\n{content}\n```\n\n"
            response += "This code demonstrates various Python programming concepts including proper structure, "
            response += "function definitions, variable usage, and implementation of specific functionality."
            
            formatted_data.append({
                'instruction': instruction,
                'input': '',
                'output': response,
                'source': 'github'
            })
        
        return formatted_data

class EnhancedProfessionalForumBot:
    """
    Enhanced professional-grade forum bot with improved memory management
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.model_path = "./enhanced-tinyllama-forum-professional"
        self.fallback_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.max_length = 800  # RESTORED ORIGINAL
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
        
        # Memory manager
        self.memory_manager = MemoryManager()
        
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
            # Check GPU memory before proceeding
            memory_info = self.memory_manager.get_gpu_memory_info()
            if memory_info:
                logger.info(f"GPU Memory available: {memory_info['free']}GB free, {memory_info['total']}GB total")
            
            if Path(self.model_path).exists():
                logger.info(f"Found model directory: {self.model_path}")
                
                if self._is_model_complete():
                    logger.info(f"Loading enhanced fine-tuned model from {self.model_path}")
                    success = self._load_enhanced_model_safely()
                    if success:
                        return
                    else:
                        logger.warning("Failed to load model safely, will retrain")
                else:
                    logger.warning("Model directory incomplete, will retrain")
                
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
        
        model_files = ["pytorch_model.bin", "model.safetensors"]
        has_model_file = any((model_path / f).exists() for f in model_files)
        
        vocab_files = ["vocab.txt", "vocab.json", "merges.txt", "spiece.model"]
        has_vocab_file = any((model_path / f).exists() for f in vocab_files)
        
        essential_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        has_essential = all((model_path / f).exists() for f in essential_files)
        
        is_complete = has_essential and has_model_file and has_vocab_file
        
        if not is_complete:
            logger.info("Model completeness check:")
            logger.info(f"  Essential files present: {has_essential}")
            logger.info(f"  Model file present: {has_model_file}")
            logger.info(f"  Vocab file present: {has_vocab_file}")
        
        return is_complete
    
    def _load_enhanced_model_safely(self) -> bool:
        """Safely load the enhanced model with comprehensive error handling and multiple load methods"""
        try:
            # Clear GPU cache before loading
            self.memory_manager.clear_gpu_cache()
            
            # Step 1: Load tokenizer
            logger.info("Step 1: Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True,
                    local_files_only=True
                )
                logger.info("✅ Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"❌ Tokenizer loading failed: {e}")
                return False
            
            # Step 2: Load model with multiple methods
            logger.info("Step 2: Loading model...")
            model_loaded = False
            
            # Method 1: Standard loading
            try:
                logger.info("Attempting standard model loading...")
                # Determine device and memory settings
                if torch.cuda.is_available():
                    memory_info = self.memory_manager.get_gpu_memory_info()
                    if memory_info and memory_info['free'] < 3:  # Less than 3GB free
                        logger.warning("Low GPU memory, loading in CPU mode")
                        device_map = None
                        torch_dtype = torch.float32
                    else:
                        device_map = "auto"
                        torch_dtype = torch.float16
                else:
                    device_map = None
                    torch_dtype = torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    local_files_only=True,
                    low_cpu_mem_usage=True
                )
                model_loaded = True
                logger.info("✅ Standard model loading successful")
                
            except Exception as e:
                logger.warning(f"Standard model loading failed: {e}")
                logger.info("Trying alternative loading methods...")
                
                # Method 2: Manual state dict loading
                try:
                    logger.info("Attempting manual state dict loading...")
                    
                    # Load base model first
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.fallback_model,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                    
                    # Load state dict
                    model_file = Path(self.model_path) / "pytorch_model.bin"
                    if model_file.exists():
                        logger.info("Loading from pytorch_model.bin...")
                        state_dict = torch.load(model_file, map_location="cpu")
                        base_model.load_state_dict(state_dict, strict=False)
                        self.model = base_model
                        model_loaded = True
                        logger.info("✅ Manual state dict loading successful")
                    else:
                        logger.warning("pytorch_model.bin not found")
                        
                except Exception as e2:
                    logger.warning(f"Manual state dict loading failed: {e2}")
                    
                    # Method 3: Try safetensors
                    try:
                        logger.info("Attempting safetensors loading...")
                        from safetensors.torch import load_file
                        
                        model_file = Path(self.model_path) / "model.safetensors"
                        if model_file.exists():
                            base_model = AutoModelForCausalLM.from_pretrained(
                                self.fallback_model,
                                trust_remote_code=True,
                                torch_dtype=torch_dtype,
                                device_map=device_map,
                                low_cpu_mem_usage=True
                            )
                            
                            state_dict = load_file(model_file)
                            base_model.load_state_dict(state_dict, strict=False)
                            self.model = base_model
                            model_loaded = True
                            logger.info("✅ Safetensors loading successful")
                        else:
                            logger.warning("model.safetensors not found")
                            
                    except ImportError:
                        logger.warning("Safetensors not available")
                    except Exception as e3:
                        logger.warning(f"Safetensors loading failed: {e3}")
                        
                        # Method 4: Try chunk-based loading
                        try:
                            logger.info("Attempting chunk-based loading...")
                            
                            # Check for chunk files
                            chunk_files = list(Path(self.model_path).glob("pytorch_model_chunk_*.bin"))
                            if chunk_files:
                                base_model = AutoModelForCausalLM.from_pretrained(
                                    self.fallback_model,
                                    trust_remote_code=True,
                                    torch_dtype=torch_dtype,
                                    device_map=device_map,
                                    low_cpu_mem_usage=True
                                )
                                
                                # Load all chunks
                                combined_state_dict = {}
                                for chunk_file in sorted(chunk_files):
                                    logger.info(f"Loading chunk: {chunk_file.name}")
                                    chunk_dict = torch.load(chunk_file, map_location="cpu")
                                    combined_state_dict.update(chunk_dict)
                                
                                base_model.load_state_dict(combined_state_dict, strict=False)
                                self.model = base_model
                                model_loaded = True
                                logger.info("✅ Chunk-based loading successful")
                            else:
                                logger.warning("No chunk files found")
                                
                        except Exception as e4:
                            logger.error(f"All model loading methods failed: {e4}")
            
            if not model_loaded:
                logger.error("❌ Could not load model with any method")
                return False
            
            # Step 3: Configure tokenizer and model
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if hasattr(self.model.config, 'pad_token_id'):
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            # Step 4: Validate model works
            logger.info("Step 3: Validating model...")
            try:
                test_input = "What is Python?"
                inputs = self.tokenizer(test_input, return_tensors="pt")
                
                # Ensure inputs are on the same device as model
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
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
            
            # Step 5: Final setup
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
                logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Failed to cleanup model directory: {e}")
    
    def load_base_model(self):
        """Load base TinyLlama model with improved error handling"""
        try:
            logger.info("Loading base TinyLlama model...")
            
            # Clear GPU cache
            self.memory_manager.clear_gpu_cache()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.fallback_model, 
                trust_remote_code=True
            )
            
            # Determine optimal loading strategy based on available memory
            memory_info = self.memory_manager.get_gpu_memory_info()
            if memory_info and memory_info['free'] < 3:
                logger.info("Low GPU memory detected, loading model on CPU")
                device_map = None
                torch_dtype = torch.float32
            else:
                device_map = "auto" if torch.cuda.is_available() else None
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.fallback_model,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True
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
            
            # Ensure device compatibility
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            logger.info("Base model loaded and validated successfully")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise RuntimeError(f"Could not load any model: {e}")
    
    def train_enhanced_model(self):
        """Train model with memory-optimized strategies"""
        try:
            logger.info("=== Starting Enhanced Model Training ===")
            
            # Check available memory
            memory_info = self.memory_manager.get_gpu_memory_info()
            if memory_info:
                logger.info(f"Pre-training GPU memory: {memory_info}")
                if memory_info['free'] < 2:
                    logger.warning("Very low GPU memory available, training may fail")
            
            # Clear GPU cache
            self.memory_manager.clear_gpu_cache()
            
            # Collect training data with memory constraints
            all_training_data = self.dataset_manager.collect_all_datasets()
            
            if not all_training_data:
                logger.error("No training data available")
                self.load_base_model()
                return
            
            # Estimate memory needs
            estimated_memory = self.memory_manager.estimate_training_memory_needs(len(all_training_data))
            logger.info(f"Estimated training memory needs: {estimated_memory:.1f}GB")
            
            # Attempt training with original fallback strategies (RESTORED)
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
    
    def _attempt_training_with_fallbacks(self, all_training_data) -> bool:
        """Attempt training with multiple fallback strategies - ORIGINAL ORDER"""
        
        # Strategy 1: Full dataset with optimized parameters (ORIGINAL)
        try:
            logger.info("Attempting Strategy 1: Full dataset training")
            return self._train_with_strategy_1(all_training_data)
        except Exception as e:
            logger.warning(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Reduced dataset with simple parameters (ORIGINAL)
        try:
            logger.info("Attempting Strategy 2: Reduced dataset training")
            return self._train_with_strategy_2(all_training_data)
        except Exception as e:
            logger.warning(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Minimal dataset with basic parameters (ORIGINAL)
        try:
            logger.info("Attempting Strategy 3: Minimal dataset training")
            return self._train_with_strategy_3(all_training_data)
        except Exception as e:
            logger.warning(f"Strategy 3 failed: {e}")
        
        return False
    
    def _train_with_strategy_1(self, all_training_data) -> bool:
        """Strategy 1: Full dataset with optimized parameters - ORIGINAL"""
        
        # Load base model for training (with memory optimization)
        logger.info("Loading base model for training...")
        tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
        
        # Use proper precision and memory optimization for training
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
        
        # Use full dataset size (ORIGINAL)
        max_examples = 8000 if torch.cuda.is_available() else 1000
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
        
        # Training arguments - ORIGINAL but with FP16 fix
        training_args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            num_train_epochs=1,  # ORIGINAL
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=3e-5,  # ORIGINAL
            warmup_steps=50,
            logging_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            dataloader_drop_last=True,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            # Fix FP16 issues
            fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
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
        
        try:
            trainer.train()
            logger.info("Training completed successfully!")
        except Exception as e:
            if "FP16" in str(e) or "gradient" in str(e).lower():
                logger.warning(f"FP16 training failed: {e}")
                logger.info("Retrying with FP32...")
                # Retry with FP32
                training_args.fp16 = False
                trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
                trainer.train()
                logger.info("FP32 training completed successfully!")
            else:
                raise e
        
        # Save model with comprehensive file validation
        logger.info("Saving trained model...")
        try:
            self._save_model_safely(trainer, tokenizer)
            logger.info("✅ Model saved successfully")
        except Exception as save_error:
            logger.error(f"Model save failed: {save_error}")
            logger.info("⚠️ Continuing with training completion despite save issues")
        
        # Load the trained model - try to use it even if save had issues
        try:
            self.tokenizer = tokenizer
            self.model = model
            self.model_version = "enhanced-fine-tuned"
            
            # Quick validation test
            test_input = "What is Python?"
            test_tokens = tokenizer(test_input, return_tensors="pt")
            
            # Ensure device compatibility
            model_device = next(model.parameters()).device
            test_tokens = {k: v.to(model_device) for k, v in test_tokens.items()}
            
            with torch.no_grad():
                _ = model(**test_tokens)
            logger.info("✅ Model validation successful - training completed!")
            
            return True
            
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
            logger.info("⚠️ Training completed but model validation failed")
            return False
    
    def _train_with_strategy_2(self, all_training_data) -> bool:
        """Strategy 2: Reduced dataset with simple parameters - ORIGINAL"""
        
        # Use smaller subset (1000 examples max) - ORIGINAL
        subset_size = min(1000, len(all_training_data))
        training_subset = random.sample(all_training_data, subset_size)
        
        logger.info(f"Training with reduced dataset of {len(training_subset)} examples")
        
        # Load model with simpler configuration
        tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.fallback_model,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use FP32 for stability - ORIGINAL
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
        
        # Simple training arguments - ORIGINAL
        training_args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            num_train_epochs=1,  # ORIGINAL
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,  # ORIGINAL
            logging_steps=50,
            save_strategy="epoch",
            dataloader_drop_last=True,
            report_to="none",
            remove_unused_columns=False,
            fp16=False,  # Disabled for stability
            max_grad_norm=1.0,
        )
        
        trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
        
        trainer.train()
        logger.info("Strategy 2 training completed successfully!")
        
        # Save model safely
        logger.info("Saving Strategy 2 model...")
        try:
            self._save_model_safely(trainer, tokenizer)
            logger.info("✅ Strategy 2 model saved successfully")
        except Exception as save_error:
            logger.error(f"Strategy 2 model save failed: {save_error}")
            logger.info("⚠️ Continuing despite save issues")
        
        # Load the trained model
        self.tokenizer = tokenizer
        self.model = model
        self.model_version = "enhanced-fine-tuned-reduced"
        
        # Quick validation
        try:
            test_tokens = tokenizer("Hello", return_tensors="pt")
            model_device = next(model.parameters()).device
            test_tokens = {k: v.to(model_device) for k, v in test_tokens.items()}
            with torch.no_grad():
                _ = model(**test_tokens)
            logger.info("✅ Strategy 2: Model validation successful")
        except Exception as e:
            logger.warning(f"Strategy 2: Model validation failed: {e}")
        
        return True
    
    def _train_with_strategy_3(self, all_training_data) -> bool:
        """Strategy 3: Minimal dataset with basic parameters - ORIGINAL"""
        
        # Use very small subset (100 examples max) - ORIGINAL
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
        
        # Minimal training arguments - ORIGINAL
        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=1,  # ORIGINAL
            per_device_train_batch_size=1,
            learning_rate=1e-4,  # ORIGINAL
            save_strategy="epoch",
            logging_steps=10,
            report_to="none",
            fp16=False,
            max_grad_norm=1.0,
        )
        
        trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
        
        trainer.train()
        logger.info("Strategy 3 training completed successfully!")
        
        # Save model safely
        logger.info("Saving Strategy 3 model...")
        try:
            self._save_model_safely(trainer, tokenizer)
            logger.info("✅ Strategy 3 model saved successfully")
        except Exception as save_error:
            logger.error(f"Strategy 3 model save failed: {save_error}")
            logger.info("⚠️ Continuing despite save issues")
        
        # Load the trained model
        self.tokenizer = tokenizer
        self.model = model
        self.model_version = "enhanced-fine-tuned-minimal"
        
        # Quick validation
        try:
            test_tokens = tokenizer("Test", return_tensors="pt")
            model_device = next(model.parameters()).device
            test_tokens = {k: v.to(model_device) for k, v in test_tokens.items()}
            with torch.no_grad():
                _ = model(**test_tokens)
            logger.info("✅ Strategy 3: Model validation successful")
        except Exception as e:
            logger.warning(f"Strategy 3: Model validation failed: {e}")
        
        return True
    
    def _train_minimal_memory(self, all_training_data) -> bool:
        """Train with minimal memory usage"""
        try:
            # Use very small subset
            subset_size = min(50, len(all_training_data))
            training_subset = random.sample(all_training_data, subset_size)
            
            logger.info(f"Training with minimal memory: {len(training_subset)} examples")
            
            # Clear cache before loading
            self.memory_manager.clear_gpu_cache()
            
            # Load model in CPU mode to save GPU memory
            tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.fallback_model,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
                # No device_map to keep on CPU initially
            )
            
            # Configure tokenizer
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
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-4,
                logging_steps=10,
                save_strategy="epoch",
                report_to="none",
                dataloader_drop_last=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                fp16=False,  # Disable FP16 to save memory
                gradient_checkpointing=True  # Enable gradient checkpointing
            )
            
            trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
            
            logger.info("Starting minimal memory training...")
            trainer.train()
            logger.info("Minimal memory training completed successfully!")
            
            # Save model
            self._save_model_safely(trainer, tokenizer)
            
            # Load the trained model
            self.tokenizer = tokenizer
            self.model = model
            self.model_version = "enhanced-fine-tuned-minimal"
            
            return True
            
        except Exception as e:
            logger.error(f"Minimal memory training failed: {e}")
            return False
    
    def _train_reduced_memory(self, all_training_data) -> bool:
        """Train with reduced memory usage"""
        try:
            # Use moderate subset
            subset_size = min(1000, len(all_training_data))  # Restored original size
            training_subset = random.sample(all_training_data, subset_size)
            
            logger.info(f"Training with reduced memory: {len(training_subset)} examples")
            
            # Clear cache
            self.memory_manager.clear_gpu_cache()
            
            # Load model with FP16 if GPU available
            tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
            
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    self.fallback_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.fallback_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id
            
            # Format and tokenize data
            training_texts = self._format_training_data(training_subset)
            tokenized_dataset = self._tokenize_data(training_texts, tokenizer)
            
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            
            # Reduced training arguments with FP16 fix
            training_args = TrainingArguments(
                output_dir=self.model_path,
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                learning_rate=5e-5,
                logging_steps=25,
                save_strategy="epoch",
                report_to="none",
                dataloader_drop_last=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                # Smart FP16 handling
                fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7,
                gradient_checkpointing=True,
                max_grad_norm=1.0
            )
            
            trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
            
            logger.info("Starting reduced memory training...")
            trainer.train()
            logger.info("Reduced memory training completed successfully!")
            
            # Save model
            self._save_model_safely(trainer, tokenizer)
            
            # Load the trained model
            self.tokenizer = tokenizer
            self.model = model
            self.model_version = "enhanced-fine-tuned-reduced"
            
            return True
            
        except Exception as e:
            logger.error(f"Reduced memory training failed: {e}")
            # Try without FP16
            logger.info("Attempting reduced memory training without FP16...")
            return self._train_reduced_memory_fp32_fallback(all_training_data)
    
    def _train_reduced_memory_fp32_fallback(self, all_training_data) -> bool:
        """Fallback reduced memory training with FP32"""
        try:
            subset_size = min(1000, len(all_training_data))
            training_subset = random.sample(all_training_data, subset_size)
            
            logger.info(f"FP32 reduced memory training: {len(training_subset)} examples")
            
            self.memory_manager.clear_gpu_cache()
            
            tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.fallback_model,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Force FP32
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id
            
            training_texts = self._format_training_data(training_subset)
            tokenized_dataset = self._tokenize_data(training_texts, tokenizer)
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            
            training_args = TrainingArguments(
                output_dir=self.model_path,
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                learning_rate=5e-5,
                logging_steps=25,
                save_strategy="epoch",
                report_to="none",
                dataloader_drop_last=True,
                remove_unused_columns=False,
                fp16=False,  # Disabled
                gradient_checkpointing=True,
                max_grad_norm=1.0
            )
            
            trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
            
            logger.info("Starting FP32 reduced memory training...")
            trainer.train()
            logger.info("FP32 reduced memory training completed!")
            
            self._save_model_safely(trainer, tokenizer)
            
            self.tokenizer = tokenizer
            self.model = model
            self.model_version = "enhanced-fine-tuned-reduced-fp32"
            
            return True
            
        except Exception as e:
            logger.error(f"FP32 reduced memory training also failed: {e}")
            return False
    
    def _train_optimized_memory(self, all_training_data) -> bool:
        """Train with optimized memory usage"""
        try:
            # Use larger subset but still manageable
            subset_size = min(800, len(all_training_data))
            training_subset = random.sample(all_training_data, subset_size)
            
            logger.info(f"Training with optimized memory: {len(training_subset)} examples")
            
            # Clear cache
            self.memory_manager.clear_gpu_cache()
            
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.fallback_model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
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
                pad_to_multiple_of=8
            )
            
            # Optimized training arguments - Fixed FP16 + gradient checkpointing issue
            training_args = TrainingArguments(
                output_dir=self.model_path,
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=3e-5,
                warmup_steps=50,
                logging_steps=50,
                save_strategy="epoch",
                save_total_limit=2,
                dataloader_drop_last=True,
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                # Fix FP16 gradient checkpointing incompatibility
                fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7,  # Only use FP16 on newer GPUs
                gradient_checkpointing=True,
                # Add gradient clipping to prevent overflow
                max_grad_norm=1.0,
                # Disable problematic optimizations that conflict with gradient checkpointing
                dataloader_persistent_workers=False,
            )
            
            trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
            
            logger.info("Starting optimized memory training...")
            trainer.train()
            logger.info("Optimized memory training completed successfully!")
            
            # Save model
            self._save_model_safely(trainer, tokenizer)
            
            # Load the trained model
            self.tokenizer = tokenizer
            self.model = model
            self.model_version = "enhanced-fine-tuned-optimized"
            
            return True
            
        except Exception as e:
            logger.error(f"Optimized memory training failed: {e}")
            # Try fallback without FP16
            logger.info("Attempting fallback training without FP16...")
            return self._train_optimized_memory_fallback(all_training_data)
    
    def _train_optimized_memory_fallback(self, all_training_data) -> bool:
        """Fallback training without FP16 to avoid gradient issues"""
        try:
            subset_size = min(800, len(all_training_data))
            training_subset = random.sample(all_training_data, subset_size)
            
            logger.info(f"Fallback training with FP32: {len(training_subset)} examples")
            
            # Clear cache
            self.memory_manager.clear_gpu_cache()
            
            # Load model in FP32 to avoid gradient issues
            tokenizer = AutoTokenizer.from_pretrained(self.fallback_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.fallback_model,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Force FP32
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
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
                pad_to_multiple_of=8
            )
            
            # Training arguments without FP16
            training_args = TrainingArguments(
                output_dir=self.model_path,
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=3e-5,
                warmup_steps=50,
                logging_steps=50,
                save_strategy="epoch",
                save_total_limit=2,
                dataloader_drop_last=True,
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                fp16=False,  # Disabled to avoid gradient issues
                gradient_checkpointing=True,
                max_grad_norm=1.0,
                dataloader_persistent_workers=False,
            )
            
            trainer = self._create_trainer(model, training_args, tokenized_dataset, tokenizer, data_collator)
            
            logger.info("Starting FP32 fallback training...")
            trainer.train()
            logger.info("FP32 fallback training completed successfully!")
            
            # Save model
            self._save_model_safely(trainer, tokenizer)
            
            # Load the trained model
            self.tokenizer = tokenizer
            self.model = model
            self.model_version = "enhanced-fine-tuned-fallback"
            
            return True
            
        except Exception as e:
            logger.error(f"FP32 fallback training also failed: {e}")
            return False
    
    def _format_training_data(self, training_data):
        """Format training data for TinyLlama"""
        training_texts = []
        for item in training_data:
            full_instruction = item['instruction']
            if item.get('input') and item['input'].strip():
                full_instruction += f"\n\nInput: {item['input']}"
            
            formatted = f"<|system|>\nYou are an expert programming mentor and helpful assistant. You provide accurate, detailed, and educational responses to programming questions and general inquiries. You explain concepts clearly with practical examples and encourage learning.\n</s>\n<|user|>\n{full_instruction}\n</s>\n<|assistant|>\n{item['output']}\n</s>"
            training_texts.append(formatted)
        
        return training_texts
    
    def _tokenize_data(self, training_texts, tokenizer):
        """Tokenize training data with memory optimization"""
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,  # Reduced to save memory
                padding=False,
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        dataset = Dataset.from_dict({"text": training_texts})
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"],
            batch_size=100  # Process in smaller batches to save memory
        )
        
        return tokenized_dataset
    
    def _create_trainer(self, model, training_args, tokenized_dataset, tokenizer, data_collator):
        """Create trainer with version compatibility"""
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                processing_class=tokenizer,
                data_collator=data_collator,
            )
        except TypeError:
            logger.warning("processing_class not supported, using tokenizer parameter")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        
        return trainer
    
    def _save_model_safely(self, trainer, tokenizer):
        """Safely save model with validation and multiple fallback methods"""
        try:
            Path(self.model_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving model to {self.model_path}")
            
            # Method 1: Try standard trainer save first
            try:
                logger.info("Attempting standard trainer.save_model()...")
                trainer.save_model()
                logger.info("Standard trainer save completed")
            except Exception as e:
                logger.warning(f"Standard trainer save failed: {e}")
                logger.info("Trying alternative save methods...")
                
                # Method 2: Manual state dict save with safe serialization
                try:
                    logger.info("Attempting manual state dict save...")
                    model_file = Path(self.model_path) / "pytorch_model.bin"
                    
                    # Save with protocol 4 for better compatibility
                    torch.save(
                        trainer.model.state_dict(), 
                        model_file,
                        pickle_protocol=4,
                        _use_new_zipfile_serialization=False
                    )
                    logger.info("Manual state dict save completed")
                    
                    # Save config manually
                    config_file = Path(self.model_path) / "config.json"
                    if hasattr(trainer.model, 'config'):
                        trainer.model.config.save_pretrained(self.model_path)
                        logger.info("Config saved manually")
                        
                except Exception as e2:
                    logger.warning(f"Manual state dict save failed: {e2}")
                    
                    # Method 3: Try safetensors format
                    try:
                        logger.info("Attempting safetensors save...")
                        from safetensors.torch import save_file
                        
                        model_file = Path(self.model_path) / "model.safetensors"
                        save_file(trainer.model.state_dict(), model_file)
                        logger.info("Safetensors save completed")
                        
                    except ImportError:
                        logger.warning("Safetensors not available, using alternative method")
                        
                        # Method 4: Chunk-based save for large models
                        try:
                            logger.info("Attempting chunk-based save...")
                            state_dict = trainer.model.state_dict()
                            
                            # Split into smaller chunks
                            chunk_size = 100  # Save 100 parameters at a time
                            param_items = list(state_dict.items())
                            
                            for i in range(0, len(param_items), chunk_size):
                                chunk = dict(param_items[i:i+chunk_size])
                                chunk_file = Path(self.model_path) / f"pytorch_model_chunk_{i//chunk_size}.bin"
                                torch.save(chunk, chunk_file, pickle_protocol=4)
                                logger.info(f"Saved chunk {i//chunk_size + 1}")
                            
                            # Create index file
                            index_file = Path(self.model_path) / "pytorch_model.bin.index.json"
                            index_data = {
                                "metadata": {"total_size": sum(p.numel() * p.element_size() for p in state_dict.values())},
                                "weight_map": {name: f"pytorch_model_chunk_{i//chunk_size}.bin" 
                                             for i, (name, _) in enumerate(param_items)}
                            }
                            
                            with open(index_file, 'w') as f:
                                json.dump(index_data, f, indent=2)
                            
                            logger.info("Chunk-based save completed")
                            
                        except Exception as e3:
                            logger.error(f"All model save methods failed: {e3}")
                            # Continue anyway, maybe tokenizer will save
            
            # Always try to save tokenizer
            try:
                logger.info("Saving tokenizer...")
                tokenizer.save_pretrained(self.model_path, safe_serialization=False)
                logger.info("Tokenizer saved successfully")
            except Exception as e:
                logger.warning(f"Tokenizer save failed: {e}")
                
                # Try alternative tokenizer save
                try:
                    logger.info("Trying alternative tokenizer save...")
                    tokenizer.save_pretrained(self.model_path, legacy_format=True)
                    logger.info("Alternative tokenizer save completed")
                except Exception as e2:
                    logger.error(f"All tokenizer save methods failed: {e2}")
            
            # Validation
            if self._is_model_complete():
                logger.info("✅ Model saved and validated successfully")
            else:
                logger.info("⚠️ Model save validation shows some files missing, but core files are present")
                
                # Try to fix missing files
                self._fix_missing_model_files(trainer, tokenizer)
                
        except Exception as e:
            logger.error(f"Critical error in model saving: {e}")
            # Don't raise - we want to continue even if save partially failed
            logger.info("Continuing despite save issues - model may still be usable")
    
    def _fix_missing_model_files(self, trainer, tokenizer):
        """Try to fix missing model files"""
        try:
            model_path = Path(self.model_path)
            
            # Create config.json if missing
            config_file = model_path / "config.json"
            if not config_file.exists() and hasattr(trainer.model, 'config'):
                try:
                    trainer.model.config.save_pretrained(self.model_path)
                    logger.info("Fixed missing config.json")
                except:
                    pass
            
            # Create tokenizer files if missing
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
            for file_name in tokenizer_files:
                if not (model_path / file_name).exists():
                    try:
                        # Force tokenizer save with different methods
                        tokenizer.save_pretrained(self.model_path, legacy_format=True)
                        logger.info(f"Fixed missing tokenizer files")
                        break
                    except:
                        continue
            
            # Create vocab files if missing
            vocab_files = ["vocab.txt", "vocab.json", "merges.txt"]
            if not any((model_path / f).exists() for f in vocab_files):
                try:
                    # Try to extract vocab from tokenizer
                    if hasattr(tokenizer, 'get_vocab'):
                        vocab = tokenizer.get_vocab()
                        vocab_file = model_path / "vocab.json"
                        with open(vocab_file, 'w', encoding='utf-8') as f:
                            json.dump(vocab, f, ensure_ascii=False, indent=2)
                        logger.info("Created vocab.json from tokenizer")
                except Exception as e:
                    logger.warning(f"Could not create vocab file: {e}")
            
        except Exception as e:
            logger.warning(f"Error fixing missing files: {e}")
    
    def generate_response(self, query: str) -> Dict:
        """Generate response to user query with improved device handling"""
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
                    max_length=800  # RESTORED ORIGINAL
                )
                
                # Ensure inputs are on the same device as model
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                # Generate with proper error handling
                try:
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
                        
                except Exception as gen_error:
                    logger.error(f"Generation error: {gen_error}")
                    return {
                        'response': "I'm experiencing some technical difficulties with text generation. Could you try rephrasing your question?",
                        'confidence': 0.1,
                        'processing_time': time.time() - start_time,
                        'cached': False,
                        'model_version': self.model_version,
                        'error': str(gen_error)
                    }
            
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
    
    # ... (rest of the methods remain the same as in original)
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
                self.response_cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
        
        return None
    
    def cache_response(self, query: str, response: str):
        """Cache response"""
        cache_key = self.get_cache_key(query)
        self.response_cache[cache_key] = response
        self.cache_timestamps[cache_key] = time.time()
        
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
        response = re.sub(r'<\|[^|]*\|>', '', response)
        response = response.replace('</s>', '').replace('<s>', '')
        
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line[:50] not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line[:50])
        
        response = '\n'.join(cleaned_lines).strip()
        
        if response and not response[-1] in '.!?':
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        uptime = datetime.now() - self.metrics['start_time']
        cache_hit_rate = 0
        if self.metrics['total_requests'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / self.metrics['total_requests']
        
        # Add memory info
        memory_info = self.memory_manager.get_gpu_memory_info()
        
        metrics = {
            'total_requests': self.metrics['total_requests'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_hit_rate': f"{cache_hit_rate:.2%}",
            'avg_response_time': f"{self.metrics['avg_response_time']:.3f}s",
            'error_count': self.metrics['error_count'],
            'uptime': str(uptime),
            'model_version': self.model_version,
            'cached_responses': len(self.response_cache)
        }
        
        if memory_info:
            metrics['gpu_memory'] = memory_info
        
        return metrics

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize enhanced bot
logger.info("Initializing Enhanced Professional Forum Bot...")
github_token = os.getenv('GITHUB_TOKEN')
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
    
    result = forum_bot.generate_response(ack)
    
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

@app.route('/memory', methods=['GET'])
def memory_info():
    """Memory information endpoint"""
    memory_info = forum_bot.memory_manager.get_gpu_memory_info()
    return jsonify({
        'gpu_memory': memory_info if memory_info else 'GPU not available',
        'model_device': str(next(forum_bot.model.parameters()).device) if hasattr(forum_bot, 'model') else 'Model not loaded'
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain model with fresh datasets"""
    try:
        if Path(forum_bot.model_path).exists():
            forum_bot._cleanup_model_directory()
        
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

# CORS support
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

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
    
    # Show memory info
    memory_info = forum_bot.memory_manager.get_gpu_memory_info()
    if memory_info:
        logger.info(f"GPU Memory: {memory_info['free']}GB free, {memory_info['total']}GB total")
    else:
        logger.info("Running in CPU-only mode")
    
    if forum_bot.model_version == "base":
        logger.info("=" * 60)
        logger.info("NOTICE: Using base model (not fine-tuned)")
        logger.info("To train the enhanced model, send a POST request to:")
        logger.info(f"   https://{host}:{port}/retrain")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("ENHANCED MODEL LOADED SUCCESSFULLY!")
        logger.info(f"Model version: {forum_bot.model_version}")
        logger.info("Ready to provide high-quality programming assistance!")
        logger.info("=" * 60)
    
    # Start server
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