import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import json
import random
import os
import ssl
import threading
import requests
import asyncio
import aiohttp
import mysql.connector
from mysql.connector import Error
import wikipedia
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse, unquote
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import math
import time
from pathlib import Path
import logging
import re
from datetime import datetime
from collections import defaultdict
import sys

# NLTK handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  Warning: NLTK not available. Install with: pip install nltk")

# Transformers handling
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: transformers not available. Install with: pip install transformers torch")

# Sentence transformers handling
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FIXED DATACLASSES - Proper field ordering
# =============================================================================

@dataclass
class ProcessingStep:
    step_name: str
    input_data: Any
    output_data: Any
    name: str = "Unknown Step"
    status: str = "pending"
    duration: float = 0.0
    processing_time: float = 0.0
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ContextualGrade:
    # Required fields first with defaults
    relevance_score: float = 0.0
    topic_continuity: float = 0.0
    context_shift_health: float = 0.0
    recall_efficiency: float = 0.0
    uniqueness_score: float = 0.0
    overall_grade: float = 0.0
    # Optional fields with defaults
    score: float = 0.0
    feedback: str = ""

@dataclass
class VerboseResponse:
    processing_steps: List[ProcessingStep]
    total_processing_time: float
    pipeline_summary: str
    contextual_analysis: Dict[str, Any]
    database_interactions: List[Dict]
    wikipedia_queries: List[Dict]
    contextual_grade: ContextualGrade
    gemini_response: str = ""
    tinyllama_response: str = ""
    homebrew_response: str = ""
    final_response: str = ""

@dataclass
class ConversationTurn:
    user_query: str
    gemini_response: str
    tinyllama_response: str
    homebrew_response: str
    verbose_response: VerboseResponse
    anxiety_level: float
    emotion_state: Dict[str, float]
    confidence_score: float
    context_richness: float
    timestamp: float
    session_id: str
    correction_applied: bool = False

@dataclass
class ClientSession:
    session_id: str
    gemini_api_key: str
    created_at: float
    last_active: float
    conversation_count: int = 0

# =============================================================================
# SINGLETON COMPONENT MANAGER - Prevents duplicate loading
# =============================================================================

class ComponentManager:
    """Singleton manager to ensure components are only loaded once"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ComponentManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ComponentManager._initialized:
            self.db = None
            self.wiki = None
            self.tinyllama = None
            self.emotion_detector = None
            self._lock = threading.Lock()
            ComponentManager._initialized = True
            logger.info("🔧 Component Manager initialized")
    
    def get_database(self, host='localhost', database='reservesphp', user='root', password=''):
        """Get or create database connection (singleton)"""
        if self.db is None:
            with self._lock:
                if self.db is None:  # Double-check locking
                    try:
                        self.db = MySQLContextDatabase(host, database, user, password)
                        logger.info(f"✅ Database initialized: {user}@{host}/{database}")
                    except Exception as e:
                        logger.error(f"❌ Database initialization failed: {e}")
                        self.db = None
        return self.db
    
    def get_wikipedia(self):
        """Get or create Wikipedia builder (singleton)"""
        if self.wiki is None:
            with self._lock:
                if self.wiki is None:  # Double-check locking
                    try:
                        self.wiki = WikipediaContextBuilder()
                        logger.info("✅ Wikipedia integration initialized")
                    except Exception as e:
                        logger.error(f"❌ Wikipedia initialization failed: {e}")
                        self.wiki = None
        return self.wiki
    
    def get_tinyllama(self, model_path="./tinyllama-forum-professional"):
        """Get or create TinyLlama integration (singleton)"""
        if self.tinyllama is None:
            with self._lock:
                if self.tinyllama is None:  # Double-check locking
                    try:
                        self.tinyllama = TinyLlamaIntegration()
                        if model_path and os.path.exists(model_path):
                            self.tinyllama.model_path = model_path
                        self.tinyllama.setup_model()
                        logger.info(f"✅ TinyLlama initialized: {getattr(self.tinyllama, 'model_version', 'fallback')}")
                    except Exception as e:
                        logger.error(f"❌ TinyLlama initialization failed: {e}")
                        self.tinyllama = TinyLlamaIntegration()  # Fallback
        return self.tinyllama
    
    def get_emotion_detector(self):
        """Get or create emotion detector (singleton)"""
        if self.emotion_detector is None:
            with self._lock:
                if self.emotion_detector is None:  # Double-check locking
                    try:
                        self.emotion_detector = EmotionDetector()
                        logger.info("✅ Emotion detector initialized")
                    except Exception as e:
                        logger.error(f"❌ Emotion detector initialization failed: {e}")
                        self.emotion_detector = EmotionDetector()  # Fallback
        return self.emotion_detector

# =============================================================================
# CORE COMPONENTS - Unchanged but optimized
# =============================================================================

class MySQLContextDatabase:
    """MySQL database integration for context building and word relationships"""
    
    def __init__(self, host='localhost', database='reservesphp', user='root', password=''):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.connection_lock = threading.Lock()
        
        self.initialize_database()
        self.ensure_tables_exist()
    
    def initialize_database(self):
        """Initialize MySQL connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                autocommit=True
            )
            
            if self.connection.is_connected():
                logger.info(f"Successfully connected to MySQL database: {self.database}")
            
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            self.connection = None
    
    def ensure_tables_exist(self):
        """Create tables if they don't exist"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            
            # Create word table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS word (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    word VARCHAR(255) UNIQUE NOT NULL,
                    related_word TEXT,
                    related_topic TEXT,
                    definition TEXT,
                    usage_count INT DEFAULT 0,
                    context_score FLOAT DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_word (word),
                    INDEX idx_usage_count (usage_count)
                )
            """)
            
            # Create sentences table (note: keeping original "sentances" name for compatibility)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentances (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    sentance TEXT NOT NULL,
                    name VARCHAR(255),
                    topic_category VARCHAR(255),
                    quality_score FLOAT DEFAULT 0.0,
                    usage_count INT DEFAULT 0,
                    context_relevance FLOAT DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_name (name),
                    INDEX idx_topic_category (topic_category),
                    INDEX idx_quality_score (quality_score)
                )
            """)
            
            # Create conversation_context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_context (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    turn_number INT NOT NULL,
                    primary_topic VARCHAR(255),
                    secondary_topics TEXT,
                    context_keywords TEXT,
                    topic_shift_score FLOAT DEFAULT 0.0,
                    relevance_to_previous FLOAT DEFAULT 0.0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session_id (session_id),
                    INDEX idx_primary_topic (primary_topic)
                )
            """)
            
            # Create response_patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS response_patterns (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    topic_signature VARCHAR(255) NOT NULL,
                    response_template TEXT NOT NULL,
                    usage_frequency INT DEFAULT 1,
                    success_rate FLOAT DEFAULT 1.0,
                    avg_response_time FLOAT DEFAULT 0.0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_topic_signature (topic_signature),
                    INDEX idx_usage_frequency (usage_frequency)
                )
            """)
            
            self.connection.commit()
            logger.info("Database tables ensured to exist")
            
        except Error as e:
            logger.error(f"Error creating tables: {e}")
    
    def reconnect_if_needed(self):
        """Reconnect to database if connection is lost"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.initialize_database()
        except Error as e:
            logger.error(f"Error reconnecting to database: {e}")
    
    def store_word_relationship(self, word: str, related_words: List[str], related_topics: List[str], definition: str):
        """Store word relationships in database"""
        if not self.connection:
            return
        
        try:
            with self.connection_lock:
                self.reconnect_if_needed()
                cursor = self.connection.cursor()
                
                related_words_json = json.dumps(related_words)
                related_topics_json = json.dumps(related_topics)
                
                query = """
                    INSERT INTO word (word, related_word, related_topic, definition, usage_count)
                    VALUES (%s, %s, %s, %s, 1)
                    ON DUPLICATE KEY UPDATE
                    related_word = VALUES(related_word),
                    related_topic = VALUES(related_topic),
                    definition = VALUES(definition),
                    usage_count = usage_count + 1,
                    context_score = (context_score + 0.1)
                """
                
                cursor.execute(query, (word, related_words_json, related_topics_json, definition))
                self.connection.commit()
                
        except Error as e:
            logger.error(f"Error storing word relationship: {e}")
    
    def get_word_context(self, word: str) -> Dict:
        """Retrieve word context from database"""
        if not self.connection:
            return {}
        
        try:
            with self.connection_lock:
                self.reconnect_if_needed()
                cursor = self.connection.cursor(dictionary=True)
                
                query = "SELECT * FROM word WHERE word = %s"
                cursor.execute(query, (word,))
                result = cursor.fetchone()
                
                if result:
                    if result['related_word']:
                        try:
                            result['related_word'] = json.loads(result['related_word'])
                        except:
                            result['related_word'] = []
                    if result['related_topic']:
                        try:
                            result['related_topic'] = json.loads(result['related_topic'])
                        except:
                            result['related_topic'] = []
                    
                    return result
                
                return {}
                
        except Error as e:
            logger.error(f"Error retrieving word context: {e}")
            return {}
    
    def store_sentence_pattern(self, sentence: str, name: str, topic_category: str, quality_score: float):
        """Store sentence patterns for reuse"""
        if not self.connection:
            return
        
        try:
            with self.connection_lock:
                self.reconnect_if_needed()
                cursor = self.connection.cursor()
                
                query = """
                    INSERT INTO sentances (sentance, name, topic_category, quality_score, usage_count)
                    VALUES (%s, %s, %s, %s, 1)
                    ON DUPLICATE KEY UPDATE
                    usage_count = usage_count + 1,
                    quality_score = (quality_score + VALUES(quality_score)) / 2
                """
                
                cursor.execute(query, (sentence, name, topic_category, quality_score))
                self.connection.commit()
                
        except Error as e:
            logger.error(f"Error storing sentence pattern: {e}")
    
    def get_sentence_patterns(self, topic_category: str, limit: int = 10) -> List[Dict]:
        """Retrieve sentence patterns for a topic"""
        if not self.connection:
            return []
        
        try:
            with self.connection_lock:
                self.reconnect_if_needed()
                cursor = self.connection.cursor(dictionary=True)
                
                query = """
                    SELECT * FROM sentances 
                    WHERE topic_category = %s 
                    ORDER BY quality_score DESC, usage_count DESC 
                    LIMIT %s
                """
                
                cursor.execute(query, (topic_category, limit))
                return cursor.fetchall()
                
        except Error as e:
            logger.error(f"Error retrieving sentence patterns: {e}")
            return []
    
    def store_conversation_context(self, session_id: str, turn_number: int, primary_topic: str, 
                                 secondary_topics: List[str], context_keywords: List[str], 
                                 topic_shift_score: float, relevance_to_previous: float):
        """Store conversation context for topic evolution tracking"""
        if not self.connection:
            return
        
        try:
            with self.connection_lock:
                self.reconnect_if_needed()
                cursor = self.connection.cursor()
                
                secondary_topics_json = json.dumps(secondary_topics)
                context_keywords_json = json.dumps(context_keywords)
                
                query = """
                    INSERT INTO conversation_context 
                    (session_id, turn_number, primary_topic, secondary_topics, context_keywords, 
                     topic_shift_score, relevance_to_previous)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(query, (session_id, turn_number, primary_topic, secondary_topics_json, 
                                     context_keywords_json, topic_shift_score, relevance_to_previous))
                self.connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing conversation context: {e}")
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation context for a session"""
        if not self.connection:
            return []
        
        try:
            with self.connection_lock:
                self.reconnect_if_needed()
                cursor = self.connection.cursor(dictionary=True)
                
                query = """
                    SELECT * FROM conversation_context 
                    WHERE session_id = %s 
                    ORDER BY turn_number DESC 
                    LIMIT %s
                """
                
                cursor.execute(query, (session_id, limit))
                results = cursor.fetchall()
                
                for result in results:
                    if result['secondary_topics']:
                        try:
                            result['secondary_topics'] = json.loads(result['secondary_topics'])
                        except:
                            result['secondary_topics'] = []
                    if result['context_keywords']:
                        try:
                            result['context_keywords'] = json.loads(result['context_keywords'])
                        except:
                            result['context_keywords'] = []
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def store_response_pattern(self, topic_signature: str, response_template: str, response_time: float):
        """Store response patterns for faster recall"""
        if not self.connection:
            return
        
        try:
            with self.connection_lock:
                self.reconnect_if_needed()
                cursor = self.connection.cursor()
                
                query = """
                    INSERT INTO response_patterns (topic_signature, response_template, avg_response_time)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    usage_frequency = usage_frequency + 1,
                    avg_response_time = (avg_response_time + VALUES(avg_response_time)) / 2,
                    last_used = CURRENT_TIMESTAMP
                """
                
                cursor.execute(query, (topic_signature, response_template, response_time))
                self.connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing response pattern: {e}")
    
    def get_response_pattern(self, topic_signature: str) -> Dict:
        """Retrieve cached response pattern for faster processing"""
        if not self.connection:
            return {}
        
        try:
            with self.connection_lock:
                self.reconnect_if_needed()
                cursor = self.connection.cursor(dictionary=True)
                
                query = """
                    SELECT * FROM response_patterns 
                    WHERE topic_signature = %s 
                    ORDER BY usage_frequency DESC, success_rate DESC 
                    LIMIT 1
                """
                
                cursor.execute(query, (topic_signature,))
                result = cursor.fetchone()
                
                return result if result else {}
                
        except Exception as e:
            logger.error(f"Error retrieving response pattern: {e}")
            return {}

class WikipediaContextBuilder:
    """Wikipedia integration for building rich contextual information"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 3600  # 1 hour cache
        wikipedia.set_lang("en")
    
    def clean_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def get_wikipedia_context(self, topic: str, max_sentences: int = 3) -> Dict:
        """Get contextual information from Wikipedia"""
        cache_key = f"{topic}_{max_sentences}"
        current_time = time.time()
        
        if cache_key in self.cache:
            if current_time - self.cache_timestamps[cache_key] < self.cache_ttl:
                return self.cache[cache_key]
        
        try:
            search_results = wikipedia.search(topic, results=3)
            
            if not search_results:
                return {'error': 'No Wikipedia results found', 'topic': topic}
            
            page_title = search_results[0]
            page = wikipedia.page(page_title)
            
            summary_sentences = sent_tokenize(page.summary) if NLTK_AVAILABLE else page.summary.split('. ')
            selected_sentences = summary_sentences[:max_sentences]
            
            related_links = page.links[:10] if hasattr(page, 'links') else []
            categories = page.categories[:5] if hasattr(page, 'categories') else []
            
            context_data = {
                'title': page.title,
                'summary_sentences': selected_sentences,
                'related_topics': related_links,
                'categories': categories,
                'url': page.url,
                'full_summary': page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                'success': True,
                'topic': topic
            }
            
            self.cache[cache_key] = context_data
            self.cache_timestamps[cache_key] = current_time
            
            if len(self.cache) > 100:
                self.clean_cache()
            
            return context_data
            
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0])
                summary_sentences = sent_tokenize(page.summary) if NLTK_AVAILABLE else page.summary.split('. ')
                
                context_data = {
                    'title': page.title,
                    'summary_sentences': summary_sentences[:max_sentences],
                    'related_topics': page.links[:10] if hasattr(page, 'links') else [],
                    'categories': page.categories[:5] if hasattr(page, 'categories') else [],
                    'url': page.url,
                    'disambiguation_handled': True,
                    'original_options': e.options[:5],
                    'success': True,
                    'topic': topic
                }
                
                self.cache[cache_key] = context_data
                self.cache_timestamps[cache_key] = current_time
                
                return context_data
                
            except Exception as inner_e:
                return {
                    'error': f'Disambiguation error: {str(inner_e)}',
                    'options': e.options[:5],
                    'success': False,
                    'topic': topic
                }
        
        except wikipedia.exceptions.PageError:
            return {
                'error': 'Wikipedia page not found',
                'success': False,
                'topic': topic
            }
        
        except Exception as e:
            return {
                'error': f'Wikipedia error: {str(e)}',
                'success': False,
                'topic': topic
            }
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for Wikipedia lookup"""
        if not NLTK_AVAILABLE:
            words = text.split()
            concepts = [word for word in words if len(word) > 4 and word.isalpha()]
            return list(set(concepts))[:5]
        
        try:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text.lower())
            
            filtered_words = [
                word for word in word_tokens 
                if word.isalpha() and len(word) > 3 and word not in stop_words
            ]
            
            word_freq = defaultdict(int)
            for word in filtered_words:
                word_freq[word] += 1
            
            sorted_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [concept[0] for concept in sorted_concepts[:5]]
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {e}")
            return []

class TinyLlamaIntegration:
    """TinyLlama integration with fallback support"""
    def __init__(self):
        self.model_path = "./tinyllama-forum-professional"
        self.fallback_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.max_length = 400
        self.temperature = 0.8
        
        self.model_lock = threading.Lock()
        self.response_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 1800  # 30 minutes
        
        self.model = None
        self.tokenizer = None
        self.model_version = "not_loaded"
    
    def setup_model(self):
        """Load TinyLlama model with detailed logging"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using fallback responses")
            self.model_version = "fallback"
            return
        
        try:
            if Path(self.model_path).exists():
                logger.info(f"Loading fine-tuned TinyLlama from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.model_version = "fine-tuned"
            else:
                logger.info("Fine-tuned model not found, using base model")
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
            logger.info(f"TinyLlama model loaded successfully ({self.model_version})")
            
        except Exception as e:
            logger.error(f"Failed to load TinyLlama model: {e}")
            self.model = None
            self.tokenizer = None
            self.model_version = "fallback"
    
    def generate_response_with_step(self, query: str, gemini_guidance: str = "", step_name: str = "TinyLlama Processing") -> Tuple[str, ProcessingStep]:
        """Generate response with detailed step tracking"""
        start_time = time.time()
        
        if self.model is None or self.tokenizer is None:
            fallback_response = self._generate_fallback_response(query, gemini_guidance)
            
            step = ProcessingStep(
                step_name=f"{step_name} (Fallback Mode)",
                input_data=query[:200] + "..." if len(query) > 200 else query,
                output_data=fallback_response,
                processing_time=time.time() - start_time,
                confidence=0.4,
                metadata={
                    'model_status': 'unavailable',
                    'fallback_used': True,
                    'guidance_provided': bool(gemini_guidance)
                }
            )
            
            return fallback_response, step
        
        try:
            with self.model_lock:
                if gemini_guidance:
                    context_info = f"Context from analysis: {gemini_guidance[:200]}..."
                    formatted_prompt = f"<|system|>\nYou are a knowledgeable AI assistant. Use the provided context to inform your response.\n{context_info}\n</s>\n<|user|>\n{query}\n</s>\n<|assistant|>\n"
                else:
                    formatted_prompt = f"<|system|>\nYou are a knowledgeable AI assistant providing detailed and engaging responses.\n</s>\n<|user|>\n{query}\n</s>\n<|assistant|>\n"
                
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=800
                )
                
                if torch.cuda.is_available() and hasattr(self.model, 'device') and self.model.device.type == 'cuda':
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
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
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1].strip()
                else:
                    response = response[len(formatted_prompt):].strip()
                
                response = self.clean_response(response)
                
                processing_time = time.time() - start_time
                
                step = ProcessingStep(
                    step_name=step_name,
                    input_data=query[:200] + "..." if len(query) > 200 else query,
                    output_data=response[:200] + "..." if len(response) > 200 else response,
                    processing_time=processing_time,
                    confidence=0.8,
                    metadata={
                        'model_version': self.model_version,
                        'guidance_used': bool(gemini_guidance),
                        'response_length': len(response),
                        'temperature': self.temperature,
                        'success': True
                    }
                )
                
                return response, step
                
        except Exception as e:
            logger.error(f"Error generating TinyLlama response: {e}")
            fallback_response = self._generate_fallback_response(query, gemini_guidance)
            
            step = ProcessingStep(
                step_name=f"{step_name} (Error Recovery)",
                input_data=query[:200] + "..." if len(query) > 200 else query,
                output_data=fallback_response,
                processing_time=time.time() - start_time,
                confidence=0.3,
                metadata={
                    'error': str(e)[:100],
                    'fallback_used': True,
                    'success': False
                }
            )
            
            return fallback_response, step
    
    def clean_response(self, response: str) -> str:
        """Clean and post-process response"""
        response = re.sub(r'<\|[^|]*\|>', '', response)
        response = response.replace('</s>', '').replace('<s>', '')
        
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line[:30] not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line[:30])
        
        response = '\n'.join(cleaned_lines).strip()
        
        if response and not response[-1] in '.!?':
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response if response else "I understand your question and am processing a thoughtful response."
    
    def _generate_fallback_response(self, query: str, guidance: str = "") -> str:
        """Generate fallback response when model is unavailable"""
        fallback_responses = {
            'hi': "Hello! I'm here to provide detailed and thoughtful responses to your questions.",
            'hello': "Greetings! I'm ready to engage in comprehensive discussions on any topic you'd like to explore.",
            'consciousness': "Consciousness represents a fascinating intersection of neuroscience, philosophy, and cognitive science, involving self-awareness and subjective experience.",
            'neural': "Neural networks form the computational foundation of modern AI systems, processing information through interconnected nodes.",
            'creative': "Creativity involves the novel combination of existing ideas and concepts to generate innovative solutions and artistic expressions.",
            'intelligence': "Intelligence encompasses reasoning, learning, problem-solving, and the ability to adapt to new situations and environments.",
            'learning': "Learning is the process of acquiring new knowledge, skills, and behaviors through experience, study, and instruction.",
            'future': "The future holds exciting possibilities for technological advancement and human development across multiple domains."
        }
        
        query_lower = query.lower()
        for keyword, response in fallback_responses.items():
            if keyword in query_lower:
                base_response = response
                break
        else:
            base_response = f"Your question about '{query}' touches on important concepts that deserve thoughtful consideration."
        
        if guidance and len(guidance) > 20:
            enhanced_response = f"{base_response} {guidance[:100]}..."
            return enhanced_response
        
        return base_response

class EmotionDetector:
    """Enhanced emotion detection with contextual awareness"""
    def __init__(self):
        self.emotion_keywords = {
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'livid'],
            'joy': ['happy', 'excited', 'pleased', 'delighted', 'cheerful', 'elated', 'thrilled', 'joyful', 'ecstatic'],
            'sadness': ['sad', 'depressed', 'melancholy', 'disappointed', 'sorrowful', 'grief', 'dejected'],
            'fear': ['afraid', 'scared', 'anxious', 'worried', 'nervous', 'terrified', 'panic', 'fearful'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'bewildered', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled', 'nauseated'],
            'anticipation': ['excited', 'eager', 'hopeful', 'optimistic', 'expecting', 'anticipating'],
            'curiosity': ['curious', 'wondering', 'interested', 'intrigued', 'questioning'],
            'confusion': ['confused', 'puzzled', 'perplexed', 'bewildered', 'unclear'],
            'satisfaction': ['satisfied', 'content', 'fulfilled', 'pleased', 'gratified']
        }
    
    def detect_emotion(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        emotions = {}
        total_intensity = 0
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            normalized_score = min(score / len(keywords), 1.0)
            emotions[emotion] = normalized_score
            total_intensity += normalized_score
        
        intensity = min(total_intensity / len(self.emotion_keywords), 1.0)
        emotions['intensity'] = intensity
        
        max_emotion = max(emotions.items(), key=lambda x: x[1] if x[0] != 'intensity' else 0)
        emotions['dominant_emotion'] = max_emotion[0] if max_emotion[1] > 0.1 else 'neutral'
        
        return emotions

class ContextualGrader:
    """Grade responses for contextual relevance and topic evolution health"""
    
    def __init__(self, db: MySQLContextDatabase):
        self.db = db
        self.topic_shift_threshold = 0.7
        
    def calculate_topic_signature(self, text: str) -> str:
        """Create a signature for topic identification"""
        if not NLTK_AVAILABLE:
            words = text.lower().split()
            key_words = [w for w in words if len(w) > 4][:3]
            return "_".join(sorted(key_words))
        
        try:
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
            filtered_words = [
                word for word in words 
                if word.isalpha() and len(word) > 3 and word not in stop_words
            ]
            
            word_freq = defaultdict(int)
            for word in filtered_words:
                word_freq[word] += 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            signature_words = [word[0] for word in top_words]
            
            return "_".join(sorted(signature_words))
            
        except:
            words = text.lower().split()
            key_words = [w for w in words if len(w) > 4][:3]
            return "_".join(sorted(key_words))
    
    def grade_contextual_relevance(self, current_query: str, session_id: str, turn_number: int) -> ContextualGrade:
        """Grade the contextual relevance of current query to conversation history"""
        
        history = self.db.get_conversation_history(session_id, limit=5) if self.db else []
        
        if not history:
            return ContextualGrade(
                relevance_score=1.0,
                topic_continuity=1.0,
                context_shift_health=1.0,
                recall_efficiency=1.0,
                uniqueness_score=1.0,
                overall_grade=1.0
            )
        
        current_signature = self.calculate_topic_signature(current_query)
        
        relevance_scores = []
        topic_continuity_score = 0.0
        
        for i, turn in enumerate(history):
            if turn['primary_topic']:
                prev_signature = turn['primary_topic']
                similarity = self.calculate_signature_similarity(current_signature, prev_signature)
                
                weight = 1.0 / (i + 1)
                relevance_scores.append(similarity * weight)
                
                if i == 0:
                    topic_continuity_score = similarity
        
        relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        context_shift_health = self.calculate_context_shift_health(history, current_signature)
        
        cached_pattern = self.db.get_response_pattern(current_signature) if self.db else {}
        recall_efficiency = 1.0 if cached_pattern else 0.5
        
        uniqueness_score = self.calculate_uniqueness_score(current_query, history)
        
        overall_grade = (
            relevance_score * 0.3 +
            topic_continuity_score * 0.25 +
            context_shift_health * 0.2 +
            recall_efficiency * 0.15 +
            uniqueness_score * 0.1
        )
        
        return ContextualGrade(
            relevance_score=relevance_score,
            topic_continuity=topic_continuity_score,
            context_shift_health=context_shift_health,
            recall_efficiency=recall_efficiency,
            uniqueness_score=uniqueness_score,
            overall_grade=overall_grade
        )
    
    def calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between topic signatures"""
        words1 = set(sig1.split('_'))
        words2 = set(sig2.split('_'))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_context_shift_health(self, history: List[Dict], current_signature: str) -> float:
        """Calculate how healthy the topic shifts are (gradual vs abrupt)"""
        if len(history) < 2:
            return 1.0
        
        shift_scores = []
        signatures = [current_signature] + [turn['primary_topic'] for turn in history if turn['primary_topic']]
        
        for i in range(len(signatures) - 1):
            similarity = self.calculate_signature_similarity(signatures[i], signatures[i + 1])
            shift_scores.append(similarity)
        
        if len(shift_scores) > 1:
            mean_shift = sum(shift_scores) / len(shift_scores)
            variance = sum((x - mean_shift) ** 2 for x in shift_scores) / len(shift_scores)
            health_score = max(0.0, 1.0 - variance)
        else:
            health_score = shift_scores[0] if shift_scores else 1.0
        
        return health_score
    
    def calculate_uniqueness_score(self, current_query: str, history: List[Dict]) -> float:
        """Calculate how unique the current query is compared to recent history"""
        if not history:
            return 1.0
        
        current_words = set(current_query.lower().split())
        
        similarity_scores = []
        for turn in history[:3]:
            if 'context_keywords' in turn and turn['context_keywords']:
                prev_words = set(word.lower() for word in turn['context_keywords'])
                
                if len(current_words | prev_words) > 0:
                    similarity = len(current_words & prev_words) / len(current_words | prev_words)
                    similarity_scores.append(similarity)
        
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            uniqueness = 1.0 - avg_similarity
        else:
            uniqueness = 1.0
        
        return max(0.0, min(1.0, uniqueness))

class GeminiAPI:
    """Enhanced Gemini API integration with detailed processing tracking"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-1.5-flash"
        
    async def generate_response_async(self, prompt: str, context: str = "", step_name: str = "Gemini Processing") -> Tuple[str, ProcessingStep]:
        """Generate response with detailed step tracking"""
        start_time = time.time()
        
        url = f"{self.base_url}/{self.model}:generateContent"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        full_prompt = f"{context}\n\nUser Query: {prompt}" if context else prompt
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}?key={self.api_key}",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        if 'candidates' in data and len(data['candidates']) > 0:
                            content = data['candidates'][0]['content']
                            if 'parts' in content and len(content['parts']) > 0:
                                response_text = content['parts'][0]['text']
                                
                                step = ProcessingStep(
                                    step_name=step_name,
                                    input_data=full_prompt[:200] + "..." if len(full_prompt) > 200 else full_prompt,
                                    output_data=response_text[:200] + "..." if len(response_text) > 200 else response_text,
                                    processing_time=processing_time,
                                    confidence=0.9,
                                    metadata={
                                        'api_status': response.status,
                                        'model_used': self.model,
                                        'tokens_requested': 1024,
                                        'response_length': len(response_text),
                                        'success': True
                                    }
                                )
                                
                                return response_text, step
                    
                    error_text = await response.text()
                    fallback_response = self._fallback_response(prompt)
                    
                    step = ProcessingStep(
                        step_name=f"{step_name} (Fallback)",
                        input_data=full_prompt[:200] + "..." if len(full_prompt) > 200 else full_prompt,
                        output_data=fallback_response,
                        processing_time=processing_time,
                        confidence=0.3,
                        metadata={
                            'api_status': response.status,
                            'error': error_text[:100],
                            'fallback_used': True,
                            'success': False
                        }
                    )
                    
                    return fallback_response, step
                    
        except Exception as e:
            processing_time = time.time() - start_time
            fallback_response = self._fallback_response(prompt)
            
            step = ProcessingStep(
                step_name=f"{step_name} (Error Fallback)",
                input_data=full_prompt[:200] + "..." if len(full_prompt) > 200 else full_prompt,
                output_data=fallback_response,
                processing_time=processing_time,
                confidence=0.2,
                metadata={
                    'error': str(e)[:100],
                    'fallback_used': True,
                    'success': False
                }
            )
            
            return fallback_response, step
    
    def generate_response_with_step(self, prompt: str, context: str = "", step_name: str = "Gemini Processing") -> Tuple[str, ProcessingStep]:
        """Synchronous wrapper with step tracking"""
        try:
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_async, prompt, context, step_name)
                    return future.result(timeout=60)
            except RuntimeError:
                return asyncio.run(self.generate_response_async(prompt, context, step_name))
        except Exception as e:
            logger.error(f"Failed to get Gemini response: {e}")
            fallback_response = self._fallback_response(prompt)
            step = ProcessingStep(
                step_name=f"{step_name} (System Error)",
                input_data=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                output_data=fallback_response,
                processing_time=0.1,
                confidence=0.1,
                metadata={'system_error': str(e)[:100], 'success': False}
            )
            return fallback_response, step
    
    def _run_async(self, prompt: str, context: str = "", step_name: str = "Gemini Processing") -> Tuple[str, ProcessingStep]:
        """Helper method to run async code in thread"""
        return asyncio.run(self.generate_response_async(prompt, context, step_name))
    
    def _fallback_response(self, prompt: str) -> str:
        """Enhanced fallback responses with more context"""
        fallback_responses = {
            'consciousness': "Consciousness represents one of the most profound mysteries in cognitive science, involving the subjective experience of awareness, the integration of sensory information across multiple modalities, and the emergence of self-reflective thought processes that enable beings to contemplate their own existence and mental states within broader philosophical and scientific frameworks.",
            'neural network': "Neural networks constitute sophisticated computational architectures fundamentally inspired by biological brain structures, utilizing interconnected processing nodes that transform information through weighted mathematical connections, learning complex patterns through iterative adjustment of these weights via optimization algorithms like backpropagation, and enabling machines to recognize intricate patterns and make intelligent decisions across diverse problem domains.",
            'creative': "Creativity emerges from the complex interplay of cognitive processes including divergent thinking, associative memory retrieval, pattern recognition, and the novel recombination of existing knowledge structures, often facilitated by states of reduced cognitive inhibition that allow for unconventional connections between disparate conceptual domains, leading to innovative solutions and artistic expressions.",
            'intelligence': "Intelligence encompasses the multifaceted capacity for reasoning, learning, adaptation, problem-solving, and understanding complex relationships, manifesting through various cognitive mechanisms including working memory, attention control, processing speed, and the ability to transfer knowledge across different domains and contexts.",
            'learning': "Learning represents the fundamental process through which organisms and artificial systems acquire, integrate, and apply new knowledge and skills, involving complex mechanisms of memory formation, pattern recognition, generalization, and the adaptive modification of behavior based on experience and feedback.",
            'future': "The future technological landscape promises unprecedented convergence across artificial intelligence, biotechnology, quantum computing, nanotechnology, and renewable energy systems, potentially reshaping human society through enhanced cognitive capabilities, extended healthy lifespans, sustainable resource management, and novel forms of human-machine collaborative intelligence.",
            'hi': "Hello! I understand you're greeting me, and I'm ready to engage in meaningful conversation about any topic you'd like to explore."
        }
        
        prompt_lower = prompt.lower()
        for keyword, response in fallback_responses.items():
            if keyword in prompt_lower:
                return response
        
        return f"The multifaceted topic of '{prompt}' represents a complex area of inquiry that intersects with numerous disciplines including cognitive science, technology, philosophy, and practical applications, requiring comprehensive analysis that considers both theoretical foundations and real-world implications for understanding, implementation, and future development within broader contextual frameworks."

# =============================================================================
# ENHANCED HOMEBREW MODEL - Uses Singleton Components
# =============================================================================

class EnhancedVerboseHomebrewModel:
    """Enhanced homebrew model with MySQL and Wikipedia integration"""
    
    def __init__(self, db: MySQLContextDatabase, wiki: WikipediaContextBuilder):
        self.db = db
        self.wiki = wiki
        self.grader = ContextualGrader(db) if db else None
        self.response_history = []
        
        self.sentence_builders = {
            'introduction': [
                "Drawing from comprehensive knowledge sources and contextual analysis, we can examine",
                "Integrating multiple perspectives and established research, this topic reveals",
                "Through systematic analysis incorporating both contemporary and foundational insights",
                "Leveraging extensive contextual understanding, we can explore the multifaceted nature of",
                "Building upon established knowledge frameworks and emerging research patterns"
            ],
            'wikipedia_integration': [
                "According to comprehensive encyclopedic sources, this concept encompasses",
                "Established documentation and research indicates that",
                "Authoritative sources demonstrate that this topic involves",
                "Scholarly consensus and documented evidence suggest",
                "Well-documented research and analysis reveal that"
            ],
            'database_recall': [
                "Previous contextual analysis and stored knowledge patterns indicate",
                "Building upon accumulated wisdom and documented relationships",
                "Leveraging established connections and proven analytical frameworks",
                "Drawing from recognized patterns and verified conceptual relationships",
                "Utilizing documented precedents and contextual linkages"
            ],
            'contextual_expansion': [
                "Furthermore, the interconnected nature of these concepts extends to",
                "Additionally, contextual analysis reveals deeper connections involving",
                "Expanding this analysis to encompass related domains demonstrates",
                "The broader implications encompass multiple interconnected aspects including",
                "Contextual exploration reveals significant relationships with"
            ],
            'synthesis_with_grading': [
                "Synthesizing these multiple analytical threads with contextual relevance assessment",
                "Integrating these perspectives while maintaining topical coherence and depth",
                "Bringing together these various analytical dimensions in a contextually graded framework",
                "Combining these insights with systematic evaluation of contextual continuity",
                "Merging these comprehensive viewpoints through structured contextual analysis"
            ]
        }
    
    def extract_keywords_for_database(self, text: str) -> List[str]:
        """Extract keywords for database storage and retrieval"""
        if not NLTK_AVAILABLE:
            words = text.split()
            return [word for word in words if len(word) > 4 and word.isalpha()][:10]
        
        try:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text.lower())
            
            filtered_words = [
                word for word in word_tokens 
                if word.isalpha() and len(word) > 3 and word not in stop_words
            ]
            
            word_freq = defaultdict(int)
            for word in filtered_words:
                word_freq[word] += 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word[0] for word in sorted_words[:10]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            words = text.split()
            return [word for word in words if len(word) > 4 and word.isalpha()][:10]
    
    def build_contextual_response_with_database(self, user_query: str, gemini_context: str, 
                                              tinyllama_context: str, session_id: str, 
                                              turn_number: int) -> Dict:
        """Generate comprehensive response using database and Wikipedia integration"""
        
        start_time = time.time()
        processing_steps = []
        database_interactions = []
        wikipedia_queries = []
        
        # Step 1: Extract keywords and get contextual grade
        step_start = time.time()
        keywords = self.extract_keywords_for_database(user_query)
        topic_signature = self.grader.calculate_topic_signature(user_query) if self.grader else user_query[:50]
        contextual_grade = self.grader.grade_contextual_relevance(user_query, session_id, turn_number) if self.grader else ContextualGrade()
        
        processing_steps.append(ProcessingStep(
            step_name="Keyword Extraction & Contextual Grading",
            input_data=user_query,
            output_data=f"Extracted {len(keywords)} keywords, Topic signature: {topic_signature}",
            processing_time=time.time() - step_start,
            confidence=0.9,
            metadata={
                'keywords': keywords,
                'topic_signature': topic_signature,
                'contextual_grade': asdict(contextual_grade)
            }
        ))
        
        # Step 2: Check for cached response patterns
        step_start = time.time()
        cached_pattern = self.db.get_response_pattern(topic_signature) if self.db else {}
        using_cached_pattern = bool(cached_pattern)
        
        if cached_pattern:
            database_interactions.append({
                'operation': 'pattern_recall',
                'table': 'response_patterns',
                'result': f"Found cached pattern with {cached_pattern['usage_frequency']} uses",
                'success': True
            })
        
        processing_steps.append(ProcessingStep(
            step_name="Pattern Recall Check",
            input_data=topic_signature,
            output_data=f"Cached pattern {'found' if using_cached_pattern else 'not found'}",
            processing_time=time.time() - step_start,
            confidence=1.0 if using_cached_pattern else 0.5,
            metadata={'cached_pattern': cached_pattern}
        ))
        
        # Step 3: Get database context for keywords
        step_start = time.time()
        word_contexts = {}
        if self.db:
            for keyword in keywords[:5]:
                context = self.db.get_word_context(keyword)
                if context:
                    word_contexts[keyword] = context
                    database_interactions.append({
                        'operation': 'word_context_retrieval',
                        'table': 'word',
                        'keyword': keyword,
                        'usage_count': context.get('usage_count', 0),
                        'success': True
                    })
        
        processing_steps.append(ProcessingStep(
            step_name="Database Context Retrieval",
            input_data=f"Keywords: {keywords[:5]}",
            output_data=f"Retrieved context for {len(word_contexts)} keywords",
            processing_time=time.time() - step_start,
            confidence=0.8,
            metadata={'word_contexts': list(word_contexts.keys())}
        ))
        
        # Step 4: Get Wikipedia context
        step_start = time.time()
        wikipedia_contexts = {}
        if self.wiki:
            primary_topics = keywords[:3]
            
            for topic in primary_topics:
                wiki_context = self.wiki.get_wikipedia_context(topic, max_sentences=2)
                if wiki_context.get('success'):
                    wikipedia_contexts[topic] = wiki_context
                    wikipedia_queries.append({
                        'topic': topic,
                        'title': wiki_context.get('title', ''),
                        'success': True,
                        'sentences_retrieved': len(wiki_context.get('summary_sentences', []))
                    })
                else:
                    wikipedia_queries.append({
                        'topic': topic,
                        'success': False,
                        'error': wiki_context.get('error', 'Unknown error')
                    })
        
        processing_steps.append(ProcessingStep(
            step_name="Wikipedia Context Enrichment",
            input_data=f"Topics: {keywords[:3]}",
            output_data=f"Enriched {len(wikipedia_contexts)} topics with Wikipedia content",
            processing_time=time.time() - step_start,
            confidence=0.7,
            metadata={'wikipedia_contexts': list(wikipedia_contexts.keys())}
        ))
        
        # Step 5: Build comprehensive response
        step_start = time.time()
        response_sections = []
        
        primary_topics = keywords[:3] if keywords else ['general topic']
        
        # Introduction
        if contextual_grade.overall_grade > 0.7:
            intro_type = 'contextual_expansion'
        elif using_cached_pattern:
            intro_type = 'database_recall'
        else:
            intro_type = 'introduction'
        
        intro_sentence = random.choice(self.sentence_builders[intro_type])
        intro_sentence += f" the multifaceted nature of {primary_topics[0]}"
        if len(primary_topics) > 1:
            intro_sentence += f" and its relationship to {', '.join(primary_topics[1:])}"
        intro_sentence += "."
        response_sections.append(intro_sentence)
        
        # Wikipedia-enhanced sections
        for topic, wiki_data in wikipedia_contexts.items():
            if wiki_data.get('summary_sentences'):
                wiki_intro = random.choice(self.sentence_builders['wikipedia_integration'])
                wiki_content = f"{wiki_intro} {wiki_data['summary_sentences'][0]}"
                
                if wiki_data.get('related_topics'):
                    related = wiki_data['related_topics'][:3]
                    wiki_content += f" This concept interconnects with established fields including {', '.join(related)}."
                
                response_sections.append(wiki_content)
        
        # Database-enhanced contextual expansion
        for keyword, context_data in word_contexts.items():
            if context_data.get('related_topic'):
                try:
                    related_topics = json.loads(context_data['related_topic']) if isinstance(context_data['related_topic'], str) else context_data['related_topic']
                    if related_topics:
                        contextual_intro = random.choice(self.sentence_builders['contextual_expansion'])
                        contextual_content = f"{contextual_intro} {', '.join(related_topics[:4])}"
                        
                        if context_data.get('definition'):
                            contextual_content += f", where {keyword} can be understood as {context_data['definition'][:100]}..."
                        
                        contextual_content += "."
                        response_sections.append(contextual_content)
                except:
                    pass
        
        # Gemini insights integration
        if gemini_context and len(gemini_context) > 20:
            gemini_insights = self.extract_key_insights(gemini_context, max_length=150)
            gemini_section = f"Advanced AI analysis suggests that {gemini_insights}"
            response_sections.append(gemini_section)
        
        # TinyLlama perspective integration
        if tinyllama_context and len(tinyllama_context) > 10:
            tinyllama_insights = self.extract_key_insights(tinyllama_context, max_length=100)
            tinyllama_section = f"Specialized language model processing reveals that {tinyllama_insights}"
            response_sections.append(tinyllama_section)
        
        # Synthesis
        synthesis_intro = random.choice(self.sentence_builders['synthesis_with_grading'])
        synthesis_content = f"{synthesis_intro}, we observe a contextual relevance score of {contextual_grade.overall_grade:.2f}, indicating {'strong' if contextual_grade.overall_grade > 0.7 else 'moderate' if contextual_grade.overall_grade > 0.5 else 'developing'} topical coherence."
        response_sections.append(synthesis_content)
        
        # Conclusion
        conclusion_options = [
            f"This analysis contributes to our evolving understanding of {primary_topics[0]}, with insights now integrated into our knowledge framework for enhanced future discourse.",
            f"The multidimensional exploration of {primary_topics[0]} demonstrates the value of combining established knowledge with dynamic contextual analysis.",
            f"Through systematic integration of multiple knowledge sources, our comprehension of {primary_topics[0]} continues to deepen and evolve."
        ]
        response_sections.append(random.choice(conclusion_options))
        
        full_response = " ".join(response_sections)
        
        processing_steps.append(ProcessingStep(
            step_name="Response Synthesis & Section Building",
            input_data=f"Sections: {len(response_sections)}",
            output_data=full_response[:200] + "..." if len(full_response) > 200 else full_response,
            processing_time=time.time() - step_start,
            confidence=0.9,
            metadata={
                'sections_count': len(response_sections),
                'total_length': len(full_response),
                'used_wikipedia': len(wikipedia_contexts) > 0,
                'used_database': len(word_contexts) > 0
            }
        ))
        
        # Step 6: Store learned patterns
        step_start = time.time()
        total_processing_time = time.time() - start_time
        
        if self.db:
            # Store conversation context
            self.db.store_conversation_context(
                session_id=session_id,
                turn_number=turn_number,
                primary_topic=topic_signature,
                secondary_topics=list(wikipedia_contexts.keys()),
                context_keywords=keywords,
                topic_shift_score=contextual_grade.context_shift_health,
                relevance_to_previous=contextual_grade.relevance_score
            )
            
            # Store response pattern
            self.db.store_response_pattern(
                topic_signature=topic_signature,
                response_template=full_response[:500],
                response_time=total_processing_time
            )
            
            # Store word relationships
            for keyword in keywords[:3]:
                if keyword in wikipedia_contexts:
                    wiki_data = wikipedia_contexts[keyword]
                    related_words = wiki_data.get('related_topics', [])[:5]
                    definition = wiki_data.get('summary_sentences', [''])[0][:200]
                    
                    self.db.store_word_relationship(
                        word=keyword,
                        related_words=related_words,
                        related_topics=primary_topics,
                        definition=definition
                    )
            
            # Store quality sentence patterns
            for i, section in enumerate(response_sections):
                if len(section) > 50:
                    topic_category = primary_topics[0] if primary_topics else 'general'
                    quality_score = contextual_grade.overall_grade * 0.8 + 0.2
                    
                    self.db.store_sentence_pattern(
                        sentence=section,
                        name=f"response_section_{i}",
                        topic_category=topic_category,
                        quality_score=quality_score
                    )
            
            database_interactions.append({
                'operation': 'learning_storage',
                'tables': ['conversation_context', 'response_patterns', 'word', 'sentances'],
                'items_stored': len(response_sections) + len(keywords) + 2,
                'success': True
            })
        
        processing_steps.append(ProcessingStep(
            step_name="Database Learning & Storage",
            input_data=f"Storing {len(keywords)} words, {len(response_sections)} patterns",
            output_data="Successfully stored learning patterns and context",
            processing_time=time.time() - step_start,
            confidence=1.0,
            metadata={'storage_success': True}
        ))
        
        # Calculate final metrics
        confidence = min(0.95, 0.7 + (contextual_grade.overall_grade * 0.2) + (len(wikipedia_contexts) * 0.05))
        creativity = min(0.9, 0.6 + (contextual_grade.uniqueness_score * 0.3))
        quality_score = (confidence + creativity + contextual_grade.overall_grade) / 3
        
        return {
            'response': full_response,
            'processing_steps': processing_steps,
            'database_interactions': database_interactions,
            'wikipedia_queries': wikipedia_queries,
            'contextual_grade': contextual_grade,
            'metrics': {
                'confidence': confidence,
                'creativity': creativity,
                'quality_score': quality_score,
                'response_length': len(full_response.split()),
                'keywords_processed': len(keywords),
                'wikipedia_enrichments': len(wikipedia_contexts),
                'database_contexts': len(word_contexts),
                'processing_time': total_processing_time,
                'sections_generated': len(response_sections),
                'pattern_recall_used': using_cached_pattern,
                'contextual_relevance': contextual_grade.overall_grade
            }
        }
    
    def extract_key_insights(self, text: str, max_length: int = 100) -> str:
        """Extract key insights from context text with improved selection"""
        if not text or len(text) < 10:
            return ""
        
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            sentences = text.split('. ')
        
        scored_sentences = []
        for sentence in sentences:
            if len(sentence) > 15:
                words = sentence.split()
                
                info_words = [w for w in words if len(w) > 4]
                density_score = len(info_words) / len(words) if words else 0
                
                key_terms = ['research', 'analysis', 'demonstrates', 'reveals', 'indicates', 'suggests', 'encompasses']
                key_term_bonus = sum(1 for term in key_terms if term in sentence.lower()) * 0.1
                
                total_score = density_score + key_term_bonus
                scored_sentences.append((total_score, sentence))
        
        scored_sentences.sort(reverse=True)
        selected_text = ""
        
        for score, sentence in scored_sentences:
            if len(selected_text + sentence) <= max_length:
                selected_text += sentence + ". "
            else:
                remaining_length = max_length - len(selected_text)
                if remaining_length > 20:
                    selected_text += sentence[:remaining_length-3] + "..."
                break
        
        return selected_text.strip() if selected_text else text[:max_length] + "..."

# =============================================================================
# MAIN ENHANCED COGNITIVE AI SYSTEM - Uses Singleton Manager
# =============================================================================

class EnhancedCognitiveConversationalAI:
    """Main orchestrator with singleton component management"""
    
    def __init__(self):
        # Use component manager for singleton behavior
        self.component_manager = ComponentManager()
        
        # Get singleton components
        self.db = self.component_manager.get_database()
        self.wiki = self.component_manager.get_wikipedia()
        self.tinyllama = self.component_manager.get_tinyllama()
        self.emotion_detector = self.component_manager.get_emotion_detector()
        
        # Initialize homebrew model with singleton components
        if self.db and self.wiki:
            self.homebrew_model = EnhancedVerboseHomebrewModel(self.db, self.wiki)
        else:
            self.homebrew_model = None
        
        # Client sessions and conversation state
        self.client_sessions: Dict[str, ClientSession] = {}
        self.conversation_history: List[ConversationTurn] = []
        self.anxiety_level = 0.0
        self.current_emotion_state = {'intensity': 0.0}
        
        logger.info("Enhanced Cognitive AI with MySQL and Wikipedia Integration Initialized")
    
    def register_client(self, session_id: str, gemini_api_key: str) -> bool:
        """Register a new client with their Gemini API key"""
        try:
            test_gemini = GeminiAPI(gemini_api_key)
            test_response, test_step = test_gemini.generate_response_with_step("Hello, this is a test.")
            
            if test_response and len(test_response) > 10 and test_step.metadata.get('success', False):
                self.client_sessions[session_id] = ClientSession(
                    session_id=session_id,
                    gemini_api_key=gemini_api_key,
                    created_at=time.time(),
                    last_active=time.time()
                )
                logger.info(f"Client {session_id} registered successfully")
                return True
            else:
                logger.error(f"Invalid API key for client {session_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to register client {session_id}: {e}")
            return False
    
    def process_enhanced_pipeline(self, user_query: str, session_id: str) -> Dict:
        """Process the complete enhanced AI pipeline"""
        if session_id not in self.client_sessions:
            return {"error": "Session not registered. Please provide a valid Gemini API key."}
        
        # Update session activity
        self.client_sessions[session_id].last_active = time.time()
        self.client_sessions[session_id].conversation_count += 1
        turn_number = self.client_sessions[session_id].conversation_count
        
        gemini_api = GeminiAPI(self.client_sessions[session_id].gemini_api_key)
        
        logger.info(f"Processing enhanced pipeline for session {session_id}: '{user_query}'")
        
        current_time = time.time()
        emotions = self.emotion_detector.detect_emotion(user_query)
        self.current_emotion_state = emotions
        
        processing_steps = []
        database_interactions = []
        wikipedia_queries = []
        
        try:
            # Step 1: User -> Gemini (Initial Analysis)
            logger.info("Step 1: User -> Gemini (Initial Analysis)")
            context_1 = "Provide a comprehensive analysis of this query, focusing on key concepts and relationships:"
            gemini_response_1, gemini_step_1 = gemini_api.generate_response_with_step(
                user_query, context_1, "Gemini Initial Analysis"
            )
            processing_steps.append(gemini_step_1)
            logger.info(f"Gemini Response 1: {gemini_response_1[:100]}...")
            
            # Step 2: Gemini -> TinyLlama (with Gemini guidance)
            logger.info("Step 2: Gemini -> TinyLlama (with Gemini guidance)")
            tinyllama_response_1, tinyllama_step_1 = self.tinyllama.generate_response_with_step(
                user_query, gemini_response_1, "TinyLlama Guided Processing"
            )
            processing_steps.append(tinyllama_step_1)
            logger.info(f"TinyLlama Response: {tinyllama_response_1}")
            
            # Step 3: TinyLlama -> Gemini (Quality Assessment & Enhancement)
            logger.info("Step 3: TinyLlama -> Gemini (Quality Assessment)")
            context_2 = f"Evaluate and enhance this response to '{user_query}', providing additional depth and context:"
            gemini_correction, gemini_step_2 = gemini_api.generate_response_with_step(
                tinyllama_response_1, context_2, "Gemini Quality Enhancement"
            )
            processing_steps.append(gemini_step_2)
            logger.info(f"Gemini Enhancement: {gemini_correction[:100]}...")
            
            # Step 4: Enhanced Pipeline -> Homebrew with Database & Wikipedia
            logger.info("Step 4: Enhanced Pipeline -> Homebrew with Database & Wikipedia")
            if self.homebrew_model and self.db:
                homebrew_result = self.homebrew_model.build_contextual_response_with_database(
                    user_query=user_query,
                    gemini_context=gemini_correction,
                    tinyllama_context=tinyllama_response_1,
                    session_id=session_id,
                    turn_number=turn_number
                )
                
                processing_steps.extend(homebrew_result['processing_steps'])
                database_interactions = homebrew_result['database_interactions']
                wikipedia_queries = homebrew_result['wikipedia_queries']
                contextual_grade = homebrew_result['contextual_grade']
                
                homebrew_response = homebrew_result['response']
                homebrew_metrics = homebrew_result['metrics']
                
                logger.info(f"Homebrew Response: {homebrew_response[:100]}...")
            else:
                homebrew_response = f"Based on the comprehensive analysis: {gemini_correction} Additionally, specialized processing suggests: {tinyllama_response_1}"
                homebrew_metrics = {'confidence': 0.6, 'quality_score': 0.6}
                contextual_grade = ContextualGrade(overall_grade=0.8)
                
                processing_steps.append(ProcessingStep(
                    step_name="Homebrew Fallback Processing",
                    input_data=f"Query: {user_query[:100]}...",
                    output_data=homebrew_response[:200] + "...",
                    processing_time=0.1,
                    confidence=0.6,
                    metadata={'database_available': False, 'fallback_mode': True}
                ))
            
            # Step 5: Final Synthesis -> Gemini (Ultimate Polish)
            logger.info("Step 5: Final Synthesis -> Gemini (Ultimate Polish)")
            context_3 = f"Provide the final, comprehensive response that synthesizes all insights for the query '{user_query}':"
            final_response, final_step = gemini_api.generate_response_with_step(
                homebrew_response, context_3, "Gemini Final Synthesis"
            )
            processing_steps.append(final_step)
            logger.info(f"Final Response: {final_response[:100]}...")
            
            # Create comprehensive verbose response
            total_processing_time = sum(step.processing_time for step in processing_steps)
            
            pipeline_summary = f"""
🔄 PROCESSING PIPELINE SUMMARY
════════════════════════════════════════════════════════════════
Total Processing Time: {total_processing_time:.3f}s
Steps Completed: {len(processing_steps)}
Database Interactions: {len(database_interactions)}
Wikipedia Queries: {len(wikipedia_queries)}
Contextual Grade: {contextual_grade.overall_grade:.2f}/1.0

🎯 QUALITY METRICS
═══════════════════
Relevance Score: {contextual_grade.relevance_score:.2f}
Topic Continuity: {contextual_grade.topic_continuity:.2f}
Context Shift Health: {contextual_grade.context_shift_health:.2f}
Recall Efficiency: {contextual_grade.recall_efficiency:.2f}
Uniqueness Score: {contextual_grade.uniqueness_score:.2f}
"""
            
            verbose_response = VerboseResponse(
                processing_steps=processing_steps,
                total_processing_time=total_processing_time,
                pipeline_summary=pipeline_summary,
                contextual_analysis={
                    'emotions_detected': emotions,
                    'contextual_grade': asdict(contextual_grade),
                    'session_turn': turn_number
                },
                database_interactions=database_interactions,
                wikipedia_queries=wikipedia_queries,
                contextual_grade=contextual_grade,
                gemini_response=final_response,
                tinyllama_response=tinyllama_response_1,
                homebrew_response=homebrew_response,
                final_response=final_response
            )
            
            # Save conversation turn
            turn = ConversationTurn(
                user_query=user_query,
                gemini_response=final_response,
                tinyllama_response=tinyllama_response_1,
                homebrew_response=homebrew_response,
                verbose_response=verbose_response,
                anxiety_level=self.anxiety_level,
                emotion_state=emotions,
                confidence_score=homebrew_metrics.get('confidence', 0.8),
                context_richness=contextual_grade.overall_grade,
                timestamp=current_time,
                session_id=session_id
            )
            
            self.conversation_history.append(turn)
            
            return {
                'user_query': user_query,
                'final_response': final_response,
                'verbose_response': verbose_response,
                'pipeline_responses': {
                    'initial_gemini': gemini_response_1,
                    'guided_tinyllama': tinyllama_response_1,
                    'enhanced_gemini': gemini_correction,
                    'database_homebrew': homebrew_response,
                    'final_synthesis': final_response
                },
                'processing_steps': [asdict(step) for step in processing_steps],
                'database_interactions': database_interactions,
                'wikipedia_queries': wikipedia_queries,
                'contextual_grade': asdict(contextual_grade),
                'metrics': homebrew_metrics,
                'emotions': emotions,
                'session_id': session_id,
                'conversation_count': turn_number,
                'total_processing_time': total_processing_time
            }
            
        except Exception as e:
            logger.error(f"Enhanced pipeline processing error: {e}")
            return {
                'error': f'Enhanced processing failed: {str(e)}',
                'user_query': user_query,
                'session_id': session_id,
                'processing_steps': [asdict(step) for step in processing_steps],
                'database_interactions': database_interactions,
                'wikipedia_queries': wikipedia_queries
            }

# =============================================================================
# ENHANCED COGNITIVE AI SYSTEM - Main Wrapper Class
# =============================================================================

class EnhancedCognitiveAISystem:
    """
    Enhanced Cognitive AI System - Main wrapper class that provides a unified interface
    for the CLI and main application. This class orchestrates all the components and
    provides high-level methods for system operations.
    """
    
    def __init__(self, db_config: Dict[str, str], tinyllama_model_path: str = "./tinyllama-forum-professional", debug_mode: bool = False):
        """
        Initialize the Enhanced Cognitive AI System
        
        Args:
            db_config: Dictionary with database configuration (host, user, password, database)
            tinyllama_model_path: Path to TinyLlama model directory
            debug_mode: Enable debug logging and verbose output
        """
        self.debug_mode = debug_mode
        self.db_config = db_config
        self.tinyllama_model_path = tinyllama_model_path
        
        # Configure logging based on debug mode
        if self.debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            logger.info("Debug mode enabled")
        
        # Initialize using singleton component manager
        self.component_manager = ComponentManager()
        self._initialize_components()
        
        # System statistics
        self.start_time = time.time()
        self.query_count = 0
        self.error_count = 0
        
        logger.info("Enhanced Cognitive AI System initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components using singleton manager"""
        try:
            # Get singleton components
            self.db = self.component_manager.get_database(
                host=self.db_config.get('host', 'localhost'),
                database=self.db_config.get('database', 'reservesphp'),
                user=self.db_config.get('user', 'root'),
                password=self.db_config.get('password', '')
            )
            
            self.wiki = self.component_manager.get_wikipedia()
            self.tinyllama = self.component_manager.get_tinyllama(self.tinyllama_model_path)
            self.emotion_detector = self.component_manager.get_emotion_detector()
            
            # Initialize core AI system using singleton components
            self.cognitive_ai = EnhancedCognitiveConversationalAI()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def register_client(self, session_id: str, gemini_api_key: str) -> bool:
        """Register a new client with their Gemini API key"""
        try:
            # Validate API key format first
            if not gemini_api_key or len(gemini_api_key) < 10:
                logger.error(f"Invalid API key format for client {session_id}: key too short")
                return False
            
            if not gemini_api_key.startswith('AIza'):
                logger.error(f"Invalid API key format for client {session_id}: should start with 'AIza'")
                return False
            
            # Test the API key with a simple request
            test_gemini = GeminiAPI(gemini_api_key)
            test_response, test_step = test_gemini.generate_response_with_step("Hello", "", "API Key Test")
            
            # Check if the response indicates success
            if test_response and len(test_response) > 10 and test_step.metadata.get('success', False):
                self.client_sessions[session_id] = ClientSession(
                    session_id=session_id,
                    gemini_api_key=gemini_api_key,
                    created_at=time.time(),
                    last_active=time.time()
                )
                logger.info(f"✅ Client {session_id} registered successfully")
                return True
            else:
                # More specific error logging
                error_info = test_step.metadata.get('error', 'Unknown API error')
                if 'API_KEY_INVALID' in str(error_info):
                    logger.error(f"❌ Invalid API key for client {session_id}: Key rejected by Google")
                elif 'PERMISSION_DENIED' in str(error_info):
                    logger.error(f"❌ Permission denied for client {session_id}: API key may not have Gemini access")
                elif 'QUOTA_EXCEEDED' in str(error_info):
                    logger.error(f"❌ Quota exceeded for client {session_id}: API key has exceeded usage limits")
                else:
                    logger.error(f"❌ API key test failed for client {session_id}: {error_info}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to register client {session_id}: {str(e)}")
            
            # Provide more helpful error messages
            if "API key not valid" in str(e):
                logger.error(f"   💡 Get a valid API key at: https://ai.google.dev/")
            elif "quota" in str(e).lower():
                logger.error(f"   💡 Check your API quota limits")
            elif "permission" in str(e).lower():
                logger.error(f"   💡 Ensure your API key has Gemini access enabled")
            
            return False
        
    def process_query(self, session_id: str, message: str) -> str:
        """Process a query through the enhanced AI pipeline"""
        if not session_id or not message:
            raise ValueError("Session ID and message are required")
        
        if session_id not in self.cognitive_ai.client_sessions:
            raise ValueError(f"Session '{session_id}' not registered. Please register with a valid Gemini API key first.")
        
        try:
            self.query_count += 1
            logger.info(f"Processing query #{self.query_count} for session {session_id}")
            
            result = self.cognitive_ai.process_enhanced_pipeline(message, session_id)
            
            if 'error' in result:
                self.error_count += 1
                raise ValueError(result['error'])
            
            response = result.get('final_response', 'No response generated')
            logger.info(f"✅ Query processed successfully: {len(response)} characters")
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Query processing error: {e}")
            raise
    
    def analyze_topic(self, session_id: str, topic: str):
        """Perform deep analysis of a topic with Wikipedia enrichment"""
        if session_id not in self.cognitive_ai.client_sessions:
            raise ValueError(f"Session '{session_id}' not registered")
        
        try:
            logger.info(f"Analyzing topic '{topic}' for session {session_id}")
            
            if self.wiki:
                concepts = self.wiki.extract_key_concepts(topic)
                logger.info(f"Key concepts extracted: {concepts}")
                
                for concept in concepts[:3]:
                    context = self.wiki.get_wikipedia_context(concept)
                    if context.get('success'):
                        logger.info(f"Wikipedia context for '{concept}': {context['title']}")
            
            analysis_query = f"Provide a comprehensive analysis of: {topic}"
            response = self.process_query(session_id, analysis_query)
            
            print(f"📊 TOPIC ANALYSIS COMPLETE")
            print(f"Topic: {topic}")
            print(f"Response: {response}")
            
        except Exception as e:
            logger.error(f"❌ Topic analysis error: {e}")
            raise
    
    def get_contextual_grade(self, session_id: str) -> Dict:
        """Get contextual grading information for a session"""
        try:
            if session_id not in self.cognitive_ai.client_sessions:
                return {"error": f"Session '{session_id}' not found"}
            
            history = []
            for turn in self.cognitive_ai.conversation_history:
                if turn.session_id == session_id:
                    history.append(turn)
            
            if not history:
                return {"error": "No conversation history found for session"}
            
            latest_turn = history[-1]
            grade_info = {
                "session_id": session_id,
                "conversation_count": len(history),
                "latest_grade": asdict(latest_turn.verbose_response.contextual_grade),
                "confidence_score": latest_turn.confidence_score,
                "context_richness": latest_turn.context_richness,
                "emotion_state": latest_turn.emotion_state
            }
            
            return grade_info
            
        except Exception as e:
            logger.error(f"❌ Error getting contextual grade: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status information"""
        try:
            uptime = time.time() - self.start_time
            
            status = {
                "system_status": "online",
                "uptime_seconds": uptime,
                "uptime_formatted": f"{uptime/3600:.1f} hours",
                "total_queries_processed": self.query_count,
                "total_errors": self.error_count,
                "success_rate": (self.query_count - self.error_count) / max(self.query_count, 1) * 100,
                "active_sessions": len(self.cognitive_ai.client_sessions),
                "total_conversations": len(self.cognitive_ai.conversation_history),
                "database_status": "connected" if self.db and self.db.connection else "disconnected",
                "wikipedia_status": "available" if self.wiki else "unavailable",
                "tinyllama_status": getattr(self.tinyllama, 'model_version', 'unknown'),
                "debug_mode": self.debug_mode
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {"error": str(e)}
    
    def get_database_stats(self) -> Dict:
        """Get database statistics and information"""
        if not self.db or not self.db.connection:
            return {"error": "Database not available"}
        
        try:
            stats = {}
            cursor = self.db.connection.cursor(dictionary=True)
            
            tables = ['word', 'sentances', 'conversation_context', 'response_patterns']
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    result = cursor.fetchone()
                    stats[f"{table}_count"] = result['count'] if result else 0
                except:
                    stats[f"{table}_count"] = "error"
            
            cursor.execute("SELECT DATABASE() as db_name")
            result = cursor.fetchone()
            stats["database_name"] = result['db_name'] if result else "unknown"
            
            cursor.execute("SELECT VERSION() as version")
            result = cursor.fetchone()
            stats["database_version"] = result['version'] if result else "unknown"
            
            try:
                cursor.execute("SELECT COUNT(*) as count FROM conversation_context WHERE timestamp > DATE_SUB(NOW(), INTERVAL 1 HOUR)")
                result = cursor.fetchone()
                stats["recent_conversations"] = result['count'] if result else 0
            except:
                stats["recent_conversations"] = "error"
            
            cursor.close()
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {"error": str(e)}
    
    def background_learning_cycle(self):
        """Trigger a background learning cycle to update models and patterns"""
        try:
            logger.info("🎓 Starting background learning cycle")
            
            if self.db and self.db.connection:
                cursor = self.db.connection.cursor(dictionary=True)
                
                cursor.execute("""
                    SELECT primary_topic, COUNT(*) as frequency 
                    FROM conversation_context 
                    WHERE timestamp > DATE_SUB(NOW(), INTERVAL 24 HOUR)
                    GROUP BY primary_topic 
                    ORDER BY frequency DESC 
                    LIMIT 10
                """)
                
                patterns = cursor.fetchall()
                logger.info(f"Found {len(patterns)} recent conversation patterns")
                
                cursor.execute("""
                    UPDATE response_patterns 
                    SET success_rate = success_rate * 0.95 + 0.05
                    WHERE last_used > DATE_SUB(NOW(), INTERVAL 7 DAY)
                """)
                
                affected_rows = cursor.rowcount
                logger.info(f"Updated {affected_rows} response patterns")
                
                cursor.close()
                self.db.connection.commit()
            
            logger.info("✅ Background learning cycle completed")
            
        except Exception as e:
            logger.error(f"❌ Background learning cycle error: {e}")
            self.error_count += 1
    
    def get_active_sessions(self) -> Dict:
        """Get information about active sessions"""
        try:
            sessions = {}
            for session_id, session in self.cognitive_ai.client_sessions.items():
                sessions[session_id] = {
                    "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.created_at)),
                    "last_active": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.last_active)),
                    "conversation_count": session.conversation_count,
                    "active_duration": time.time() - session.created_at
                }
            
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Error getting active sessions: {e}")
            return {"error": str(e)}
    
    def run_demo(self):
        """Run a demonstration of the enhanced AI system capabilities"""
        try:
            logger.info("🎬 Running enhanced demonstration")
            
            demo_session = f"demo_{int(time.time())}"
            demo_api_key = "demo_key_fallback"
            
            print(f"🎬 ENHANCED COGNITIVE AI DEMONSTRATION")
            print(f"Session: {demo_session}")
            print("=" * 60)
            
            demo_queries = [
                "What is artificial intelligence?",
                "How do neural networks learn?",
                "What are the applications of machine learning?",
                "Explain deep learning concepts",
                "What is the future of AI?"
            ]
            
            for i, query in enumerate(demo_queries, 1):
                print(f"\n🔄 Demo Query {i}: {query}")
                print("-" * 40)
                
                emotions = self.emotion_detector.detect_emotion(query)
                if emotions.get('intensity', 0) > 0.1:
                    print(f"🎭 Emotions: {emotions['dominant_emotion']} ({emotions['intensity']:.2f})")
                
                if self.tinyllama:
                    response, _ = self.tinyllama.generate_response_with_step(query)
                    print(f"🤖 TinyLlama: {response[:100]}...")
                
                if self.wiki:
                    concepts = self.wiki.extract_key_concepts(query)
                    if concepts:
                        print(f"📚 Key Concepts: {', '.join(concepts[:3])}")
                
                print(f"✅ Demo query {i} completed")
                time.sleep(1)
            
            print(f"\n🎉 Demonstration completed successfully!")
            print(f"📊 Processed {len(demo_queries)} demonstration queries")
            
        except Exception as e:
            logger.error(f"❌ Demo error: {e}")
            print(f"❌ Demo failed: {e}")
    
    def start_web_server(self, port: int = 8443, host: str = 'localhost'):
        """Start the web server interface"""
        try:
            logger.info(f"🌐 Starting web server on {host}:{port}")
            
            server = CognitiveAIServer(host=host, port=port, use_ssl=True)
            server.cognitive_ai = self.cognitive_ai
            server.start_server()
            
        except Exception as e:
            logger.error(f"❌ Web server error: {e}")
            raise
    
    def export_session_data(self, session_id: str):
        """Export session data to a file"""
        try:
            if session_id not in self.cognitive_ai.client_sessions:
                raise ValueError(f"Session '{session_id}' not found")
            
            session_data = {
                "session_id": session_id,
                "session_info": {
                    "created_at": self.cognitive_ai.client_sessions[session_id].created_at,
                    "conversation_count": self.cognitive_ai.client_sessions[session_id].conversation_count
                },
                "conversations": []
            }
            
            for turn in self.cognitive_ai.conversation_history:
                if turn.session_id == session_id:
                    session_data["conversations"].append({
                        "timestamp": turn.timestamp,
                        "user_query": turn.user_query,
                        "final_response": turn.gemini_response,
                        "confidence_score": turn.confidence_score,
                        "emotion_state": turn.emotion_state
                    })
            
            filename = f"session_export_{session_id}_{int(time.time())}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Session data exported to: {filename}")
            print(f"📊 Exported {len(session_data['conversations'])} conversations")
            
        except Exception as e:
            logger.error(f"❌ Export error: {e}")
            raise
    
    def import_training_data(self, file_path: str):
        """Import training data from a file"""
        try:
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            logger.info(f"📥 Importing training data from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'conversations' in data:
                conversations = data['conversations']
                processed_count = 0
                
                for conv in conversations:
                    if 'user_query' in conv and 'final_response' in conv:
                        if self.db and self.cognitive_ai.homebrew_model:
                            keywords = self.cognitive_ai.homebrew_model.extract_keywords_for_database(conv['user_query'])
                            
                            for keyword in keywords[:3]:
                                self.db.store_word_relationship(
                                    word=keyword,
                                    related_words=[],
                                    related_topics=keywords,
                                    definition=conv['final_response'][:200]
                                )
                        
                        processed_count += 1
                
                print(f"✅ Training data imported successfully")
                print(f"📊 Processed {processed_count} conversation examples")
                
            else:
                raise ValueError("Invalid training data format")
            
        except Exception as e:
            logger.error(f"❌ Import error: {e}")
            raise
    
    def close(self):
        """Gracefully shutdown the system and clean up resources"""
        try:
            logger.info("🛑 Shutting down Enhanced Cognitive AI System")
            
            final_stats = self.get_system_status()
            logger.info(f"Final stats: {final_stats['total_queries_processed']} queries, {final_stats['success_rate']:.1f}% success rate")
            
            if self.db and self.db.connection:
                try:
                    self.db.connection.close()
                    logger.info("✅ Database connection closed")
                except:
                    pass
            
            if hasattr(self, 'cognitive_ai'):
                self.cognitive_ai.client_sessions.clear()
                logger.info("✅ Session data cleared")
            
            logger.info("✅ System shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# =============================================================================
# CLI INTERFACE - Fixed
# =============================================================================

class CLIInterface:
    """Enhanced command line interface for the Cognitive AI system"""
    
    def __init__(self, ai_system: EnhancedCognitiveAISystem):
        self.ai_system = ai_system
        self.cognitive_ai = ai_system.cognitive_ai
        
        self.commands = {
            'register': self.handle_register_command,
            'chat': self.handle_chat_command,
            'analyze': self.handle_analyze_command,
            'grade': self.handle_grade_command,
            'status': self.handle_status_command,
            'database': self.handle_database_command,
            'train': self.handle_train_command,
            'sessions': self.handle_sessions_command,
            'demo': self.handle_demo_command,
            'server': self.handle_server_command,
            'export': self.handle_export_command,
            'import': self.handle_import_command,
            'quit': self.handle_quit_command,
        }
        print("\n================================================================================")
        print("🧠 ENHANCED COGNITIVE AI - COMMAND LINE INTERFACE")
        print("================================================================================")
        self.display_commands()
        print("================================================================================")

    def display_commands(self):
        print("Enhanced Commands:")
        commands_info = {
            'register': 'register <session_id> <api_key> - Register with Gemini API key',
            'chat': 'chat <session_id> <message> - Process message through full pipeline',
            'analyze': 'analyze <session_id> <topic> - Deep analysis with Wikipedia enrichment',
            'grade': 'grade <session_id> - Show contextual grading for session',
            'status': 'status - Show comprehensive system status',
            'database': 'database - Show database statistics',
            'train': 'train - Trigger learning cycle',
            'sessions': 'sessions - List active sessions with details',
            'demo': 'demo - Run enhanced demonstration',
            'server': 'server [port] [host] - Start web server',
            'export': 'export <session_id> - Export session data',
            'import': 'import <file> - Import training data',
            'quit': 'quit - Exit system'
        }
        
        for cmd in sorted(commands_info.keys()):
            print(f"  '{commands_info[cmd]}'")

    def run(self):
        """Starts the command-line interface loop."""
        while True:
            try:
                command_line = input("🧠 Enhanced AI> ").strip()
                if not command_line:
                    continue

                parts = command_line.split(maxsplit=1)
                command = parts[0].lower()
                args_str = parts[1] if len(parts) > 1 else ""
                args = args_str.split()

                handler = self.commands.get(command)
                if handler:
                    handler(args)
                else:
                    print(f"❌ Error: Unknown command '{command}'. See above for available commands.")
            except EOFError:
                print("\nExiting CLI.")
                break
            except KeyboardInterrupt:
                print("\nExiting CLI.")
                self.handle_quit_command([])
                break
            except Exception as e:
                print(f"❌ CLI error: {e}")
                if self.ai_system.debug_mode:
                    import traceback
                    traceback.print_exc()

    def handle_register_command(self, args):
        """Handles the 'register' command: register <session_id> <api_key>"""
        if len(args) < 2:
            print("❌ Error: Usage: register <session_id> <api_key>")
            print("💡 Get your free Gemini API key at: https://ai.google.dev/")
            return

        session_id = args[0]
        api_key = args[1]

        # Basic validation
        if not api_key.startswith('AIza'):
            print("❌ Error: Invalid API key format. Gemini API keys start with 'AIza'")
            print("💡 Get your free Gemini API key at: https://ai.google.dev/")
            return
        
        if len(api_key) < 35:
            print("❌ Error: API key appears too short. Please check your key.")
            return

        try:
            print(f"🔄 Testing API key for session '{session_id}'...")
            success = self.ai_system.register_client(session_id, api_key)
            if success:
                print(f"✅ Client '{session_id}' registered successfully!")
                print(f"💬 You can now use: chat {session_id} <your message>")
            else:
                print(f"❌ Failed to register client '{session_id}'.")
                print(f"🔧 Common issues:")
                print(f"   • API key is invalid or expired")
                print(f"   • API key doesn't have Gemini access enabled")
                print(f"   • Quota limits exceeded")
                print(f"   • Network connectivity issues")
                print(f"💡 Get a fresh API key at: https://ai.google.dev/")
        except Exception as e:
            print(f"❌ Error registering client '{session_id}': {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()


    def handle_chat_command(self, args):
        """Handles the 'chat' command: chat <session_id> <message>"""
        if len(args) < 2:
            print("❌ Error: Usage: chat <session_id> <message>")
            return

        session_id = args[0]
        message = ' '.join(args[1:])

        try:
            print(f"🔄 Processing chat for session '{session_id}' with message: '{message}'...")
            response = self.ai_system.process_query(session_id, message)
            print(f"🤖 AI Response: {response}")
        except ValueError as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred during chat: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_analyze_command(self, args):
        """Handles the 'analyze' command: analyze <session_id> <topic>"""
        if len(args) < 2:
            print("❌ Error: Usage: analyze <session_id> <topic>")
            return
        session_id = args[0]
        topic = ' '.join(args[1:])
        print(f"📈 Analyzing topic '{topic}' for session '{session_id}'...")
        try:
            self.ai_system.analyze_topic(session_id, topic)
            print(f"✅ Analysis complete for topic '{topic}'.")
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_grade_command(self, args):
        """Handles the 'grade' command: grade <session_id>"""
        if len(args) < 1:
            print("❌ Error: Usage: grade <session_id>")
            return
        session_id = args[0]
        print(f"🎯 Displaying contextual grade for session '{session_id}'...")
        try:
            grade_info = self.ai_system.get_contextual_grade(session_id)
            if "error" in grade_info:
                print(f"❌ {grade_info['error']}")
            else:
                print(f"Grade for session '{session_id}':")
                for key, value in grade_info.items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"❌ Error getting grade: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_status_command(self, args):
        """Handles the 'status' command: status"""
        print("📋 System Status:")
        try:
            status = self.ai_system.get_system_status()
            for key, value in status.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"❌ Error getting system status: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_database_command(self, args):
        """Handles the 'database' command: database"""
        print("🗄️ Database Statistics:")
        try:
            stats = self.ai_system.get_database_stats()
            if "error" in stats:
                print(f"❌ {stats['error']}")
            else:
                for key, value in stats.items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"❌ Error getting database statistics: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_train_command(self, args):
        """Handles the 'train' command: train"""
        print("🚀 Triggering learning cycle...")
        try:
            self.ai_system.background_learning_cycle()
            print("✅ Learning cycle completed.")
        except Exception as e:
            print(f"❌ Error triggering learning cycle: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_sessions_command(self, args):
        """Handles the 'sessions' command: sessions"""
        print("📋 Active Sessions:")
        try:
            sessions = self.ai_system.get_active_sessions()
            if "error" in sessions:
                print(f"❌ {sessions['error']}")
            elif not sessions:
                print("  No active sessions.")
            else:
                for session_id, details in sessions.items():
                    print(f"  - Session ID: {session_id}")
                    for key, value in details.items():
                        print(f"    {key}: {value}")
        except Exception as e:
            print(f"❌ Error listing sessions: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_demo_command(self, args):
        """Handles the 'demo' command: demo"""
        print("🚀 Running enhanced demonstration...")
        try:
            self.ai_system.run_demo()
            print("✅ Demonstration complete.")
        except Exception as e:
            print(f"❌ Error running demo: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_server_command(self, args):
        """Handles the 'server' command: server [port] [host]"""
        port = int(args[0]) if len(args) > 0 else 8443
        host = args[1] if len(args) > 1 else 'localhost'
        print(f"🌐 Starting Enhanced Cognitive AI Server on {host}:{port}...")
        try:
            self.ai_system.start_web_server(port, host)
            print("✅ Server started. Press Ctrl+C to stop.")
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_export_command(self, args):
        """Handles the 'export' command: export <session_id>"""
        if len(args) < 1:
            print("❌ Error: Usage: export <session_id>")
            return
        session_id = args[0]
        print(f"📤 Exporting session '{session_id}' data...")
        try:
            self.ai_system.export_session_data(session_id)
            print(f"✅ Session '{session_id}' data exported.")
        except Exception as e:
            print(f"❌ Error exporting session data: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_import_command(self, args):
        """Handles the 'import' command: import <file>"""
        if len(args) < 1:
            print("❌ Error: Usage: import <file>")
            return
        file_path = args[0]
        print(f"📥 Importing training data from '{file_path}'...")
        try:
            self.ai_system.import_training_data(file_path)
            print(f"✅ Training data imported from '{file_path}'.")
        except Exception as e:
            print(f"❌ Error importing training data: {e}")
            if self.ai_system.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_quit_command(self, args):
        """Handles the 'quit' command: quit"""
        print("Shutting down Enhanced Cognitive AI System...")
        self.ai_system.close()
        print("Goodbye!")
        sys.exit(0)

# =============================================================================
# HTTP SERVER - Enhanced with web interface
# =============================================================================

class EnhancedCognitiveAIHTTPSHandler(BaseHTTPRequestHandler):
    """Enhanced HTTPS server handler with verbose terminal display"""
    
    def __init__(self, cognitive_ai: EnhancedCognitiveConversationalAI, *args, **kwargs):
        self.cognitive_ai = cognitive_ai
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        if path == '/':
            self.serve_enhanced_main_page()
        elif path == '/register':
            self.serve_register_page()
        elif path == '/chat':
            self.serve_enhanced_chat_page()
        elif path == '/api/status':
            self.serve_status_api()
        else:
            self.send_404()
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        if path == '/api/register':
            self.handle_register_api(post_data)
        elif path == '/api/chat':
            self.handle_enhanced_chat_api(post_data)
        else:
            self.send_404()
    
    def serve_enhanced_main_page(self):
        """Serve enhanced main page"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Cognitive AI System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
        }
        .container {
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff41;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 0 30px rgba(0, 255, 65, 0.3);
            text-align: center;
            max-width: 800px;
            backdrop-filter: blur(10px);
        }
        h1 {
            color: #00ff41;
            margin-bottom: 20px;
            font-size: 2.5em;
            text-shadow: 0 0 20px #00ff41;
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 20px #00ff41; }
            to { text-shadow: 0 0 30px #00ff41, 0 0 40px #00ff41; }
        }
        .subtitle {
            color: #ccc;
            margin-bottom: 30px;
            font-size: 1.1em;
            line-height: 1.6;
        }
        .btn {
            display: inline-block;
            padding: 15px 30px;
            margin: 10px;
            background: linear-gradient(45deg, #00ff41, #00cc33);
            color: #000;
            text-decoration: none;
            border-radius: 25px;
            transition: all 0.3s ease;
            font-weight: bold;
            text-transform: uppercase;
        }
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0, 255, 65, 0.4);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Enhanced Cognitive AI System</h1>
        <p class="subtitle">
            Advanced multi-model AI pipeline with MySQL database integration, 
            Wikipedia knowledge enrichment, and real-time contextual learning.
        </p>
        
        <div style="margin-top: 30px;">
            <a href="/register" class="btn">🚀 Initialize System</a>
            <a href="/chat" class="btn">💬 Access Terminal</a>
        </div>
        
        <p style="margin-top: 20px; color: #888; font-size: 0.9em;">
            Requires Gemini API key • MySQL database • Wikipedia access
        </p>
    </div>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def serve_register_page(self):
        """Serve registration page"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Initialize - Enhanced Cognitive AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
        }
        .container {
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff41;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 0 30px rgba(0, 255, 65, 0.3);
            max-width: 600px;
            backdrop-filter: blur(10px);
        }
        h1 {
            color: #00ff41;
            margin-bottom: 20px;
            text-align: center;
            text-shadow: 0 0 20px #00ff41;
        }
        .form-group {
            margin: 20px 0;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #00ff41;
            font-weight: bold;
            font-size: 0.9em;
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 15px;
            background: #000;
            border: 2px solid #00ff41;
            border-radius: 10px;
            color: #00ff41;
            font-size: 16px;
            font-family: monospace;
            transition: all 0.3s ease;
        }
        input[type="text"]:focus, input[type="password"]:focus {
            outline: none;
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.5);
            border-color: #00ff41;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(45deg, #00ff41, #00cc33);
            color: #000;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            font-family: monospace;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 255, 65, 0.4);
        }
        .status {
            margin: 15px 0;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-family: monospace;
        }
        .success { 
            background: rgba(0, 255, 65, 0.2); 
            border: 1px solid #00ff41; 
            color: #00ff41;
        }
        .error { 
            background: rgba(255, 68, 68, 0.2); 
            border: 1px solid #ff4444; 
            color: #ff4444;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 System Initialization</h1>
        
        <form id="registerForm">
            <div class="form-group">
                <label for="sessionId">SESSION IDENTIFIER:</label>
                <input type="text" id="sessionId" name="sessionId" required 
                       placeholder="Enter unique session ID">
            </div>
            
            <div class="form-group">
                <label for="apiKey">GEMINI API KEY:</label>
                <input type="password" id="apiKey" name="apiKey" required 
                       placeholder="Enter your Gemini API key">
            </div>
            
            <button type="submit" class="btn">🚀 Initialize Enhanced System</button>
        </form>
        
        <div id="status"></div>
        
        <div style="text-align: center; margin-top: 20px;">
            <a href="/" style="color: #00ff41; text-decoration: none;">← Back to Main System</a>
        </div>
    </div>
    
    <script>
        document.getElementById('registerForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const sessionId = document.getElementById('sessionId').value;
            const apiKey = document.getElementById('apiKey').value;
            const statusDiv = document.getElementById('status');
            
            statusDiv.innerHTML = '<div class="status">🔄 Initializing enhanced system...</div>';
            
            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, api_key: apiKey })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    statusDiv.innerHTML = '<div class="status success">✅ System initialized successfully!</div>';
                    localStorage.setItem('sessionId', sessionId);
                    setTimeout(() => {
                        window.location.href = '/chat';
                    }, 1500);
                } else {
                    statusDiv.innerHTML = `<div class="status error">❌ ${result.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = '<div class="status error">❌ Network error occurred</div>';
            }
        });
        
        document.getElementById('sessionId').value = 'enhanced_' + Math.random().toString(36).substr(2, 9);
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def serve_enhanced_chat_page(self):
        """Serve enhanced chat interface"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Cognitive AI Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            height: 100vh;
            display: flex;
            flex-direction: column;
            color: #ffffff;
        }
        .header {
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 20px;
            border-bottom: 2px solid #00ff41;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            color: #00ff41;
            font-size: 1.5em;
            text-shadow: 0 0 10px #00ff41;
        }
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 20px;
        }
        .messages {
            flex: 1;
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid #00ff41;
            border-radius: 10px;
            padding: 20px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
        }
        .message {
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
            position: relative;
        }
        .user-message {
            background: linear-gradient(45deg, #0066cc, #004499);
            border-left: 4px solid #00aaff;
        }
        .ai-message {
            background: linear-gradient(45deg, #001122, #002244);
            border-left: 4px solid #00ff41;
        }
        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .message-title {
            font-weight: bold;
            color: #00ff41;
        }
        .message-time {
            font-size: 0.7em;
            color: #888;
        }
        .input-container {
            display: flex;
            gap: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #00ff41;
        }
        .input-container input {
            flex: 1;
            padding: 12px;
            background: #000;
            border: 1px solid #00ff41;
            border-radius: 5px;
            color: #00ff41;
            font-family: monospace;
            font-size: 14px;
        }
        .input-container input:focus {
            outline: none;
            box-shadow: 0 0 10px #00ff41;
        }
        .input-container button {
            padding: 12px 20px;
            background: linear-gradient(45deg, #00ff41, #00cc33);
            color: #000;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-family: monospace;
            transition: all 0.3s ease;
        }
        .input-container button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 65, 0.4);
        }
        .status-message {
            text-align: center;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status-error { background: rgba(255, 68, 68, 0.2); border: 1px solid #ff4444; }
        .status-success { background: rgba(0, 255, 65, 0.2); border: 1px solid #00ff41; }
        .status-loading { background: rgba(255, 170, 0, 0.2); border: 1px solid #ffaa00; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Enhanced Cognitive AI Terminal</h1>
    </div>
    
    <div class="chat-container">
        <div class="messages" id="messages">
            <div class="ai-message message">
                <div class="message-header">
                    <span class="message-title">🤖 Enhanced Cognitive AI System</span>
                    <span class="message-time">System Ready</span>
                </div>
                <div>
                    Welcome to the Enhanced Cognitive AI Terminal! This system features real-time MySQL database integration, Wikipedia knowledge enrichment, and multi-model AI pipeline processing.
                </div>
            </div>
        </div>
        
        <div id="status"></div>
        
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Enter your query for enhanced AI processing..." 
                   onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">PROCESS 🚀</button>
        </div>
    </div>
    
    <script>
        let sessionId = localStorage.getItem('sessionId');
        
        if (!sessionId) {
            document.getElementById('status').innerHTML = 
                '<div class="status-message status-error">❌ No session found. Please <a href="/register">register</a> first.</div>';
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message || !sessionId) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = '<div class="status-message status-loading">🔄 Processing through enhanced AI pipeline...</div>';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: message, 
                        session_id: sessionId 
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    statusDiv.innerHTML = '';
                    addMessage(result.data.final_response, 'ai');
                } else {
                    statusDiv.innerHTML = `<div class="status-message status-error">❌ ${result.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = '<div class="status-message status-error">❌ Network error occurred</div>';
            }
        }
        
        function addMessage(text, type) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            const currentTime = new Date().toLocaleTimeString();
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <span class="message-title">${type === 'user' ? '👤 User Input' : '🤖 AI Response'}</span>
                    <span class="message-time">${currentTime}</span>
                </div>
                <div>${text}</div>
            `;
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def handle_register_api(self, post_data):
        """Handle registration API requests with better error messages"""
        try:
            data = json.loads(post_data)
            session_id = data.get('session_id')
            api_key = data.get('api_key')
            
            if not session_id or not api_key:
                self.send_json_response({
                    'success': False, 
                    'error': 'Missing session ID or API key',
                    'help': 'Get your free Gemini API key at https://ai.google.dev/'
                })
                return
            
            # Basic validation
            if not api_key.startswith('AIza'):
                self.send_json_response({
                    'success': False,
                    'error': 'Invalid API key format. Gemini API keys start with "AIza"',
                    'help': 'Get your free Gemini API key at https://ai.google.dev/'
                })
                return
            
            if len(api_key) < 35:
                self.send_json_response({
                    'success': False,
                    'error': 'API key appears too short. Please check your key.',
                    'help': 'Copy the complete API key from https://ai.google.dev/'
                })
                return
            
            success = self.cognitive_ai.register_client(session_id, api_key)
            
            if success:
                self.send_json_response({
                    'success': True, 
                    'message': 'Enhanced system initialized successfully',
                    'session_id': session_id
                })
            else:
                self.send_json_response({
                    'success': False, 
                    'error': 'Invalid API key or Google AI service unavailable',
                    'help': 'Verify your API key at https://ai.google.dev/ and ensure it has Gemini access'
                })
                    
        except Exception as e:
            logger.error(f"Registration error: {e}")
            self.send_json_response({
                'success': False, 
                'error': 'Server error during initialization',
                'help': 'Please try again or contact support'
            })
    def handle_enhanced_chat_api(self, post_data):
        """Handle enhanced chat API requests"""
        try:
            data = json.loads(post_data)
            message = data.get('message')
            session_id = data.get('session_id')
            
            if not message or not session_id:
                self.send_json_response({'success': False, 'error': 'Missing message or session_id'})
                return
            
            result = self.cognitive_ai.process_enhanced_pipeline(message, session_id)
            
            if 'error' in result:
                self.send_json_response({'success': False, 'error': result['error']})
            else:
                self.send_json_response({'success': True, 'data': result})
                
        except Exception as e:
            logger.error(f"Enhanced chat error: {e}")
            self.send_json_response({'success': False, 'error': 'Server error during enhanced processing'})
    
    def serve_status_api(self):
        """Serve system status API"""
        try:
            status = {
                'active_sessions': len(self.cognitive_ai.client_sessions),
                'total_conversations': len(self.cognitive_ai.conversation_history),
                'system_status': 'online',
                'database_status': 'connected' if self.cognitive_ai.db and self.cognitive_ai.db.connection else 'disconnected',
                'tinyllama_status': getattr(self.cognitive_ai.tinyllama, 'model_version', 'unknown'),
                'wikipedia_status': 'available',
                'uptime': time.time()
            }
            self.send_json_response(status)
        except Exception as e:
            logger.error(f"Status API error: {e}")
            self.send_json_response({'error': 'Failed to get system status'})
    
    def send_json_response(self, data):
        """Send JSON response with proper headers"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
    
    def send_404(self):
        """Send 404 response"""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        error_page = """
<!DOCTYPE html>
<html>
<head><title>404 - Not Found</title></head>
<body style="font-family: monospace; background: #1a1a2e; color: #00ff41; text-align: center; padding: 50px;">
    <h1>🚫 404 - Page Not Found</h1>
    <p>The requested resource was not found on this server.</p>
    <a href="/" style="color: #00ff41;">← Return to Main System</a>
</body>
</html>"""
        self.wfile.write(error_page.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to reduce logging noise in production"""
        pass

class CognitiveAIServer:
    """Main server class for the Enhanced Cognitive AI system"""
    
    def __init__(self, host: str = 'localhost', port: int = 8443, use_ssl: bool = True):
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.cognitive_ai = EnhancedCognitiveConversationalAI()
        self.server = None
        
    def create_handler(self):
        """Create request handler with cognitive AI instance"""
        cognitive_ai = self.cognitive_ai
        
        class CognitiveAIHandler(EnhancedCognitiveAIHTTPSHandler):
            def __init__(self, *args, **kwargs):
                self.cognitive_ai = cognitive_ai
                super(BaseHTTPRequestHandler, self).__init__(*args, **kwargs)
        
        return CognitiveAIHandler
    
    def start_server(self):
        """Start the enhanced HTTPS server"""
        try:
            handler_class = self.create_handler()
            self.server = HTTPServer((self.host, self.port), handler_class)
            
            if self.use_ssl:
                try:
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    
                    cert_files = [
                        ('server.crt', 'server.key'),
                        (r'C:\portablexampp\apache\conf\ssl.crt\certificate.crt', 
                         r'C:\portablexampp\apache\conf\ssl.key\private.key')
                    ]
                    
                    ssl_configured = False
                    for cert_file, key_file in cert_files:
                        if os.path.exists(cert_file) and os.path.exists(key_file):
                            context.load_cert_chain(cert_file, key_file)
                            self.server.socket = context.wrap_socket(
                                self.server.socket, server_side=True
                            )
                            ssl_configured = True
                            protocol = "HTTPS"
                            break
                    
                    if not ssl_configured:
                        logger.warning("SSL certificates not found, falling back to HTTP")
                        self.use_ssl = False
                        protocol = "HTTP"
                        
                except Exception as e:
                    logger.warning(f"SSL setup failed: {e}, using HTTP")
                    self.use_ssl = False
                    protocol = "HTTP"
            else:
                protocol = "HTTP"
            
            logger.info("="*80)
            logger.info("🧠 ENHANCED COGNITIVE AI SYSTEM STARTING")
            logger.info("="*80)
            logger.info(f"🌐 Server: {protocol}://{self.host}:{self.port}")
            logger.info(f"🗄️  Database: {'Connected' if self.cognitive_ai.db else 'Disconnected'}")
            logger.info(f"🤖 TinyLlama: {getattr(self.cognitive_ai.tinyllama, 'model_version', 'Unknown')}")
            logger.info(f"📚 Wikipedia: Available")
            logger.info(f"🎯 Contextual Grading: Active")
            logger.info(f"📟 Verbose Terminal: Enabled")
            logger.info("="*80)
            
            print(f"\n🚀 Enhanced Cognitive AI System Ready!")
            print(f"📱 Access the system at: {protocol.lower()}://{self.host}:{self.port}")
            print(f"🔐 Registration page: {protocol.lower()}://{self.host}:{self.port}/register")
            print(f"💬 Chat interface: {protocol.lower()}://{self.host}:{self.port}/chat")
            print(f"📊 System status: {protocol.lower()}://{self.host}:{self.port}/api/status")
            print(f"\n🔑 Requirements:")
            print(f"   • Gemini API key (get free at: https://ai.google.dev/)")
            print(f"   • MySQL database (reservesphp)")
            print(f"   • TinyLlama model (optional, will use fallback)")
            print(f"\n🎯 Features:")
            print(f"   ✅ Multi-model AI pipeline (Gemini → TinyLlama → Homebrew)")
            print(f"   ✅ Real-time MySQL database learning")
            print(f"   ✅ Wikipedia knowledge enrichment")
            print(f"   ✅ Contextual grading and topic evolution")
            print(f"   ✅ Verbose terminal processing readout")
            print(f"   ✅ Emotional intelligence analysis")
            print(f"\n⚡ Performance:")
            print(f"   • Database Tables: 4 (word, sentences, conversation_context, response_patterns)")
            print(f"   • Processing Pipeline: 8+ steps per query")
            print(f"   • Response Caching: 30-minute TTL")
            print(f"   • Background Learning: Every 5 minutes")
            
            self.start_background_services()
            
            print(f"\n🔄 Server running... Press Ctrl+C to stop")
            
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutdown requested by user")
            self.stop_server()
        except Exception as e:
            logger.error(f"🚨 Server error: {e}")
            raise
    
    def start_background_services(self):
        """Start background services for the cognitive AI system"""
        def training_scheduler():
            """Background training and maintenance scheduler"""
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    
                    if len(self.cognitive_ai.conversation_history) >= 5:
                        logger.info("🎓 Running background learning cycle...")
                        
                        if self.cognitive_ai.db:
                            try:
                                result = self.cognitive_ai.db.connection.ping(reconnect=True)
                                logger.info(f"✅ Background learning completed")
                            except Exception as e:
                                logger.error(f"❌ Background learning failed: {e}")
                        
                        if hasattr(self.cognitive_ai, 'wiki'):
                            self.cognitive_ai.wiki.clean_cache()
                            logger.info("🧹 Cache cleanup completed")
                        
                        active_sessions = len(self.cognitive_ai.client_sessions)
                        total_conversations = len(self.cognitive_ai.conversation_history)
                        logger.info(f"📊 System status: {active_sessions} active sessions, {total_conversations} total conversations")
                    
                except Exception as e:
                    logger.error(f"🚨 Background service error: {e}")
        
        def session_cleanup():
            """Clean up inactive sessions periodically"""
            while True:
                try:
                    time.sleep(1800)  # Run every 30 minutes
                    current_time = time.time()
                    inactive_sessions = []
                    
                    for session_id, session in self.cognitive_ai.client_sessions.items():
                        if current_time - session.last_active > 7200:
                            inactive_sessions.append(session_id)
                    
                    for session_id in inactive_sessions:
                        del self.cognitive_ai.client_sessions[session_id]
                        logger.info(f"🧹 Cleaned up inactive session: {session_id}")
                    
                    if inactive_sessions:
                        logger.info(f"🗑️  Removed {len(inactive_sessions)} inactive sessions")
                    
                except Exception as e:
                    logger.error(f"🚨 Session cleanup error: {e}")
        
        training_thread = threading.Thread(target=training_scheduler, daemon=True)
        training_thread.start()
        logger.info("🔄 Background learning scheduler started")
        
        cleanup_thread = threading.Thread(target=session_cleanup, daemon=True)
        cleanup_thread.start()
        logger.info("🧹 Session cleanup scheduler started")
    
    def stop_server(self):
        """Stop the server gracefully"""
        if self.server:
            logger.info("🛑 Shutting down Enhanced Cognitive AI Server...")
            
            if self.cognitive_ai.db and self.cognitive_ai.db.connection:
                self.cognitive_ai.db.connection.close()
                logger.info("🗄️  Database connection closed")
            
            self.server.shutdown()
            self.server.server_close()
            logger.info("✅ Server stopped successfully")
            
            total_sessions = len(self.cognitive_ai.client_sessions)
            total_conversations = len(self.cognitive_ai.conversation_history)
            
            print(f"\n📊 SESSION SUMMARY")
            print(f"════════════════════════════════════════")
            print(f"💬 Total Sessions: {total_sessions}")
            print(f"🗣️  Total Conversations: {total_conversations}")
            print(f"🎯 Average Conversations per Session: {total_conversations/max(total_sessions, 1):.1f}")
            print(f"════════════════════════════════════════")
            print(f"🙏 Thank you for using Enhanced Cognitive AI!")

# =============================================================================
# MAIN FUNCTION - Fixed and Enhanced
# =============================================================================

def main():
    """Enhanced main entry point with comprehensive configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Cognitive AI System with MySQL and Wikipedia Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start CLI mode
  python learningbot.py --mode cli
  
  # Start web server with default settings
  python learningbot.py --mode server
  
  # Start web server on specific port without SSL
  python learningbot.py --mode server --port 8080 --no-ssl
  
  # Start with custom database settings
  python learningbot.py --mode server --db-host 192.168.1.100 --db-user myuser
  
  # Enable debug mode
  python learningbot.py --mode cli --debug

System Requirements:
  - Python 3.8+
  - MySQL database 'reservesphp'
  - Gemini API key
  - Optional: TinyLlama model in ./tinyllama-forum-professional/
"""
    )
    
    parser.add_argument('--mode', choices=['cli', 'server'], default='cli',
                       help='Run mode: cli for command line, server for web interface')
    
    parser.add_argument('--host', default='localhost', 
                       help='Server host address (default: localhost)')
    parser.add_argument('--port', type=int, default=8443, 
                       help='Server port (default: 8443)')
    parser.add_argument('--no-ssl', action='store_true', 
                       help='Disable SSL/HTTPS (use HTTP instead)')
    
    parser.add_argument('--db-host', default='localhost',
                       help='MySQL database host (default: localhost)')
    parser.add_argument('--db-name', default='reservesphp',
                       help='MySQL database name (default: reservesphp)')
    parser.add_argument('--db-user', default='root',
                       help='MySQL database user (default: root)')
    parser.add_argument('--db-password', default='',
                       help='MySQL database password (default: empty)')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--no-wikipedia', action='store_true',
                       help='Disable Wikipedia integration')
    parser.add_argument('--cache-ttl', type=int, default=1800,
                       help='Cache TTL in seconds (default: 1800)')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    print("🚀 ENHANCED COGNITIVE AI SYSTEM")
    print("=" * 80)
    print("🧠 Multi-Model AI Pipeline with Database Learning")
    print("🗄️  MySQL Integration • 📚 Wikipedia Enrichment • 🎯 Contextual Grading")
    print("=" * 80)
    
    print("📋 System Configuration:")
    print(f"   🖥️  Mode: {args.mode.upper()}")
    print(f"   🗄️  Database: {args.db_user}@{args.db_host}/{args.db_name}")
    print(f"   📚 Wikipedia: {'Disabled' if args.no_wikipedia else 'Enabled'}")
    print(f"   🔧 Debug: {'On' if args.debug else 'Off'}")
    
    if args.mode == 'server':
        protocol = 'HTTP' if args.no_ssl else 'HTTPS'
        print(f"   🌐 Server: {protocol}://{args.host}:{args.port}")
    
    print("=" * 80)
    
    try:
        print("🔧 Initializing Enhanced Cognitive AI System...")
        
        # Create database config
        db_config = {
            'host': args.db_host,
            'database': args.db_name,
            'user': args.db_user,
            'password': args.db_password
        }
        
        # Initialize the enhanced cognitive AI system
        ai_system = EnhancedCognitiveAISystem(
            db_config=db_config,
            tinyllama_model_path="./tinyllama-forum-professional",
            debug_mode=args.debug
        )
        
        print("✅ Enhanced Cognitive AI System initialized successfully!")
        
        print("\n🎯 FEATURE STATUS:")
        print(f"   🤖 TinyLlama Model: {getattr(ai_system.tinyllama, 'model_version', 'Fallback')}")
        print(f"   🗄️  Database: {'Connected' if ai_system.db else 'Unavailable'}")
        print(f"   📚 Wikipedia: {'Available' if ai_system.wiki and not args.no_wikipedia else 'Disabled'}")
        print(f"   🧠 Homebrew AI: {'Active' if ai_system.cognitive_ai.homebrew_model else 'Fallback'}")
        print(f"   🎭 Emotion Detection: Active")
        print(f"   🎯 Contextual Grading: Active")
        print(f"   📟 Verbose Terminal: Active")
        
        if args.mode == 'server':
            print(f"\n🌐 Starting Enhanced Web Server...")
            server = CognitiveAIServer(
                host=args.host,
                port=args.port,
                use_ssl=not args.no_ssl
            )
            server.cognitive_ai = ai_system.cognitive_ai
            server.start_server()
        else:
            print(f"\n💻 Starting Enhanced Command Line Interface...")
            cli = CLIInterface(ai_system)
            cli.run()
            
    except KeyboardInterrupt:
        print(f"\n🛑 System shutdown requested by user")
    except Exception as e:
        logger.error(f"🚨 System initialization error: {e}")
        print(f"❌ Failed to start Enhanced Cognitive AI System: {e}")
        print(f"💡 Try running with --debug for more information")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())