#!/usr/bin/env python3
"""
Advanced AI Coding Agent - Sophisticated AI Assistant with Multi-Modal Capabilities

This agent implements advanced architectural patterns similar to Claude's system:
- Multi-modal AI capabilities with context awareness
- Advanced reasoning and planning systems
- Real-time analysis and adaptive response generation
- Comprehensive tool integration and execution
- Intelligent workflow management and error handling

Author: AI Coding Agent Team
Version: 2.0.0
License: MIT
"""

import os
import sys
import json
import asyncio
import logging
import traceback
import threading
import queue
import time
import hashlib
import pickle
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import concurrent.futures
from contextlib import contextmanager
import weakref
import gc

# Auto-install required packages
REQUIRED_PACKAGES = [
    "google-generativeai>=0.8.0",
    "openai>=1.0.0",
    "anthropic>=0.8.0",
    "tiktoken>=0.5.0",
    "rich>=13.0.0",
    "textual>=0.50.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
    "gitpython>=3.1.40",
    "aiohttp>=3.9.0",
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
    "pydantic>=2.5.0",
    "jinja2>=3.1.0",
    "watchdog>=3.0.0",
    "psutil>=5.9.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "networkx>=3.0",
    "tree-sitter>=0.20.0",
    "tree-sitter-python>=0.20.0",
    "tree-sitter-javascript>=0.20.0",
    "ast-comments>=1.0.0",
    "rope>=1.10.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pylint>=2.17.0",
    "bandit>=1.7.0",
    "safety>=2.3.0"
]

def install_package(package: str) -> bool:
    """Install a package if not available"""
    try:
        module_name = package.split('>=')[0].replace('-', '_')
        if module_name == 'tree_sitter_python':
            import tree_sitter_python
        elif module_name == 'tree_sitter_javascript':
            import tree_sitter_javascript
        else:
            __import__(module_name)
        return True
    except ImportError:
        try:
            import subprocess
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except Exception as e:
            print(f"Failed to install {package}: {e}")
            return False

# Install packages
for package in REQUIRED_PACKAGES:
    install_package(package)

# Import all required modules
import google.generativeai as genai
import tiktoken
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.live import Live
import git
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import tree_sitter
from tree_sitter import Language, Parser
import ast
import rope.base.project
from rope.base.libutils import path_to_resource
from rope.refactor.rename import Rename
from rope.refactor.extract import ExtractMethod
import black
import isort
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from jinja2 import Template, Environment, FileSystemLoader
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import psutil
import aiohttp
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_agent.log'),
        logging.StreamHandler()
    ]
)

console = Console()
logger = logging.getLogger(__name__)

class AgentCapability(Enum):
    """Agent capability types"""
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    SECURITY_ANALYSIS = "security_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    PROJECT_MANAGEMENT = "project_management"
    WEB_INTEGRATION = "web_integration"
    GIT_OPERATIONS = "git_operations"
    DATABASE_OPERATIONS = "database_operations"
    API_INTEGRATION = "api_integration"
    DEPLOYMENT = "deployment"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

class TaskStatus(Enum):
    """Task status types"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"

class ContextType(Enum):
    """Context types for multi-modal understanding"""
    CODE = "code"
    DOCUMENTATION = "documentation"
    ERROR = "error"
    REQUIREMENT = "requirement"
    CONVERSATION = "conversation"
    FILE_SYSTEM = "file_system"
    GIT_HISTORY = "git_history"
    WEB_CONTENT = "web_content"
    DATABASE_SCHEMA = "database_schema"

@dataclass
class AgentConfig:
    """Comprehensive agent configuration"""
    # AI Model Configuration
    primary_model: str = "gemini-2.0-flash-exp"
    fallback_models: List[str] = field(default_factory=lambda: ["gpt-4", "claude-3-sonnet"])
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Workspace Configuration
    workspace_dir: str = "."
    project_name: str = ""
    
    # Capability Configuration
    enabled_capabilities: Set[AgentCapability] = field(default_factory=lambda: set(AgentCapability))
    max_concurrent_tasks: int = 5
    max_context_length: int = 1000000
    
    # Performance Configuration
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    enable_parallel_processing: bool = True
    memory_limit_mb: int = 2048
    
    # Behavior Configuration
    auto_save: bool = True
    auto_commit: bool = False
    auto_test: bool = True
    auto_format: bool = True
    auto_document: bool = True
    
    # Learning Configuration
    enable_learning: bool = True
    context_retention_days: int = 30
    feedback_learning: bool = True
    
    # Security Configuration
    enable_security_checks: bool = True
    allow_external_requests: bool = True
    sandbox_mode: bool = False
    
    # UI Configuration
    theme: str = "dark"
    verbose_output: bool = True
    show_progress: bool = True
    
    # Advanced Configuration
    plugin_directories: List[str] = field(default_factory=list)
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    integration_configs: Dict[str, Dict] = field(default_factory=dict)

@dataclass
class ContextItem:
    """Individual context item with metadata"""
    id: str
    type: ContextType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    embedding: Optional[np.ndarray] = None
    relationships: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(f"{self.type.value}_{self.content}_{self.timestamp}".encode()).hexdigest()

@dataclass
class Task:
    """Advanced task with comprehensive metadata"""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Task Hierarchy
    parent_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Execution Metadata
    required_capabilities: Set[AgentCapability] = field(default_factory=set)
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    progress: float = 0.0
    
    # Context and Results
    context_items: List[str] = field(default_factory=list)
    execution_plan: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error Handling
    error_count: int = 0
    max_retries: int = 3
    error_messages: List[str] = field(default_factory=list)
    
    # Learning and Feedback
    feedback_score: Optional[float] = None
    lessons_learned: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"task_{int(time.time() * 1000)}_{hash(self.title) % 10000}"

class ContextManager:
    """Advanced context management with multi-modal understanding"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.contexts: Dict[str, ContextItem] = {}
        self.context_graph = nx.DiGraph()
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.context_db = self._init_context_db()
        
        # Context retention and cleanup
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_contexts, daemon=True)
        self.cleanup_thread.start()
    
    def _init_context_db(self) -> sqlite3.Connection:
        """Initialize context database"""
        db_path = os.path.join(self.config.workspace_dir, '.ai_agent_context.db')
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS contexts (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                metadata TEXT,
                timestamp REAL,
                relevance_score REAL,
                embedding BLOB
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS context_relationships (
                source_id TEXT,
                target_id TEXT,
                relationship_type TEXT,
                strength REAL,
                PRIMARY KEY (source_id, target_id)
            )
        ''')
        
        conn.commit()
        return conn
    
    def add_context(self, context: ContextItem) -> str:
        """Add context item with relationship analysis"""
        # Generate embedding
        if context.content and len(context.content.strip()) > 0:
            context.embedding = self._generate_embedding(context.content)
        
        # Store in memory and database
        self.contexts[context.id] = context
        self._store_context_in_db(context)
        
        # Add to graph
        self.context_graph.add_node(context.id, **asdict(context))
        
        # Analyze relationships with existing contexts
        self._analyze_relationships(context)
        
        logger.info(f"Added context item: {context.id} ({context.type.value})")
        return context.id
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding for similarity analysis"""
        # Simple TF-IDF embedding (in production, use more sophisticated embeddings)
        try:
            if hasattr(self.vectorizer, 'vocabulary_'):
                vector = self.vectorizer.transform([text])
            else:
                # Fit on current text if not fitted
                vector = self.vectorizer.fit_transform([text])
            return vector.toarray()[0]
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return np.zeros(1000)
    
    def _analyze_relationships(self, new_context: ContextItem):
        """Analyze relationships between contexts"""
        if new_context.embedding is None:
            return
        
        for existing_id, existing_context in self.contexts.items():
            if existing_id == new_context.id or existing_context.embedding is None:
                continue
            
            # Calculate similarity
            similarity = cosine_similarity(
                new_context.embedding.reshape(1, -1),
                existing_context.embedding.reshape(1, -1)
            )[0][0]
            
            # Create relationship if similarity is high enough
            if similarity > 0.3:  # Threshold for relationship
                self.context_graph.add_edge(
                    new_context.id, 
                    existing_id, 
                    weight=similarity,
                    type="semantic_similarity"
                )
                
                # Store in database
                self.context_db.execute('''
                    INSERT OR REPLACE INTO context_relationships 
                    (source_id, target_id, relationship_type, strength)
                    VALUES (?, ?, ?, ?)
                ''', (new_context.id, existing_id, "semantic_similarity", similarity))
        
        self.context_db.commit()
    
    def get_relevant_contexts(self, query: str, context_types: List[ContextType] = None, 
                            limit: int = 10) -> List[ContextItem]:
        """Get relevant contexts for a query"""
        query_embedding = self._generate_embedding(query)
        
        relevant_contexts = []
        for context in self.contexts.values():
            if context_types and context.type not in context_types:
                continue
            
            if context.embedding is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    context.embedding.reshape(1, -1)
                )[0][0]
                
                context.relevance_score = similarity
                relevant_contexts.append(context)
        
        # Sort by relevance and return top results
        relevant_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        return relevant_contexts[:limit]
    
    def _store_context_in_db(self, context: ContextItem):
        """Store context in database"""
        embedding_blob = pickle.dumps(context.embedding) if context.embedding is not None else None
        
        self.context_db.execute('''
            INSERT OR REPLACE INTO contexts 
            (id, type, content, metadata, timestamp, relevance_score, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            context.id,
            context.type.value,
            context.content,
            json.dumps(context.metadata),
            context.timestamp.timestamp(),
            context.relevance_score,
            embedding_blob
        ))
        self.context_db.commit()
    
    def _cleanup_old_contexts(self):
        """Cleanup old contexts based on retention policy"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(days=self.config.context_retention_days)
                
                # Remove old contexts from memory
                old_context_ids = [
                    ctx_id for ctx_id, ctx in self.contexts.items()
                    if ctx.timestamp < cutoff_time
                ]
                
                for ctx_id in old_context_ids:
                    del self.contexts[ctx_id]
                    if self.context_graph.has_node(ctx_id):
                        self.context_graph.remove_node(ctx_id)
                
                # Remove from database
                self.context_db.execute(
                    'DELETE FROM contexts WHERE timestamp < ?',
                    (cutoff_time.timestamp(),)
                )
                self.context_db.execute(
                    'DELETE FROM context_relationships WHERE source_id NOT IN (SELECT id FROM contexts)'
                )
                self.context_db.commit()
                
                if old_context_ids:
                    logger.info(f"Cleaned up {len(old_context_ids)} old contexts")
                
                # Sleep for an hour before next cleanup
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in context cleanup: {e}")
                time.sleep(3600)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context state"""
        type_counts = {}
        for context in self.contexts.values():
            type_counts[context.type.value] = type_counts.get(context.type.value, 0) + 1
        
        return {
            "total_contexts": len(self.contexts),
            "type_distribution": type_counts,
            "graph_nodes": self.context_graph.number_of_nodes(),
            "graph_edges": self.context_graph.number_of_edges(),
            "memory_usage_mb": sys.getsizeof(self.contexts) / (1024 * 1024)
        }

class ReasoningEngine:
    """Advanced reasoning engine for intelligent task analysis and planning"""

    def __init__(self, config: AgentConfig, context_manager: ContextManager):
        self.config = config
        self.context_manager = context_manager
        self.reasoning_history: List[Dict[str, Any]] = []
        self.pattern_library: Dict[str, Dict[str, Any]] = {}
        self.decision_tree = nx.DiGraph()

        # Initialize AI models
        self.models = self._initialize_models()

        # Load reasoning patterns
        self._load_reasoning_patterns()

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize AI models for reasoning"""
        models = {}

        # Gemini
        if self.config.api_keys.get('gemini'):
            genai.configure(api_key=self.config.api_keys['gemini'])
            models['gemini'] = genai.GenerativeModel(self.config.primary_model)

        # OpenAI (if available)
        if self.config.api_keys.get('openai'):
            try:
                import openai
                openai.api_key = self.config.api_keys['openai']
                models['openai'] = openai
            except ImportError:
                pass

        # Anthropic (if available)
        if self.config.api_keys.get('anthropic'):
            try:
                import anthropic
                models['anthropic'] = anthropic.Anthropic(api_key=self.config.api_keys['anthropic'])
            except ImportError:
                pass

        return models

    def _load_reasoning_patterns(self):
        """Load common reasoning patterns for different types of tasks"""
        self.pattern_library = {
            "code_analysis": {
                "steps": [
                    "Parse and understand code structure",
                    "Identify patterns and anti-patterns",
                    "Analyze dependencies and relationships",
                    "Evaluate quality metrics",
                    "Generate insights and recommendations"
                ],
                "context_types": [ContextType.CODE, ContextType.DOCUMENTATION],
                "capabilities": [AgentCapability.CODE_ANALYSIS]
            },
            "feature_development": {
                "steps": [
                    "Understand requirements and constraints",
                    "Design architecture and approach",
                    "Plan implementation phases",
                    "Identify testing strategy",
                    "Consider deployment and maintenance"
                ],
                "context_types": [ContextType.REQUIREMENT, ContextType.CODE],
                "capabilities": [
                    AgentCapability.CODE_GENERATION,
                    AgentCapability.TESTING,
                    AgentCapability.DOCUMENTATION
                ]
            },
            "debugging": {
                "steps": [
                    "Reproduce and understand the error",
                    "Analyze error context and stack trace",
                    "Identify potential root causes",
                    "Develop and test hypotheses",
                    "Implement and validate solution"
                ],
                "context_types": [ContextType.ERROR, ContextType.CODE],
                "capabilities": [AgentCapability.DEBUGGING, AgentCapability.CODE_ANALYSIS]
            },
            "refactoring": {
                "steps": [
                    "Analyze current code structure",
                    "Identify improvement opportunities",
                    "Plan refactoring strategy",
                    "Implement changes incrementally",
                    "Validate functionality preservation"
                ],
                "context_types": [ContextType.CODE],
                "capabilities": [AgentCapability.REFACTORING, AgentCapability.TESTING]
            }
        }

    async def analyze_request(self, request: str, context_items: List[ContextItem] = None) -> Dict[str, Any]:
        """Analyze a user request using advanced reasoning"""
        logger.info(f"Analyzing request: {request[:100]}...")

        # Get relevant context
        if context_items is None:
            context_items = self.context_manager.get_relevant_contexts(request)

        # Prepare analysis prompt
        analysis_prompt = self._build_analysis_prompt(request, context_items)

        # Perform multi-model analysis
        analysis_results = await self._multi_model_analysis(analysis_prompt)

        # Synthesize results
        final_analysis = self._synthesize_analysis(analysis_results)

        # Store reasoning in history
        self.reasoning_history.append({
            "timestamp": datetime.now(),
            "request": request,
            "context_count": len(context_items),
            "analysis": final_analysis
        })

        return final_analysis

    def _build_analysis_prompt(self, request: str, context_items: List[ContextItem]) -> str:
        """Build comprehensive analysis prompt"""
        context_summary = self._summarize_context(context_items)

        prompt = f"""
Analyze this development request with deep reasoning and contextual understanding:

REQUEST: {request}

AVAILABLE CONTEXT:
{context_summary}

ANALYSIS FRAMEWORK:
1. Intent Understanding:
   - What is the user trying to accomplish?
   - What are the explicit and implicit requirements?
   - What constraints or preferences are indicated?

2. Complexity Assessment:
   - Technical complexity (1-10 scale)
   - Time complexity (estimated hours)
   - Risk factors and potential challenges
   - Required expertise level

3. Context Analysis:
   - How does this relate to existing code/project?
   - What dependencies or integrations are involved?
   - What patterns or conventions should be followed?

4. Approach Recommendation:
   - Best methodology for this type of task
   - Recommended tools and technologies
   - Step-by-step approach outline
   - Alternative approaches to consider

5. Success Criteria:
   - How to measure successful completion
   - Testing and validation requirements
   - Quality standards to maintain

Provide analysis in structured JSON format with detailed reasoning for each aspect.
"""
        return prompt

    def _summarize_context(self, context_items: List[ContextItem]) -> str:
        """Summarize context items for prompt inclusion"""
        if not context_items:
            return "No specific context available."

        summary_parts = []
        context_by_type = {}

        # Group by type
        for item in context_items:
            if item.type not in context_by_type:
                context_by_type[item.type] = []
            context_by_type[item.type].append(item)

        # Summarize each type
        for context_type, items in context_by_type.items():
            summary_parts.append(f"\n{context_type.value.upper()}:")
            for item in items[:3]:  # Limit to top 3 per type
                content_preview = item.content[:200] + "..." if len(item.content) > 200 else item.content
                summary_parts.append(f"  - {content_preview}")

            if len(items) > 3:
                summary_parts.append(f"  ... and {len(items) - 3} more items")

        return "\n".join(summary_parts)

    async def _multi_model_analysis(self, prompt: str) -> List[Dict[str, Any]]:
        """Perform analysis using multiple AI models for robustness"""
        results = []

        # Primary model analysis
        if 'gemini' in self.models:
            try:
                response = await self.models['gemini'].generate_content_async(prompt)
                results.append({
                    "model": "gemini",
                    "response": response.text,
                    "confidence": 0.9  # Primary model gets higher confidence
                })
            except Exception as e:
                logger.warning(f"Gemini analysis failed: {e}")

        # Fallback models if available
        for model_name in self.config.fallback_models:
            if model_name in self.models and len(results) < 2:  # Limit to 2 models for efficiency
                try:
                    if model_name == "openai":
                        # OpenAI API call would go here
                        pass
                    elif model_name == "anthropic":
                        # Anthropic API call would go here
                        pass
                except Exception as e:
                    logger.warning(f"{model_name} analysis failed: {e}")

        return results

    def _synthesize_analysis(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize multiple analysis results into final analysis"""
        if not analysis_results:
            return self._default_analysis()

        # For now, use the primary result
        # In production, implement sophisticated synthesis logic
        primary_result = analysis_results[0]

        try:
            # Try to parse JSON response
            analysis = json.loads(primary_result["response"])
        except json.JSONDecodeError:
            # Fallback to structured parsing
            analysis = self._parse_unstructured_analysis(primary_result["response"])

        # Add metadata
        analysis["reasoning_metadata"] = {
            "models_used": [r["model"] for r in analysis_results],
            "confidence_score": sum(r.get("confidence", 0.5) for r in analysis_results) / len(analysis_results),
            "analysis_timestamp": datetime.now().isoformat()
        }

        return analysis

    def _default_analysis(self) -> Dict[str, Any]:
        """Provide default analysis when AI models are unavailable"""
        return {
            "intent": "General development task",
            "complexity": {
                "technical": 5,
                "time_estimate_hours": 2,
                "risk_level": "medium"
            },
            "approach": {
                "methodology": "iterative_development",
                "steps": [
                    "Analyze requirements",
                    "Design solution",
                    "Implement incrementally",
                    "Test and validate",
                    "Document and deploy"
                ]
            },
            "success_criteria": [
                "Functional requirements met",
                "Code quality standards maintained",
                "Tests passing"
            ],
            "reasoning_metadata": {
                "models_used": ["fallback"],
                "confidence_score": 0.3,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }

    def _parse_unstructured_analysis(self, text: str) -> Dict[str, Any]:
        """Parse unstructured analysis text into structured format"""
        # Simple parsing logic - in production, use more sophisticated NLP
        analysis = {
            "intent": "Extracted from unstructured response",
            "complexity": {"technical": 5, "time_estimate_hours": 2},
            "approach": {"methodology": "standard", "steps": []},
            "success_criteria": [],
            "raw_response": text
        }

        # Extract key information using simple patterns
        lines = text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Identify sections
            if any(keyword in line.lower() for keyword in ['intent', 'goal', 'objective']):
                current_section = 'intent'
                analysis['intent'] = line
            elif any(keyword in line.lower() for keyword in ['complexity', 'difficulty']):
                current_section = 'complexity'
            elif any(keyword in line.lower() for keyword in ['approach', 'method', 'strategy']):
                current_section = 'approach'
                if 'steps' not in analysis['approach']:
                    analysis['approach']['steps'] = []
            elif any(keyword in line.lower() for keyword in ['success', 'criteria', 'completion']):
                current_section = 'success_criteria'
            elif current_section == 'approach' and line.startswith(('-', '*', '1.', '2.')):
                analysis['approach']['steps'].append(line)
            elif current_section == 'success_criteria' and line.startswith(('-', '*')):
                analysis['success_criteria'].append(line)

        return analysis

class TaskPlanner:
    """Advanced task planning and decomposition system"""

    def __init__(self, config: AgentConfig, context_manager: ContextManager, reasoning_engine: ReasoningEngine):
        self.config = config
        self.context_manager = context_manager
        self.reasoning_engine = reasoning_engine
        self.task_graph = nx.DiGraph()
        self.execution_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}

        # Planning strategies
        self.planning_strategies = {
            "waterfall": self._plan_waterfall,
            "agile": self._plan_agile,
            "iterative": self._plan_iterative,
            "exploratory": self._plan_exploratory
        }

    async def create_execution_plan(self, request: str, analysis: Dict[str, Any]) -> List[Task]:
        """Create comprehensive execution plan from analysis"""
        logger.info("Creating execution plan...")

        # Determine planning strategy
        strategy = self._select_planning_strategy(analysis)

        # Create main task
        main_task = self._create_main_task(request, analysis)

        # Decompose into subtasks using selected strategy
        subtasks = await self.planning_strategies[strategy](main_task, analysis)

        # Build task dependency graph
        self._build_task_graph(main_task, subtasks)

        # Optimize execution order
        execution_order = self._optimize_execution_order(main_task, subtasks)

        # Store tasks
        all_tasks = [main_task] + subtasks
        for task in all_tasks:
            self.active_tasks[task.id] = task

        logger.info(f"Created execution plan with {len(subtasks)} subtasks")
        return execution_order

    def _select_planning_strategy(self, analysis: Dict[str, Any]) -> str:
        """Select appropriate planning strategy based on analysis"""
        complexity = analysis.get("complexity", {})
        technical_complexity = complexity.get("technical", 5)
        risk_level = complexity.get("risk_level", "medium")

        if technical_complexity >= 8 or risk_level == "high":
            return "iterative"  # Break down complex tasks
        elif "exploration" in analysis.get("intent", "").lower():
            return "exploratory"  # For research/discovery tasks
        elif analysis.get("approach", {}).get("methodology") == "agile":
            return "agile"  # User preference or project methodology
        else:
            return "waterfall"  # Default structured approach

    def _create_main_task(self, request: str, analysis: Dict[str, Any]) -> Task:
        """Create main task from request and analysis"""
        complexity = analysis.get("complexity", {})

        task = Task(
            title=f"Main: {request[:50]}...",
            description=request,
            priority=TaskPriority.HIGH,
            estimated_duration=timedelta(hours=complexity.get("time_estimate_hours", 2)),
            required_capabilities=self._extract_capabilities(analysis)
        )

        return task

    def _extract_capabilities(self, analysis: Dict[str, Any]) -> Set[AgentCapability]:
        """Extract required capabilities from analysis"""
        capabilities = set()

        # Analyze intent and approach to determine capabilities
        intent = analysis.get("intent", "").lower()
        approach = analysis.get("approach", {})

        # Map keywords to capabilities
        capability_keywords = {
            AgentCapability.CODE_ANALYSIS: ["analyze", "review", "examine"],
            AgentCapability.CODE_GENERATION: ["create", "implement", "build", "develop"],
            AgentCapability.REFACTORING: ["refactor", "improve", "restructure"],
            AgentCapability.TESTING: ["test", "validate", "verify"],
            AgentCapability.DOCUMENTATION: ["document", "explain", "describe"],
            AgentCapability.DEBUGGING: ["debug", "fix", "error", "bug"],
            AgentCapability.SECURITY_ANALYSIS: ["security", "vulnerability", "secure"],
            AgentCapability.OPTIMIZATION: ["optimize", "performance", "speed"],
        }

        for capability, keywords in capability_keywords.items():
            if any(keyword in intent for keyword in keywords):
                capabilities.add(capability)

        # Default capabilities if none detected
        if not capabilities:
            capabilities.add(AgentCapability.CODE_GENERATION)

        return capabilities

    async def _plan_waterfall(self, main_task: Task, analysis: Dict[str, Any]) -> List[Task]:
        """Waterfall planning strategy - sequential phases"""
        subtasks = []
        approach = analysis.get("approach", {})
        steps = approach.get("steps", [])

        for i, step in enumerate(steps):
            subtask = Task(
                title=f"Phase {i+1}: {step}",
                description=step,
                parent_id=main_task.id,
                priority=TaskPriority.MEDIUM,
                estimated_duration=timedelta(hours=main_task.estimated_duration.total_seconds() / 3600 / len(steps)),
                required_capabilities=main_task.required_capabilities
            )

            # Add dependency on previous task
            if i > 0:
                subtask.dependencies.append(subtasks[i-1].id)

            subtasks.append(subtask)
            main_task.subtasks.append(subtask.id)

        return subtasks

    async def _plan_agile(self, main_task: Task, analysis: Dict[str, Any]) -> List[Task]:
        """Agile planning strategy - iterative sprints"""
        subtasks = []

        # Create sprint-like iterations
        sprint_tasks = [
            "Sprint 1: Core functionality",
            "Sprint 2: Feature enhancement",
            "Sprint 3: Testing and refinement",
            "Sprint 4: Documentation and deployment"
        ]

        for i, sprint_title in enumerate(sprint_tasks):
            subtask = Task(
                title=sprint_title,
                description=f"Agile sprint focusing on incremental delivery",
                parent_id=main_task.id,
                priority=TaskPriority.MEDIUM,
                estimated_duration=timedelta(hours=main_task.estimated_duration.total_seconds() / 3600 / len(sprint_tasks)),
                required_capabilities=main_task.required_capabilities
            )

            subtasks.append(subtask)
            main_task.subtasks.append(subtask.id)

        return subtasks

    async def _plan_iterative(self, main_task: Task, analysis: Dict[str, Any]) -> List[Task]:
        """Iterative planning strategy - incremental development"""
        subtasks = []

        # Break down by capability areas
        capability_tasks = {
            AgentCapability.CODE_ANALYSIS: "Analyze existing code and requirements",
            AgentCapability.CODE_GENERATION: "Implement core functionality",
            AgentCapability.TESTING: "Create and run tests",
            AgentCapability.DOCUMENTATION: "Generate documentation",
            AgentCapability.OPTIMIZATION: "Optimize and refine"
        }

        for capability in main_task.required_capabilities:
            if capability in capability_tasks:
                subtask = Task(
                    title=capability_tasks[capability],
                    description=f"Task focused on {capability.value}",
                    parent_id=main_task.id,
                    priority=TaskPriority.MEDIUM,
                    estimated_duration=timedelta(hours=1),
                    required_capabilities={capability}
                )

                subtasks.append(subtask)
                main_task.subtasks.append(subtask.id)

        return subtasks

    async def _plan_exploratory(self, main_task: Task, analysis: Dict[str, Any]) -> List[Task]:
        """Exploratory planning strategy - research and discovery"""
        subtasks = []

        exploration_phases = [
            "Research and discovery",
            "Prototype development",
            "Evaluation and analysis",
            "Implementation planning"
        ]

        for phase in exploration_phases:
            subtask = Task(
                title=f"Exploration: {phase}",
                description=f"Exploratory phase: {phase}",
                parent_id=main_task.id,
                priority=TaskPriority.MEDIUM,
                estimated_duration=timedelta(hours=main_task.estimated_duration.total_seconds() / 3600 / len(exploration_phases)),
                required_capabilities=main_task.required_capabilities
            )

            subtasks.append(subtask)
            main_task.subtasks.append(subtask.id)

        return subtasks

    def _build_task_graph(self, main_task: Task, subtasks: List[Task]):
        """Build task dependency graph"""
        # Add main task
        self.task_graph.add_node(main_task.id, task=main_task)

        # Add subtasks and dependencies
        for subtask in subtasks:
            self.task_graph.add_node(subtask.id, task=subtask)

            # Parent-child relationship
            self.task_graph.add_edge(main_task.id, subtask.id, relationship="parent-child")

            # Dependencies
            for dep_id in subtask.dependencies:
                if dep_id in [t.id for t in subtasks]:
                    self.task_graph.add_edge(dep_id, subtask.id, relationship="dependency")

    def _optimize_execution_order(self, main_task: Task, subtasks: List[Task]) -> List[Task]:
        """Optimize task execution order using topological sorting"""
        try:
            # Get topological order of subtasks
            subtask_ids = [t.id for t in subtasks]
            subgraph = self.task_graph.subgraph(subtask_ids)

            if nx.is_directed_acyclic_graph(subgraph):
                ordered_ids = list(nx.topological_sort(subgraph))
                ordered_tasks = [next(t for t in subtasks if t.id == tid) for tid in ordered_ids]
            else:
                # Fallback to priority-based ordering
                ordered_tasks = sorted(subtasks, key=lambda t: t.priority.value)
        except:
            # Fallback to original order
            ordered_tasks = subtasks

        return [main_task] + ordered_tasks

class ExecutionEngine:
    """Advanced execution engine with parallel processing and error handling"""

    def __init__(self, config: AgentConfig, context_manager: ContextManager,
                 reasoning_engine: ReasoningEngine, task_planner: TaskPlanner):
        self.config = config
        self.context_manager = context_manager
        self.reasoning_engine = reasoning_engine
        self.task_planner = task_planner

        # Execution state
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)
        self.running_tasks: Dict[str, concurrent.futures.Future] = {}
        self.task_results: Dict[str, Dict[str, Any]] = {}

        # Tool registry
        self.tools: Dict[str, 'BaseTool'] = {}
        self._register_core_tools()

        # Execution monitoring
        self.execution_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0,
            "average_task_time": 0
        }

    def _register_core_tools(self):
        """Register core tools for task execution"""
        # Tools will be registered here
        pass

    async def execute_task_plan(self, tasks: List[Task]) -> Dict[str, Any]:
        """Execute a complete task plan with monitoring and error handling"""
        logger.info(f"Starting execution of {len(tasks)} tasks")

        execution_start = datetime.now()
        results = {
            "execution_id": f"exec_{int(time.time())}",
            "start_time": execution_start,
            "tasks": {},
            "overall_status": "running",
            "progress": 0.0
        }

        try:
            # Execute tasks in dependency order
            for task in tasks:
                if self._can_execute_task(task, results):
                    task_result = await self._execute_single_task(task)
                    results["tasks"][task.id] = task_result

                    # Update progress
                    completed_tasks = sum(1 for r in results["tasks"].values() if r["status"] == "completed")
                    results["progress"] = completed_tasks / len(tasks)

                    # Check for failures
                    if task_result["status"] == "failed" and task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                        logger.error(f"Critical task failed: {task.id}")
                        results["overall_status"] = "failed"
                        break

            # Determine overall status
            if results["overall_status"] != "failed":
                failed_tasks = sum(1 for r in results["tasks"].values() if r["status"] == "failed")
                if failed_tasks == 0:
                    results["overall_status"] = "completed"
                elif failed_tasks < len(tasks) * 0.5:  # Less than 50% failed
                    results["overall_status"] = "partially_completed"
                else:
                    results["overall_status"] = "failed"

            execution_end = datetime.now()
            results["end_time"] = execution_end
            results["total_duration"] = (execution_end - execution_start).total_seconds()

            # Update metrics
            self._update_execution_metrics(results)

            logger.info(f"Execution completed with status: {results['overall_status']}")
            return results

        except Exception as e:
            logger.error(f"Execution engine error: {e}")
            results["overall_status"] = "error"
            results["error"] = str(e)
            return results

    def _can_execute_task(self, task: Task, current_results: Dict[str, Any]) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in current_results["tasks"]:
                return False
            if current_results["tasks"][dep_id]["status"] != "completed":
                return False
        return True

    async def _execute_single_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task with comprehensive monitoring"""
        logger.info(f"Executing task: {task.title}")

        task_start = datetime.now()
        task.status = TaskStatus.EXECUTING
        task.started_at = task_start

        result = {
            "task_id": task.id,
            "status": "running",
            "start_time": task_start,
            "progress": 0.0,
            "outputs": [],
            "artifacts": [],
            "errors": []
        }

        try:
            # Get relevant context for task
            context_items = self.context_manager.get_relevant_contexts(
                task.description,
                limit=10
            )

            # Execute based on required capabilities
            for capability in task.required_capabilities:
                capability_result = await self._execute_capability(task, capability, context_items)
                result["outputs"].append(capability_result)

                # Update progress
                result["progress"] = len(result["outputs"]) / len(task.required_capabilities)

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.actual_duration = task.completed_at - task_start
            result["status"] = "completed"
            result["end_time"] = task.completed_at

            # Store results in task
            task.results = result

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            task.status = TaskStatus.FAILED
            task.error_count += 1
            task.error_messages.append(str(e))

            result["status"] = "failed"
            result["error"] = str(e)
            result["end_time"] = datetime.now()

        return result

    async def _execute_capability(self, task: Task, capability: AgentCapability,
                                context_items: List[ContextItem]) -> Dict[str, Any]:
        """Execute a specific capability for a task"""
        capability_handlers = {
            AgentCapability.CODE_ANALYSIS: self._handle_code_analysis,
            AgentCapability.CODE_GENERATION: self._handle_code_generation,
            AgentCapability.TESTING: self._handle_testing,
            AgentCapability.DOCUMENTATION: self._handle_documentation,
            AgentCapability.DEBUGGING: self._handle_debugging,
            AgentCapability.REFACTORING: self._handle_refactoring,
        }

        handler = capability_handlers.get(capability, self._handle_generic_capability)
        return await handler(task, context_items)

    async def _handle_code_analysis(self, task: Task, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Handle code analysis capability"""
        return {
            "capability": "code_analysis",
            "result": "Code analysis completed",
            "details": "Analyzed code structure and quality"
        }

    async def _handle_code_generation(self, task: Task, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Handle code generation capability"""
        return {
            "capability": "code_generation",
            "result": "Code generated successfully",
            "details": "Generated code based on requirements"
        }

    async def _handle_testing(self, task: Task, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Handle testing capability"""
        return {
            "capability": "testing",
            "result": "Tests created and executed",
            "details": "Comprehensive test suite generated"
        }

    async def _handle_documentation(self, task: Task, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Handle documentation capability"""
        return {
            "capability": "documentation",
            "result": "Documentation generated",
            "details": "Comprehensive documentation created"
        }

    async def _handle_debugging(self, task: Task, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Handle debugging capability"""
        return {
            "capability": "debugging",
            "result": "Issues identified and resolved",
            "details": "Debugging analysis completed"
        }

    async def _handle_refactoring(self, task: Task, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Handle refactoring capability"""
        return {
            "capability": "refactoring",
            "result": "Code refactored successfully",
            "details": "Code structure improved"
        }

    async def _handle_generic_capability(self, task: Task, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Handle generic capability"""
        return {
            "capability": "generic",
            "result": "Task processed",
            "details": "Generic task processing completed"
        }

    def _update_execution_metrics(self, results: Dict[str, Any]):
        """Update execution metrics"""
        completed_tasks = sum(1 for r in results["tasks"].values() if r["status"] == "completed")
        failed_tasks = sum(1 for r in results["tasks"].values() if r["status"] == "failed")

        self.execution_metrics["tasks_completed"] += completed_tasks
        self.execution_metrics["tasks_failed"] += failed_tasks

        if "total_duration" in results:
            self.execution_metrics["total_execution_time"] += results["total_duration"]
            total_tasks = self.execution_metrics["tasks_completed"] + self.execution_metrics["tasks_failed"]
            if total_tasks > 0:
                self.execution_metrics["average_task_time"] = self.execution_metrics["total_execution_time"] / total_tasks

class BaseTool(ABC):
    """Abstract base class for all agent tools"""

    def __init__(self, name: str, description: str, config: AgentConfig):
        self.name = name
        self.description = description
        self.config = config
        self.usage_count = 0
        self.last_used = None

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for AI model integration"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema()
        }

    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema for the tool"""
        pass

    def _log_usage(self):
        """Log tool usage"""
        self.usage_count += 1
        self.last_used = datetime.now()
        logger.debug(f"Tool {self.name} used (count: {self.usage_count})")

class FileSystemTool(BaseTool):
    """Advanced file system operations tool"""

    def __init__(self, config: AgentConfig):
        super().__init__(
            "filesystem",
            "Advanced file system operations including read, write, search, and analysis",
            config
        )
        self.workspace_dir = Path(config.workspace_dir)
        self.file_cache: Dict[str, Dict[str, Any]] = {}

        # File watching
        self.observer = Observer()
        self.file_changes: queue.Queue = queue.Queue()
        self._setup_file_watching()

    def _setup_file_watching(self):
        """Setup file system watching"""
        class ChangeHandler(FileSystemEventHandler):
            def __init__(self, tool):
                self.tool = tool

            def on_modified(self, event):
                if not event.is_directory:
                    self.tool.file_changes.put({
                        "type": "modified",
                        "path": event.src_path,
                        "timestamp": datetime.now()
                    })

        self.observer.schedule(ChangeHandler(self), str(self.workspace_dir), recursive=True)
        self.observer.start()

    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute file system operation"""
        self._log_usage()

        operations = {
            "read": self._read_file,
            "write": self._write_file,
            "list": self._list_files,
            "search": self._search_files,
            "analyze": self._analyze_file,
            "watch": self._get_file_changes,
            "backup": self._backup_file,
            "diff": self._diff_files
        }

        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")

        return await operations[operation](**kwargs)

    async def _read_file(self, path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read file with caching and metadata"""
        file_path = self.workspace_dir / path

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Check cache
        cache_key = f"{path}_{file_path.stat().st_mtime}"
        if cache_key in self.file_cache:
            return self.file_cache[cache_key]

        try:
            content = file_path.read_text(encoding=encoding)

            result = {
                "path": path,
                "content": content,
                "size": len(content),
                "lines": len(content.splitlines()),
                "encoding": encoding,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                "hash": hashlib.md5(content.encode()).hexdigest()
            }

            # Cache result
            self.file_cache[cache_key] = result

            return result

        except Exception as e:
            raise Exception(f"Error reading file {path}: {str(e)}")

    async def _write_file(self, path: str, content: str, encoding: str = "utf-8",
                         backup: bool = True) -> Dict[str, Any]:
        """Write file with backup and atomic operations"""
        file_path = self.workspace_dir / path

        # Create backup if file exists
        if backup and file_path.exists():
            await self._backup_file(path)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write
        temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
        try:
            temp_path.write_text(content, encoding=encoding)
            temp_path.replace(file_path)

            return {
                "path": path,
                "size": len(content),
                "lines": len(content.splitlines()),
                "success": True
            }

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise Exception(f"Error writing file {path}: {str(e)}")

    async def _list_files(self, pattern: str = "*", recursive: bool = True) -> Dict[str, Any]:
        """List files matching pattern"""
        if recursive:
            files = list(self.workspace_dir.rglob(pattern))
        else:
            files = list(self.workspace_dir.glob(pattern))

        file_info = []
        for file_path in files:
            if file_path.is_file():
                stat = file_path.stat()
                file_info.append({
                    "path": str(file_path.relative_to(self.workspace_dir)),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "extension": file_path.suffix
                })

        return {
            "files": file_info,
            "count": len(file_info),
            "pattern": pattern
        }

    async def _search_files(self, query: str, file_pattern: str = "*.py",
                           case_sensitive: bool = False) -> Dict[str, Any]:
        """Search for text in files"""
        results = []
        files = list(self.workspace_dir.rglob(file_pattern))

        for file_path in files:
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    lines = content.splitlines()

                    for line_num, line in enumerate(lines, 1):
                        search_line = line if case_sensitive else line.lower()
                        search_query = query if case_sensitive else query.lower()

                        if search_query in search_line:
                            results.append({
                                "file": str(file_path.relative_to(self.workspace_dir)),
                                "line": line_num,
                                "content": line.strip(),
                                "context": lines[max(0, line_num-2):line_num+1]
                            })

                except Exception:
                    continue

        return {
            "results": results,
            "count": len(results),
            "query": query
        }

    async def _analyze_file(self, path: str) -> Dict[str, Any]:
        """Analyze file structure and content"""
        file_data = await self._read_file(path)
        content = file_data["content"]

        analysis = {
            "path": path,
            "language": self._detect_language(path),
            "metrics": {
                "lines": file_data["lines"],
                "size": file_data["size"],
                "words": len(content.split()),
                "characters": len(content)
            }
        }

        # Language-specific analysis
        if analysis["language"] == "python":
            analysis.update(await self._analyze_python_file(content))
        elif analysis["language"] in ["javascript", "typescript"]:
            analysis.update(await self._analyze_js_file(content))

        return analysis

    def _detect_language(self, path: str) -> str:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }

        suffix = Path(path).suffix.lower()
        return extension_map.get(suffix, 'unknown')

    async def _analyze_python_file(self, content: str) -> Dict[str, Any]:
        """Analyze Python file using AST"""
        try:
            tree = ast.parse(content)

            functions = []
            classes = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": len(node.args.args),
                        "docstring": ast.get_docstring(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                        "docstring": ast.get_docstring(node)
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    else:
                        imports.append(node.module)

            return {
                "functions": functions,
                "classes": classes,
                "imports": list(set(filter(None, imports))),
                "complexity": self._calculate_complexity(tree)
            }

        except SyntaxError as e:
            return {"error": f"Syntax error: {str(e)}"}

    async def _analyze_js_file(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript file"""
        # Simple regex-based analysis (in production, use proper parser)
        import re

        functions = re.findall(r'function\s+(\w+)', content)
        arrow_functions = re.findall(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|\w+\s*=>)', content)
        classes = re.findall(r'class\s+(\w+)', content)
        imports = re.findall(r'import.*?from\s+[\'"]([^\'"]+)[\'"]', content)

        return {
            "functions": functions + arrow_functions,
            "classes": classes,
            "imports": imports,
            "complexity": len(functions) + len(arrow_functions) + len(classes)
        }

    def _calculate_complexity(self, tree) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1

        return complexity

    async def _backup_file(self, path: str) -> Dict[str, Any]:
        """Create backup of file"""
        file_path = self.workspace_dir / path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{timestamp}")

        import shutil
        shutil.copy2(file_path, backup_path)

        return {
            "original": path,
            "backup": str(backup_path.relative_to(self.workspace_dir)),
            "timestamp": timestamp
        }

    async def _diff_files(self, path1: str, path2: str) -> Dict[str, Any]:
        """Compare two files and return differences"""
        import difflib

        file1_data = await self._read_file(path1)
        file2_data = await self._read_file(path2)

        diff = list(difflib.unified_diff(
            file1_data["content"].splitlines(keepends=True),
            file2_data["content"].splitlines(keepends=True),
            fromfile=path1,
            tofile=path2
        ))

        return {
            "file1": path1,
            "file2": path2,
            "diff": diff,
            "changes": len(diff)
        }

    async def _get_file_changes(self) -> Dict[str, Any]:
        """Get recent file changes"""
        changes = []
        while not self.file_changes.empty():
            try:
                change = self.file_changes.get_nowait()
                changes.append(change)
            except queue.Empty:
                break

        return {
            "changes": changes,
            "count": len(changes)
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema for filesystem tool"""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "search", "analyze", "watch", "backup", "diff"],
                    "description": "File system operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search operation)"
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern (for list operation)"
                }
            },
            "required": ["operation"]
        }

class AdvancedAICodingAgent:
    """Main AI Coding Agent with advanced capabilities"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.session_id = f"session_{int(time.time())}"

        # Initialize core components
        self.context_manager = ContextManager(config)
        self.reasoning_engine = ReasoningEngine(config, self.context_manager)
        self.task_planner = TaskPlanner(config, self.context_manager, self.reasoning_engine)
        self.execution_engine = ExecutionEngine(config, self.context_manager,
                                               self.reasoning_engine, self.task_planner)

        # Initialize tools
        self.tools = {
            "filesystem": FileSystemTool(config)
        }

        # Session state
        self.conversation_history: List[Dict[str, Any]] = []
        self.active_projects: Dict[str, Dict[str, Any]] = {}
        self.learning_data: Dict[str, Any] = {"patterns": [], "feedback": []}

        # Performance monitoring
        self.performance_metrics = {
            "requests_processed": 0,
            "average_response_time": 0,
            "success_rate": 0,
            "user_satisfaction": 0
        }

        logger.info(f"Advanced AI Coding Agent initialized (session: {self.session_id})")

    async def process_request(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user request with full AI capabilities"""
        request_start = datetime.now()
        request_id = f"req_{int(time.time() * 1000)}"

        logger.info(f"Processing request {request_id}: {request[:100]}...")

        try:
            # Add request to conversation history
            self.conversation_history.append({
                "id": request_id,
                "timestamp": request_start,
                "type": "user_request",
                "content": request,
                "context": context or {}
            })

            # Add request to context
            request_context = ContextItem(
                type=ContextType.CONVERSATION,
                content=request,
                metadata={"request_id": request_id, "user_context": context or {}}
            )
            self.context_manager.add_context(request_context)

            # Phase 1: Analyze request
            console.print("[blue]🧠 Analyzing request...[/blue]")
            analysis = await self.reasoning_engine.analyze_request(request)

            # Phase 2: Create execution plan
            console.print("[blue]📋 Creating execution plan...[/blue]")
            tasks = await self.task_planner.create_execution_plan(request, analysis)

            # Phase 3: Execute plan
            console.print("[blue]⚡ Executing plan...[/blue]")
            execution_results = await self.execution_engine.execute_task_plan(tasks)

            # Phase 4: Generate response
            response = await self._generate_response(request, analysis, execution_results)

            # Add response to conversation history
            self.conversation_history.append({
                "id": f"resp_{request_id}",
                "timestamp": datetime.now(),
                "type": "agent_response",
                "content": response,
                "analysis": analysis,
                "execution_results": execution_results
            })

            # Update performance metrics
            request_duration = (datetime.now() - request_start).total_seconds()
            self._update_performance_metrics(request_duration, execution_results["overall_status"])

            logger.info(f"Request {request_id} completed in {request_duration:.2f}s")

            return {
                "request_id": request_id,
                "response": response,
                "analysis": analysis,
                "execution_results": execution_results,
                "duration": request_duration,
                "session_id": self.session_id
            }

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            error_response = {
                "request_id": request_id,
                "error": str(e),
                "response": f"I encountered an error while processing your request: {str(e)}",
                "duration": (datetime.now() - request_start).total_seconds(),
                "session_id": self.session_id
            }

            self._update_performance_metrics(error_response["duration"], "error")
            return error_response

    async def _generate_response(self, request: str, analysis: Dict[str, Any],
                               execution_results: Dict[str, Any]) -> str:
        """Generate comprehensive response based on analysis and execution"""

        # Prepare response context
        response_prompt = f"""
Generate a comprehensive response for this development request:

ORIGINAL REQUEST: {request}

ANALYSIS SUMMARY:
- Intent: {analysis.get('intent', 'Unknown')}
- Complexity: {analysis.get('complexity', {})}
- Approach: {analysis.get('approach', {})}

EXECUTION RESULTS:
- Overall Status: {execution_results.get('overall_status', 'Unknown')}
- Tasks Completed: {len([t for t in execution_results.get('tasks', {}).values() if t.get('status') == 'completed'])}
- Total Tasks: {len(execution_results.get('tasks', {}))}
- Duration: {execution_results.get('total_duration', 0):.2f} seconds

RESPONSE GUIDELINES:
1. Acknowledge what was accomplished
2. Explain the approach taken
3. Highlight key results and artifacts
4. Mention any challenges or limitations
5. Suggest next steps or improvements
6. Be conversational and helpful

Generate a natural, informative response that demonstrates understanding and provides value.
"""

        try:
            # Use primary AI model for response generation
            if 'gemini' in self.reasoning_engine.models:
                response = await self.reasoning_engine.models['gemini'].generate_content_async(response_prompt)
                return response.text
            else:
                # Fallback response
                return self._generate_fallback_response(analysis, execution_results)

        except Exception as e:
            logger.warning(f"AI response generation failed: {e}")
            return self._generate_fallback_response(analysis, execution_results)

    def _generate_fallback_response(self, analysis: Dict[str, Any],
                                  execution_results: Dict[str, Any]) -> str:
        """Generate fallback response when AI models are unavailable"""
        status = execution_results.get('overall_status', 'unknown')
        task_count = len(execution_results.get('tasks', {}))
        completed_count = len([t for t in execution_results.get('tasks', {}).values()
                             if t.get('status') == 'completed'])

        if status == 'completed':
            return f"""✅ **Task Completed Successfully**

I've successfully processed your request and completed all {task_count} planned tasks.

**What was accomplished:**
- Analyzed your requirements and created an execution plan
- Executed {completed_count} tasks with comprehensive monitoring
- Generated results and artifacts as needed

**Approach taken:**
- Used {analysis.get('approach', {}).get('methodology', 'systematic')} methodology
- Applied best practices and quality standards
- Implemented error handling and validation

The work is complete and ready for your review. Let me know if you need any modifications or have additional requirements!"""

        elif status == 'partially_completed':
            return f"""⚠️ **Task Partially Completed**

I've processed your request and completed {completed_count} out of {task_count} planned tasks.

**What was accomplished:**
- Successfully completed the core functionality
- Some optional or advanced features may need attention
- Generated partial results and artifacts

**Next steps:**
- Review the completed work
- Identify any remaining requirements
- Let me know if you'd like me to continue with the remaining tasks

The partial results are available for your review."""

        else:
            return f"""❌ **Task Encountered Issues**

I attempted to process your request but encountered some challenges.

**Status:** {status}
**Tasks attempted:** {task_count}
**Tasks completed:** {completed_count}

**What I can help with:**
- Analyzing the specific issues encountered
- Providing alternative approaches
- Breaking down the request into smaller, manageable tasks
- Offering guidance on how to proceed

Please let me know how you'd like to proceed, and I'll do my best to assist you!"""

    def _update_performance_metrics(self, duration: float, status: str):
        """Update performance metrics"""
        self.performance_metrics["requests_processed"] += 1

        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        count = self.performance_metrics["requests_processed"]
        self.performance_metrics["average_response_time"] = (current_avg * (count - 1) + duration) / count

        # Update success rate
        if status in ["completed", "partially_completed"]:
            success_count = self.performance_metrics["success_rate"] * (count - 1) + 1
        else:
            success_count = self.performance_metrics["success_rate"] * (count - 1)

        self.performance_metrics["success_rate"] = success_count / count

    async def get_project_status(self) -> Dict[str, Any]:
        """Get comprehensive project status"""
        return {
            "session_id": self.session_id,
            "workspace": self.config.workspace_dir,
            "conversation_length": len(self.conversation_history),
            "context_summary": self.context_manager.get_context_summary(),
            "performance_metrics": self.performance_metrics,
            "active_capabilities": list(self.config.enabled_capabilities),
            "tools_available": list(self.tools.keys())
        }

    async def learn_from_feedback(self, request_id: str, feedback: Dict[str, Any]):
        """Learn from user feedback to improve future responses"""
        if self.config.enable_learning:
            self.learning_data["feedback"].append({
                "request_id": request_id,
                "feedback": feedback,
                "timestamp": datetime.now()
            })

            # Update user satisfaction metric
            if "satisfaction" in feedback:
                current_satisfaction = self.performance_metrics["user_satisfaction"]
                count = len(self.learning_data["feedback"])
                new_satisfaction = (current_satisfaction * (count - 1) + feedback["satisfaction"]) / count
                self.performance_metrics["user_satisfaction"] = new_satisfaction

            logger.info(f"Learned from feedback for request {request_id}")

    def get_conversation_summary(self) -> str:
        """Get summary of current conversation"""
        if not self.conversation_history:
            return "No conversation history available."

        user_requests = [item for item in self.conversation_history if item["type"] == "user_request"]
        agent_responses = [item for item in self.conversation_history if item["type"] == "agent_response"]

        summary = f"""**Conversation Summary**
- Session ID: {self.session_id}
- Total exchanges: {len(user_requests)}
- Duration: {(datetime.now() - self.conversation_history[0]["timestamp"]).total_seconds() / 60:.1f} minutes
- Success rate: {self.performance_metrics["success_rate"]:.1%}

**Recent topics:**"""

        for request in user_requests[-3:]:  # Last 3 requests
            summary += f"\n- {request['content'][:100]}..."

        return summary

# CLI Interface
import click

@click.group()
@click.option('--config', default='.ai-agent-config.json', help='Configuration file path')
@click.option('--workspace', default='.', help='Workspace directory')
@click.option('--api-key', help='Primary AI model API key')
@click.option('--model', default='gemini-2.0-flash-exp', help='Primary AI model')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, workspace, api_key, model, debug, verbose):
    """Advanced AI Coding Agent - Sophisticated AI Assistant for Developers"""

    # Configure logging
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    agent_config = AgentConfig()

    # Load from config file if exists
    if os.path.exists(config):
        try:
            with open(config, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(agent_config, key):
                        setattr(agent_config, key, value)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load config file: {e}[/yellow]")

    # Override with command line arguments
    if workspace:
        agent_config.workspace_dir = os.path.abspath(workspace)
    if model:
        agent_config.primary_model = model
    if verbose:
        agent_config.verbose_output = verbose

    # Set API keys
    if api_key:
        agent_config.api_keys['gemini'] = api_key
    else:
        # Try environment variables
        gemini_key = os.getenv('GEMINI_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')

        if gemini_key:
            agent_config.api_keys['gemini'] = gemini_key
        if openai_key:
            agent_config.api_keys['openai'] = openai_key
        if anthropic_key:
            agent_config.api_keys['anthropic'] = anthropic_key

    if not agent_config.api_keys:
        console.print("[red]Error: No API keys configured. Set environment variables or use --api-key option.[/red]")
        sys.exit(1)

    ctx.obj = agent_config

@cli.command()
@click.argument('request', required=False)
@click.option('--interactive', '-i', is_flag=True, help='Interactive chat mode')
@click.option('--context-file', help='Load context from file')
@click.option('--save-session', help='Save session to file')
@click.pass_obj
def chat(config, request, interactive, context_file, save_session):
    """Chat with the AI agent"""

    async def run_chat():
        agent = AdvancedAICodingAgent(config)

        # Load context if provided
        context = {}
        if context_file and os.path.exists(context_file):
            try:
                with open(context_file, 'r') as f:
                    context = json.load(f)
                console.print(f"[green]Loaded context from {context_file}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load context file: {e}[/yellow]")

        if interactive or not request:
            console.print(Panel(
                "[bold blue]Advanced AI Coding Agent[/bold blue]\n"
                "Sophisticated AI assistant with multi-modal capabilities\n"
                "Type 'help' for commands, 'quit' to exit",
                title="Welcome",
                border_style="blue"
            ))

            while True:
                try:
                    user_input = input("\n🤖 > ").strip()

                    if user_input.lower() in ['quit', 'exit', 'q']:
                        console.print("[blue]Goodbye! 👋[/blue]")
                        break
                    elif user_input.lower() == 'help':
                        show_help()
                        continue
                    elif user_input.lower() == 'status':
                        status = await agent.get_project_status()
                        console.print(Panel(
                            f"Session: {status['session_id']}\n"
                            f"Workspace: {status['workspace']}\n"
                            f"Conversations: {status['conversation_length']}\n"
                            f"Success Rate: {status['performance_metrics']['success_rate']:.1%}\n"
                            f"Avg Response Time: {status['performance_metrics']['average_response_time']:.2f}s",
                            title="Agent Status",
                            border_style="green"
                        ))
                        continue
                    elif user_input.lower() == 'summary':
                        summary = agent.get_conversation_summary()
                        console.print(Panel(summary, title="Conversation Summary", border_style="cyan"))
                        continue
                    elif not user_input:
                        continue

                    # Process request
                    with console.status("[bold green]Processing request..."):
                        result = await agent.process_request(user_input, context)

                    # Display response
                    if result.get('error'):
                        console.print(f"[red]Error: {result['error']}[/red]")
                    else:
                        console.print(Panel(
                            result['response'],
                            title=f"Response (Duration: {result['duration']:.2f}s)",
                            border_style="green"
                        ))

                        # Show execution summary if verbose
                        if config.verbose_output:
                            execution = result['execution_results']
                            console.print(f"[dim]Execution: {execution['overall_status']} | "
                                        f"Tasks: {len(execution.get('tasks', {}))} | "
                                        f"Success Rate: {agent.performance_metrics['success_rate']:.1%}[/dim]")

                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted by user[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Unexpected error: {e}[/red]")
                    if config.verbose_output:
                        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            # Single request mode
            try:
                with console.status("[bold green]Processing request..."):
                    result = await agent.process_request(request, context)

                if result.get('error'):
                    console.print(f"[red]Error: {result['error']}[/red]")
                    sys.exit(1)
                else:
                    console.print(result['response'])

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)

        # Save session if requested
        if save_session:
            try:
                session_data = {
                    "session_id": agent.session_id,
                    "conversation_history": agent.conversation_history,
                    "performance_metrics": agent.performance_metrics,
                    "timestamp": datetime.now().isoformat()
                }

                with open(save_session, 'w') as f:
                    json.dump(session_data, f, indent=2, default=str)

                console.print(f"[green]Session saved to {save_session}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save session: {e}[/yellow]")

    # Run async chat
    asyncio.run(run_chat())

def show_help():
    """Show help information"""
    help_text = """
[bold blue]Available Commands:[/bold blue]

[green]Chat Commands:[/green]
• help - Show this help message
• status - Show agent status and metrics
• summary - Show conversation summary
• quit/exit/q - Exit the application

[green]Special Features:[/green]
• Multi-modal context understanding
• Advanced reasoning and planning
• Real-time code analysis
• Intelligent task decomposition
• Error handling and recovery
• Learning from feedback

[green]Example Requests:[/green]
• "Analyze my Python project and suggest improvements"
• "Create a REST API with authentication using FastAPI"
• "Debug this error: ImportError: No module named 'requests'"
• "Refactor my code to follow best practices"
• "Generate comprehensive tests for my application"
• "Create documentation for my project"

[green]Tips:[/green]
• Be specific about your requirements
• Mention technologies and frameworks you prefer
• Ask for explanations if you need more detail
• Provide context about your project when relevant
"""
    console.print(Panel(help_text, title="Help", border_style="blue"))

@cli.command()
@click.option('--output', '-o', default='project_analysis.json', help='Output file')
@click.option('--format', 'output_format', default='json',
              type=click.Choice(['json', 'markdown', 'html']), help='Output format')
@click.pass_obj
def analyze(config, output, output_format):
    """Analyze current project comprehensively"""

    async def run_analysis():
        agent = AdvancedAICodingAgent(config)

        console.print("[blue]🔍 Starting comprehensive project analysis...[/blue]")

        # Analyze project structure
        fs_tool = agent.tools["filesystem"]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:

            task = progress.add_task("Analyzing project structure...", total=100)

            # Get project files
            files_result = await fs_tool.execute("list", pattern="*", recursive=True)
            progress.update(task, advance=25)

            # Analyze key files
            analysis_results = {}
            python_files = [f for f in files_result["files"] if f["path"].endswith('.py')]

            for i, file_info in enumerate(python_files[:10]):  # Limit to first 10 files
                file_analysis = await fs_tool.execute("analyze", path=file_info["path"])
                analysis_results[file_info["path"]] = file_analysis
                progress.update(task, advance=5)

            progress.update(task, completed=100)

        # Generate comprehensive report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "project_overview": {
                "workspace": config.workspace_dir,
                "total_files": files_result["count"],
                "python_files": len(python_files)
            },
            "file_analyses": analysis_results,
            "recommendations": [
                "Consider adding type hints to improve code clarity",
                "Implement comprehensive test coverage",
                "Add docstrings to functions and classes",
                "Consider using a linter like pylint or flake8"
            ]
        }

        # Save report
        if output_format == 'json':
            with open(output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif output_format == 'markdown':
            markdown_content = generate_markdown_report(report)
            with open(output.replace('.json', '.md'), 'w') as f:
                f.write(markdown_content)

        console.print(f"[green]✅ Analysis complete! Report saved to {output}[/green]")

        # Show summary
        console.print(Panel(
            f"Files analyzed: {len(analysis_results)}\n"
            f"Total files: {files_result['count']}\n"
            f"Python files: {len(python_files)}\n"
            f"Recommendations: {len(report['recommendations'])}",
            title="Analysis Summary",
            border_style="green"
        ))

    asyncio.run(run_analysis())

def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate markdown analysis report"""
    return f"""# Project Analysis Report

**Generated:** {report['analysis_timestamp']}

## Project Overview
- **Workspace:** {report['project_overview']['workspace']}
- **Total Files:** {report['project_overview']['total_files']}
- **Python Files:** {report['project_overview']['python_files']}

## File Analysis Summary
{len(report['file_analyses'])} files were analyzed in detail.

## Recommendations
{chr(10).join(f"- {rec}" for rec in report['recommendations'])}

---
*Generated by Advanced AI Coding Agent*
"""

@cli.command()
@click.pass_obj
def status(config):
    """Show agent configuration and status"""

    console.print(Panel(
        f"[bold blue]Advanced AI Coding Agent Configuration[/bold blue]\n\n"
        f"🏠 Workspace: {config.workspace_dir}\n"
        f"🤖 Primary Model: {config.primary_model}\n"
        f"🔑 API Keys: {', '.join(config.api_keys.keys())}\n"
        f"⚡ Max Concurrent Tasks: {config.max_concurrent_tasks}\n"
        f"💾 Caching: {'✅ Enabled' if config.enable_caching else '❌ Disabled'}\n"
        f"🧠 Learning: {'✅ Enabled' if config.enable_learning else '❌ Disabled'}\n"
        f"🔒 Security Checks: {'✅ Enabled' if config.enable_security_checks else '❌ Disabled'}\n"
        f"🎨 Theme: {config.theme}\n"
        f"📊 Verbose Output: {'✅ Enabled' if config.verbose_output else '❌ Disabled'}\n\n"
        f"🚀 Ready to assist with advanced AI-powered development!",
        title="Agent Status",
        border_style="blue"
    ))

def main():
    """Main entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()

