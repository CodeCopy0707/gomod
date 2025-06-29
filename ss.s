#!/usr/bin/env python3
"""
AI Coding Agent - A comprehensive AI assistant for coding tasks
Features:
- File system operations (read, write, edit, delete, bulk operations)
- Terminal command execution (PowerShell, bash, etc.)
- Codebase indexing and analysis
- Code search across files
- Task planning and management
- Error handling and retry mechanisms
- Test generation and documentation
- Uses Google Gemini 2.0 Flash model
"""

import os
import sys
import json
import subprocess
import shutil
import pathlib
import re
import time
import traceback
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
import fnmatch

# Install required packages if not available
try:
    import google.generativeai as genai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
    import google.generativeai as genai

try:
    import tiktoken
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
    import tiktoken

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.markdown import Markdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.markdown import Markdown

# Initialize Rich console
console = Console()

@dataclass
class Task:
    """Task data structure for planning and tracking"""
    id: str
    title: str
    description: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    priority: int  # 1-5, 1 being highest
    subtasks: List['Task'] = None
    created_at: str = None
    completed_at: str = None
    error_message: str = None
    retry_count: int = 0
    
    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class FileSystemTools:
    """Comprehensive file system operations toolkit"""
    
    @staticmethod
    def read_file(file_path: str, encoding: str = 'utf-8') -> str:
        """Read file content"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    @staticmethod
    def write_file(file_path: str, content: str, encoding: str = 'utf-8', mode: str = 'w') -> bool:
        """Write content to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            raise Exception(f"Error writing file {file_path}: {str(e)}")
    
    @staticmethod
    def append_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """Append content to file"""
        return FileSystemTools.write_file(file_path, content, encoding, 'a')
    
    @staticmethod
    def delete_file(file_path: str) -> bool:
        """Delete a file"""
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            raise Exception(f"Error deleting file {file_path}: {str(e)}")
    
    @staticmethod
    def delete_directory(dir_path: str, recursive: bool = False) -> bool:
        """Delete a directory"""
        try:
            if recursive:
                shutil.rmtree(dir_path)
            else:
                os.rmdir(dir_path)
            return True
        except Exception as e:
            raise Exception(f"Error deleting directory {dir_path}: {str(e)}")
    
    @staticmethod
    def create_directory(dir_path: str) -> bool:
        """Create a directory"""
        try:
            os.makedirs(dir_path, exist_ok=True)
            return True
        except Exception as e:
            raise Exception(f"Error creating directory {dir_path}: {str(e)}")
    
    @staticmethod
    def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> List[str]:
        """List files in directory with optional pattern matching"""
        try:
            files = []
            if recursive:
                for root, dirs, filenames in os.walk(directory):
                    for filename in filenames:
                        if fnmatch.fnmatch(filename, pattern):
                            files.append(os.path.join(root, filename))
            else:
                for filename in os.listdir(directory):
                    if os.path.isfile(os.path.join(directory, filename)) and fnmatch.fnmatch(filename, pattern):
                        files.append(os.path.join(directory, filename))
            return files
        except Exception as e:
            raise Exception(f"Error listing files in {directory}: {str(e)}")
    
    @staticmethod
    def copy_file(src: str, dst: str) -> bool:
        """Copy a file"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            raise Exception(f"Error copying file {src} to {dst}: {str(e)}")
    
    @staticmethod
    def move_file(src: str, dst: str) -> bool:
        """Move a file"""
        try:
            shutil.move(src, dst)
            return True
        except Exception as e:
            raise Exception(f"Error moving file {src} to {dst}: {str(e)}")
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get file information"""
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'is_file': os.path.isfile(file_path),
                'is_directory': os.path.isdir(file_path),
                'permissions': oct(stat.st_mode)[-3:]
            }
        except Exception as e:
            raise Exception(f"Error getting file info for {file_path}: {str(e)}")
    
    @staticmethod
    def str_replace_in_file(file_path: str, old_str: str, new_str: str, count: int = -1) -> bool:
        """Replace string in file"""
        try:
            content = FileSystemTools.read_file(file_path)
            new_content = content.replace(old_str, new_str, count)
            FileSystemTools.write_file(file_path, new_content)
            return True
        except Exception as e:
            raise Exception(f"Error replacing string in file {file_path}: {str(e)}")
    
    @staticmethod
    def insert_at_line(file_path: str, line_number: int, content: str) -> bool:
        """Insert content at specific line number"""
        try:
            lines = FileSystemTools.read_file(file_path).splitlines()
            lines.insert(line_number - 1, content)
            FileSystemTools.write_file(file_path, '\n'.join(lines))
            return True
        except Exception as e:
            raise Exception(f"Error inserting at line {line_number} in file {file_path}: {str(e)}")
    
    @staticmethod
    def bulk_read_files(file_paths: List[str]) -> Dict[str, str]:
        """Read multiple files at once"""
        results = {}
        for file_path in file_paths:
            try:
                results[file_path] = FileSystemTools.read_file(file_path)
            except Exception as e:
                results[file_path] = f"Error: {str(e)}"
        return results
    
    @staticmethod
    def bulk_write_files(files_data: Dict[str, str]) -> Dict[str, bool]:
        """Write multiple files at once"""
        results = {}
        for file_path, content in files_data.items():
            try:
                results[file_path] = FileSystemTools.write_file(file_path, content)
            except Exception as e:
                results[file_path] = False
                console.print(f"[red]Error writing {file_path}: {str(e)}[/red]")
        return results

class TerminalTools:
    """Terminal command execution toolkit"""
    
    @staticmethod
    def execute_command(command: str, cwd: str = None, shell: bool = True, timeout: int = 30) -> Dict[str, Any]:
        """Execute terminal command"""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': command
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'command': command
            }
        except Exception as e:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'command': command
            }
    
    @staticmethod
    def execute_powershell(command: str, cwd: str = None, timeout: int = 30) -> Dict[str, Any]:
        """Execute PowerShell command"""
        ps_command = f"powershell.exe -Command \"{command}\""
        return TerminalTools.execute_command(ps_command, cwd, timeout=timeout)
    
    @staticmethod
    def execute_batch_commands(commands: List[str], cwd: str = None, timeout: int = 30) -> List[Dict[str, Any]]:
        """Execute multiple commands"""
        results = []
        for command in commands:
            result = TerminalTools.execute_command(command, cwd, timeout=timeout)
            results.append(result)
            if not result['success']:
                console.print(f"[red]Command failed: {command}[/red]")
                console.print(f"[red]Error: {result['stderr']}[/red]")
        return results
    
    @staticmethod
    def install_dependencies(requirements: List[str], package_manager: str = 'pip') -> Dict[str, Any]:
        """Install dependencies using specified package manager"""
        if package_manager == 'pip':
            command = f"pip install {' '.join(requirements)}"
        elif package_manager == 'npm':
            command = f"npm install {' '.join(requirements)}"
        elif package_manager == 'yarn':
            command = f"yarn add {' '.join(requirements)}"
        else:
            return {'success': False, 'error': f'Unsupported package manager: {package_manager}'}
        
        return TerminalTools.execute_command(command, timeout=300)  # 5 minutes timeout

class CodebaseIndexer:
    """Codebase indexing and analysis toolkit"""
    
    def __init__(self, root_directory: str):
        self.root_directory = root_directory
        self.index = {}
        self.file_contents = {}
        self.supported_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', 
            '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.r', '.m', '.mm',
            '.html', '.css', '.scss', '.less', '.xml', '.json', '.yaml', '.yml',
            '.md', '.txt', '.sql', '.sh', '.bat', '.ps1', '.dockerfile'
        }
    
    def build_index(self) -> None:
        """Build comprehensive codebase index"""
        console.print("[blue]Building codebase index...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Indexing files...", total=None)
            
            for root, dirs, files in os.walk(self.root_directory):
                # Skip common ignore directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = pathlib.Path(file).suffix.lower()
                    
                    if file_ext in self.supported_extensions:
                        try:
                            content = FileSystemTools.read_file(file_path)
                            self.file_contents[file_path] = content
                            self._index_file_content(file_path, content)
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not index {file_path}: {str(e)}[/yellow]")
        
        console.print(f"[green]Indexed {len(self.file_contents)} files[/green]")
    
    def _index_file_content(self, file_path: str, content: str) -> None:
        """Index content of a single file"""
        # Extract functions, classes, variables, etc.
        if file_path.endswith('.py'):
            self._index_python_file(file_path, content)
        elif file_path.endswith(('.js', '.ts')):
            self._index_javascript_file(file_path, content)
        # Add more language-specific indexing as needed
        
        # Index all words for general search
        words = re.findall(r'\b\w+\b', content.lower())
        for word in words:
            if word not in self.index:
                self.index[word] = []
            if file_path not in self.index[word]:
                self.index[word].append(file_path)
    
    def _index_python_file(self, file_path: str, content: str) -> None:
        """Index Python-specific constructs"""
        # Find classes
        class_matches = re.finditer(r'class\s+(\w+)', content)
        for match in class_matches:
            class_name = match.group(1).lower()
            if class_name not in self.index:
                self.index[class_name] = []
            self.index[class_name].append(file_path)
        
        # Find functions
        func_matches = re.finditer(r'def\s+(\w+)', content)
        for match in func_matches:
            func_name = match.group(1).lower()
            if func_name not in self.index:
                self.index[func_name] = []
            self.index[func_name].append(file_path)
    
    def _index_javascript_file(self, file_path: str, content: str) -> None:
        """Index JavaScript-specific constructs"""
        # Find functions
        func_matches = re.finditer(r'function\s+(\w+)', content)
        for match in func_matches:
            func_name = match.group(1).lower()
            if func_name not in self.index:
                self.index[func_name] = []
            self.index[func_name].append(file_path)
        
        # Find arrow functions and const functions
        arrow_matches = re.finditer(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|\w+\s*=>)', content)
        for match in arrow_matches:
            func_name = match.group(1).lower()
            if func_name not in self.index:
                self.index[func_name] = []
            self.index[func_name].append(file_path)

    def search_code(self, query: str, file_pattern: str = "*") -> Dict[str, List[Dict[str, Any]]]:
        """Search for code patterns across the codebase"""
        results = {}
        query_lower = query.lower()

        for file_path, content in self.file_contents.items():
            if not fnmatch.fnmatch(os.path.basename(file_path), file_pattern):
                continue

            matches = []
            lines = content.splitlines()

            for line_num, line in enumerate(lines, 1):
                if query_lower in line.lower():
                    matches.append({
                        'line_number': line_num,
                        'line_content': line.strip(),
                        'context_before': lines[max(0, line_num-3):line_num-1] if line_num > 1 else [],
                        'context_after': lines[line_num:min(len(lines), line_num+2)] if line_num < len(lines) else []
                    })

            if matches:
                results[file_path] = matches

        return results

    def search_by_regex(self, pattern: str, file_pattern: str = "*") -> Dict[str, List[Dict[str, Any]]]:
        """Search using regular expressions"""
        results = {}
        regex = re.compile(pattern, re.IGNORECASE)

        for file_path, content in self.file_contents.items():
            if not fnmatch.fnmatch(os.path.basename(file_path), file_pattern):
                continue

            matches = []
            lines = content.splitlines()

            for line_num, line in enumerate(lines, 1):
                match = regex.search(line)
                if match:
                    matches.append({
                        'line_number': line_num,
                        'line_content': line.strip(),
                        'match': match.group(),
                        'match_start': match.start(),
                        'match_end': match.end()
                    })

            if matches:
                results[file_path] = matches

        return results

    def find_symbol(self, symbol: str) -> List[str]:
        """Find files containing a specific symbol"""
        symbol_lower = symbol.lower()
        return self.index.get(symbol_lower, [])

    def get_file_summary(self, file_path: str) -> Dict[str, Any]:
        """Get summary information about a file"""
        if file_path not in self.file_contents:
            return {'error': 'File not found in index'}

        content = self.file_contents[file_path]
        lines = content.splitlines()

        summary = {
            'file_path': file_path,
            'line_count': len(lines),
            'character_count': len(content),
            'file_size': len(content.encode('utf-8')),
            'language': self._detect_language(file_path),
            'functions': [],
            'classes': [],
            'imports': []
        }

        # Language-specific analysis
        if file_path.endswith('.py'):
            summary.update(self._analyze_python_file(content))
        elif file_path.endswith(('.js', '.ts')):
            summary.update(self._analyze_javascript_file(content))

        return summary

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = pathlib.Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown'
        }
        return language_map.get(ext, 'unknown')

    def _analyze_python_file(self, content: str) -> Dict[str, List[str]]:
        """Analyze Python file for functions, classes, imports"""
        functions = re.findall(r'def\s+(\w+)', content)
        classes = re.findall(r'class\s+(\w+)', content)
        imports = re.findall(r'(?:from\s+[\w.]+\s+)?import\s+([\w.,\s*]+)', content)

        return {
            'functions': functions,
            'classes': classes,
            'imports': [imp.strip() for imp in imports]
        }

    def _analyze_javascript_file(self, content: str) -> Dict[str, List[str]]:
        """Analyze JavaScript file for functions, classes, imports"""
        functions = re.findall(r'function\s+(\w+)', content)
        arrow_functions = re.findall(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|\w+\s*=>)', content)
        classes = re.findall(r'class\s+(\w+)', content)
        imports = re.findall(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', content)

        return {
            'functions': functions + arrow_functions,
            'classes': classes,
            'imports': imports
        }

class TaskManager:
    """Task planning and management system"""

    def __init__(self):
        self.tasks = []
        self.current_task_id = None

    def create_task(self, title: str, description: str, priority: int = 3) -> str:
        """Create a new task"""
        task_id = f"task_{len(self.tasks) + 1}_{int(time.time())}"
        task = Task(
            id=task_id,
            title=title,
            description=description,
            status='pending',
            priority=priority
        )
        self.tasks.append(task)
        return task_id

    def add_subtask(self, parent_task_id: str, title: str, description: str, priority: int = 3) -> str:
        """Add a subtask to an existing task"""
        parent_task = self.get_task(parent_task_id)
        if not parent_task:
            raise Exception(f"Parent task {parent_task_id} not found")

        subtask_id = f"subtask_{len(parent_task.subtasks) + 1}_{int(time.time())}"
        subtask = Task(
            id=subtask_id,
            title=title,
            description=description,
            status='pending',
            priority=priority
        )
        parent_task.subtasks.append(subtask)
        return subtask_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
            for subtask in task.subtasks:
                if subtask.id == task_id:
                    return subtask
        return None

    def update_task_status(self, task_id: str, status: str, error_message: str = None) -> bool:
        """Update task status"""
        task = self.get_task(task_id)
        if not task:
            return False

        task.status = status
        if status == 'completed':
            task.completed_at = datetime.now().isoformat()
        if error_message:
            task.error_message = error_message
            task.retry_count += 1

        return True

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks sorted by priority"""
        pending = []
        for task in self.tasks:
            if task.status == 'pending':
                pending.append(task)
            for subtask in task.subtasks:
                if subtask.status == 'pending':
                    pending.append(subtask)

        return sorted(pending, key=lambda t: t.priority)

    def get_task_summary(self) -> Dict[str, int]:
        """Get summary of task statuses"""
        summary = {'pending': 0, 'in_progress': 0, 'completed': 0, 'failed': 0}

        for task in self.tasks:
            summary[task.status] += 1
            for subtask in task.subtasks:
                summary[subtask.status] += 1

        return summary

    def export_tasks(self, file_path: str) -> bool:
        """Export tasks to JSON file"""
        try:
            tasks_data = [asdict(task) for task in self.tasks]
            FileSystemTools.write_file(file_path, json.dumps(tasks_data, indent=2))
            return True
        except Exception as e:
            console.print(f"[red]Error exporting tasks: {str(e)}[/red]")
            return False

    def import_tasks(self, file_path: str) -> bool:
        """Import tasks from JSON file"""
        try:
            content = FileSystemTools.read_file(file_path)
            tasks_data = json.loads(content)
            self.tasks = []
            for task_data in tasks_data:
                task = Task(**task_data)
                self.tasks.append(task)
            return True
        except Exception as e:
            console.print(f"[red]Error importing tasks: {str(e)}[/red]")
            return False

class AIAgent:
    """Main AI Coding Agent using Google Gemini 2.0 Flash"""

    def __init__(self, api_key: str, workspace_dir: str = "."):
        """Initialize the AI agent"""
        self.api_key = api_key
        self.workspace_dir = os.path.abspath(workspace_dir)

        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Initialize tools
        self.fs_tools = FileSystemTools()
        self.terminal_tools = TerminalTools()
        self.task_manager = TaskManager()
        self.codebase_indexer = CodebaseIndexer(self.workspace_dir)

        # Build initial codebase index
        self.codebase_indexer.build_index()

        # System prompt
        self.system_prompt = """
You are an advanced AI coding agent with comprehensive capabilities. You can:

1. ANALYZE USER REQUESTS: Understand what the user wants to accomplish
2. PLAN TASKS: Break down complex requests into manageable tasks and subtasks
3. FILE OPERATIONS: Read, write, edit, delete files and directories
4. TERMINAL COMMANDS: Execute PowerShell, bash, and other terminal commands
5. CODEBASE ANALYSIS: Search and analyze code across the entire project
6. ERROR HANDLING: Detect, analyze, and fix errors including linting issues
7. TESTING: Generate and run comprehensive test suites
8. DOCUMENTATION: Create detailed documentation and summaries

WORKFLOW:
1. Analyze the user's request thoroughly
2. If it's a simple task, execute it directly
3. If it's complex, create a detailed plan with tasks and subtasks
4. Execute tasks one by one, handling errors and retrying as needed
5. Fix any linting or IDE errors that appear
6. Generate tests for implemented features
7. Create comprehensive documentation

TOOLS AVAILABLE:
- File system operations (read, write, edit, delete, bulk operations)
- Terminal command execution (any command you need)
- Codebase indexing and search
- Task planning and management
- Error detection and fixing
- Test generation
- Documentation creation

Always be thorough, handle errors gracefully, and provide detailed feedback.
"""

    def analyze_request(self, user_input: str) -> Dict[str, Any]:
        """Analyze user request and determine complexity"""
        prompt = f"""
Analyze this user request and determine:
1. Is this a simple task (can be done in 1-2 steps) or complex (needs planning)?
2. What are the main components/requirements?
3. What tools/technologies are involved?
4. Are there any potential challenges or dependencies?

User request: {user_input}

Respond in JSON format with:
{{
    "complexity": "simple" or "complex",
    "main_components": ["component1", "component2", ...],
    "technologies": ["tech1", "tech2", ...],
    "challenges": ["challenge1", "challenge2", ...],
    "estimated_tasks": number,
    "summary": "brief summary of what needs to be done"
}}
"""

        try:
            response = self.model.generate_content(prompt)
            analysis = json.loads(response.text.strip())
            return analysis
        except Exception as e:
            console.print(f"[red]Error analyzing request: {str(e)}[/red]")
            return {
                "complexity": "simple",
                "main_components": ["unknown"],
                "technologies": ["unknown"],
                "challenges": [],
                "estimated_tasks": 1,
                "summary": user_input
            }

    def create_execution_plan(self, user_input: str, analysis: Dict[str, Any]) -> str:
        """Create detailed execution plan"""
        if analysis["complexity"] == "simple":
            return self.execute_simple_task(user_input)

        prompt = f"""
Create a detailed execution plan for this request:

User request: {user_input}
Analysis: {json.dumps(analysis, indent=2)}

Create a step-by-step plan with:
1. Main tasks (high-level objectives)
2. Subtasks for each main task
3. Dependencies between tasks
4. Estimated complexity for each task
5. Required tools/commands for each task

Format as JSON:
{{
    "plan_summary": "overview of the plan",
    "main_tasks": [
        {{
            "title": "Task title",
            "description": "Detailed description",
            "priority": 1-5,
            "subtasks": [
                {{
                    "title": "Subtask title",
                    "description": "What needs to be done",
                    "tools_needed": ["tool1", "tool2"],
                    "commands": ["command1", "command2"],
                    "estimated_time": "time estimate"
                }}
            ]
        }}
    ]
}}
"""

        try:
            response = self.model.generate_content(prompt)
            plan_data = json.loads(response.text.strip())

            # Create tasks in task manager
            for main_task in plan_data["main_tasks"]:
                task_id = self.task_manager.create_task(
                    main_task["title"],
                    main_task["description"],
                    main_task.get("priority", 3)
                )

                for subtask in main_task.get("subtasks", []):
                    self.task_manager.add_subtask(
                        task_id,
                        subtask["title"],
                        subtask["description"]
                    )

            return plan_data["plan_summary"]

        except Exception as e:
            console.print(f"[red]Error creating plan: {str(e)}[/red]")
            return "Failed to create execution plan"

    def execute_simple_task(self, user_input: str) -> str:
        """Execute a simple task directly"""
        prompt = f"""
Execute this simple task directly. You have access to:
- File operations (read, write, edit, delete files)
- Terminal commands (any command you need)
- Codebase search and analysis

Task: {user_input}

Provide the exact steps to execute and the expected outcome.
If you need to run commands, specify them clearly.
If you need to create/edit files, provide the content.

Current working directory: {self.workspace_dir}
"""

        try:
            response = self.model.generate_content(prompt)
            execution_steps = response.text.strip()

            # Parse and execute the steps
            return self._execute_ai_instructions(execution_steps)

        except Exception as e:
            console.print(f"[red]Error executing simple task: {str(e)}[/red]")
            return f"Failed to execute task: {str(e)}"

    def _execute_ai_instructions(self, instructions: str) -> str:
        """Parse and execute AI-generated instructions"""
        results = []

        # This is a simplified parser - in a real implementation,
        # you'd want more sophisticated parsing
        lines = instructions.split('\n')
        current_command = None
        current_file_content = []
        in_code_block = False

        for line in lines:
            line = line.strip()

            if line.startswith('```'):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                current_file_content.append(line)
                continue

            # Command execution patterns
            if line.startswith('COMMAND:') or line.startswith('RUN:'):
                command = line.split(':', 1)[1].strip()
                result = self.terminal_tools.execute_command(command, cwd=self.workspace_dir)
                results.append(f"Command: {command}")
                results.append(f"Result: {result['stdout'] if result['success'] else result['stderr']}")

            elif line.startswith('CREATE_FILE:') or line.startswith('WRITE_FILE:'):
                file_path = line.split(':', 1)[1].strip()
                if current_file_content:
                    content = '\n'.join(current_file_content)
                    self.fs_tools.write_file(os.path.join(self.workspace_dir, file_path), content)
                    results.append(f"Created file: {file_path}")
                    current_file_content = []

            elif line.startswith('READ_FILE:'):
                file_path = line.split(':', 1)[1].strip()
                try:
                    content = self.fs_tools.read_file(os.path.join(self.workspace_dir, file_path))
                    results.append(f"Read file {file_path}: {len(content)} characters")
                except Exception as e:
                    results.append(f"Error reading {file_path}: {str(e)}")

        return '\n'.join(results)

    def execute_task_plan(self) -> None:
        """Execute all planned tasks"""
        pending_tasks = self.task_manager.get_pending_tasks()

        if not pending_tasks:
            console.print("[yellow]No pending tasks to execute[/yellow]")
            return

        console.print(f"[blue]Executing {len(pending_tasks)} pending tasks...[/blue]")

        for task in pending_tasks:
            console.print(f"\n[cyan]Executing task: {task.title}[/cyan]")
            self.task_manager.update_task_status(task.id, 'in_progress')

            try:
                result = self._execute_single_task(task)
                if result:
                    self.task_manager.update_task_status(task.id, 'completed')
                    console.print(f"[green]✓ Task completed: {task.title}[/green]")
                else:
                    self.task_manager.update_task_status(task.id, 'failed', "Task execution failed")
                    console.print(f"[red]✗ Task failed: {task.title}[/red]")

                    # Retry logic
                    if task.retry_count < 3:
                        console.print(f"[yellow]Retrying task (attempt {task.retry_count + 1}/3)[/yellow]")
                        self.task_manager.update_task_status(task.id, 'pending')

            except Exception as e:
                error_msg = str(e)
                self.task_manager.update_task_status(task.id, 'failed', error_msg)
                console.print(f"[red]✗ Task failed with error: {error_msg}[/red]")

                # Try to fix the error
                if task.retry_count < 3:
                    console.print("[yellow]Attempting to fix error and retry...[/yellow]")
                    fix_result = self._attempt_error_fix(task, error_msg)
                    if fix_result:
                        self.task_manager.update_task_status(task.id, 'pending')

    def _execute_single_task(self, task: Task) -> bool:
        """Execute a single task"""
        prompt = f"""
Execute this specific task:

Title: {task.title}
Description: {task.description}

You have access to:
1. File operations (read, write, edit, delete)
2. Terminal commands (any command needed)
3. Codebase search and analysis

Current working directory: {self.workspace_dir}

Provide step-by-step execution instructions. Be specific about:
- Commands to run
- Files to create/modify
- Content to write
- Expected outcomes

If this task depends on other files or setup, check for them first.
"""

        try:
            response = self.model.generate_content(prompt)
            instructions = response.text.strip()

            # Execute the instructions
            result = self._execute_ai_instructions(instructions)
            console.print(f"[dim]Task execution result: {result}[/dim]")

            return True

        except Exception as e:
            console.print(f"[red]Error executing task: {str(e)}[/red]")
            return False

    def _attempt_error_fix(self, task: Task, error_message: str) -> bool:
        """Attempt to fix an error and retry the task"""
        prompt = f"""
A task failed with this error. Analyze the error and provide a fix:

Task: {task.title}
Description: {task.description}
Error: {error_message}

Working directory: {self.workspace_dir}

Analyze the error and provide:
1. Root cause of the error
2. Steps to fix the error
3. Modified task execution steps

Be specific about what went wrong and how to fix it.
"""

        try:
            response = self.model.generate_content(prompt)
            fix_instructions = response.text.strip()

            console.print(f"[yellow]Applying fix: {fix_instructions[:200]}...[/yellow]")

            # Execute the fix
            fix_result = self._execute_ai_instructions(fix_instructions)
            console.print(f"[dim]Fix result: {fix_result}[/dim]")

            return True

        except Exception as e:
            console.print(f"[red]Error applying fix: {str(e)}[/red]")
            return False

    def check_and_fix_linting_errors(self) -> None:
        """Check for and fix linting errors in the codebase"""
        console.print("[blue]Checking for linting errors...[/blue]")

        # Check Python files with flake8/pylint
        python_files = self.fs_tools.list_files(self.workspace_dir, "*.py", recursive=True)

        for file_path in python_files:
            # Check with flake8
            result = self.terminal_tools.execute_command(f"flake8 {file_path}", cwd=self.workspace_dir)
            if not result['success'] and result['stderr']:
                console.print(f"[yellow]Linting issues in {file_path}[/yellow]")
                self._fix_linting_issues(file_path, result['stderr'])

    def _fix_linting_issues(self, file_path: str, linting_output: str) -> None:
        """Fix linting issues in a specific file"""
        prompt = f"""
Fix the linting issues in this file:

File: {file_path}
Linting output: {linting_output}

Current file content:
{self.fs_tools.read_file(file_path)}

Provide the corrected file content that fixes all linting issues.
Maintain the original functionality while fixing style and syntax issues.
"""

        try:
            response = self.model.generate_content(prompt)
            fixed_content = response.text.strip()

            # Remove code block markers if present
            if fixed_content.startswith('```'):
                lines = fixed_content.split('\n')
                fixed_content = '\n'.join(lines[1:-1])

            self.fs_tools.write_file(file_path, fixed_content)
            console.print(f"[green]Fixed linting issues in {file_path}[/green]")

        except Exception as e:
            console.print(f"[red]Error fixing linting issues in {file_path}: {str(e)}[/red]")

    def generate_tests(self) -> None:
        """Generate comprehensive test suites for the implemented features"""
        console.print("[blue]Generating test suites...[/blue]")

        # Find all Python files that might need tests
        python_files = [f for f in self.fs_tools.list_files(self.workspace_dir, "*.py", recursive=True)
                       if not f.endswith('_test.py') and not f.endswith('test_*.py')]

        for file_path in python_files:
            self._generate_test_for_file(file_path)

    def _generate_test_for_file(self, file_path: str) -> None:
        """Generate tests for a specific file"""
        try:
            content = self.fs_tools.read_file(file_path)
            file_summary = self.codebase_indexer.get_file_summary(file_path)

            prompt = f"""
Generate comprehensive unit tests for this Python file:

File: {file_path}
Content:
{content}

File summary: {json.dumps(file_summary, indent=2)}

Create tests that cover:
1. All functions and methods
2. Edge cases and error conditions
3. Integration scenarios
4. Mock external dependencies

Use pytest framework and follow best practices.
Include docstrings and clear test names.
"""

            response = self.model.generate_content(prompt)
            test_content = response.text.strip()

            # Remove code block markers if present
            if test_content.startswith('```'):
                lines = test_content.split('\n')
                test_content = '\n'.join(lines[1:-1])

            # Create test file
            test_file_path = file_path.replace('.py', '_test.py')
            self.fs_tools.write_file(test_file_path, test_content)
            console.print(f"[green]Generated tests: {test_file_path}[/green]")

        except Exception as e:
            console.print(f"[red]Error generating tests for {file_path}: {str(e)}[/red]")

    def run_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        console.print("[blue]Running test suites...[/blue]")

        # Run pytest
        result = self.terminal_tools.execute_command("python -m pytest -v", cwd=self.workspace_dir)

        test_results = {
            'success': result['success'],
            'output': result['stdout'],
            'errors': result['stderr'],
            'command': result['command']
        }

        if result['success']:
            console.print("[green]✓ All tests passed![/green]")
        else:
            console.print("[red]✗ Some tests failed[/red]")
            console.print(f"[red]Errors: {result['stderr']}[/red]")

        return test_results

    def generate_documentation(self) -> None:
        """Generate comprehensive documentation for the project"""
        console.print("[blue]Generating project documentation...[/blue]")

        # Analyze the entire codebase
        codebase_summary = self._analyze_codebase()

        # Generate README.md
        self._generate_readme(codebase_summary)

        # Generate API documentation
        self._generate_api_docs(codebase_summary)

        # Generate project summary
        self._generate_project_summary(codebase_summary)

    def _analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the entire codebase for documentation"""
        summary = {
            'total_files': len(self.codebase_indexer.file_contents),
            'languages': {},
            'main_modules': [],
            'key_functions': [],
            'dependencies': [],
            'project_structure': {}
        }

        for file_path in self.codebase_indexer.file_contents.keys():
            file_summary = self.codebase_indexer.get_file_summary(file_path)
            language = file_summary.get('language', 'unknown')

            if language not in summary['languages']:
                summary['languages'][language] = 0
            summary['languages'][language] += 1

            # Collect key information
            if file_summary.get('functions'):
                summary['key_functions'].extend(file_summary['functions'])

            if file_summary.get('imports'):
                summary['dependencies'].extend(file_summary['imports'])

        return summary

    def _generate_readme(self, codebase_summary: Dict[str, Any]) -> None:
        """Generate README.md file"""
        prompt = f"""
Generate a comprehensive README.md file for this project:

Codebase Summary: {json.dumps(codebase_summary, indent=2)}

Working directory: {self.workspace_dir}

Include:
1. Project title and description
2. Features and capabilities
3. Installation instructions
4. Usage examples
5. Project structure
6. Dependencies
7. Contributing guidelines
8. License information

Make it professional and informative.
"""

        try:
            response = self.model.generate_content(prompt)
            readme_content = response.text.strip()

            self.fs_tools.write_file(os.path.join(self.workspace_dir, "README.md"), readme_content)
            console.print("[green]Generated README.md[/green]")

        except Exception as e:
            console.print(f"[red]Error generating README: {str(e)}[/red]")

    def _generate_api_docs(self, codebase_summary: Dict[str, Any]) -> None:
        """Generate API documentation"""
        prompt = f"""
Generate API documentation for this project:

Codebase Summary: {json.dumps(codebase_summary, indent=2)}

Create detailed API documentation including:
1. All classes and their methods
2. Function signatures and parameters
3. Return types and descriptions
4. Usage examples
5. Error handling

Format as markdown.
"""

        try:
            response = self.model.generate_content(prompt)
            api_docs = response.text.strip()

            self.fs_tools.write_file(os.path.join(self.workspace_dir, "API_DOCS.md"), api_docs)
            console.print("[green]Generated API_DOCS.md[/green]")

        except Exception as e:
            console.print(f"[red]Error generating API docs: {str(e)}[/red]")

    def _generate_project_summary(self, codebase_summary: Dict[str, Any]) -> None:
        """Generate project summary report"""
        task_summary = self.task_manager.get_task_summary()

        prompt = f"""
Generate a comprehensive project summary report:

Codebase Summary: {json.dumps(codebase_summary, indent=2)}
Task Summary: {json.dumps(task_summary, indent=2)}

Include:
1. Project overview
2. What was implemented
3. Key features and capabilities
4. Technical details
5. Task completion status
6. Challenges faced and solutions
7. Future improvements
8. Performance metrics

Format as a detailed markdown report.
"""

        try:
            response = self.model.generate_content(prompt)
            summary_report = response.text.strip()

            self.fs_tools.write_file(os.path.join(self.workspace_dir, "PROJECT_SUMMARY.md"), summary_report)
            console.print("[green]Generated PROJECT_SUMMARY.md[/green]")

        except Exception as e:
            console.print(f"[red]Error generating project summary: {str(e)}[/red]")

    def process_user_request(self, user_input: str) -> str:
        """Main method to process user requests"""
        console.print(Panel(f"[bold blue]Processing Request:[/bold blue]\n{user_input}",
                          title="AI Coding Agent", border_style="blue"))

        try:
            # Step 1: Analyze the request
            console.print("[cyan]Step 1: Analyzing request...[/cyan]")
            analysis = self.analyze_request(user_input)
            console.print(f"[dim]Analysis: {analysis['summary']}[/dim]")

            # Step 2: Create execution plan
            console.print("[cyan]Step 2: Creating execution plan...[/cyan]")
            plan_summary = self.create_execution_plan(user_input, analysis)
            console.print(f"[dim]Plan: {plan_summary}[/dim]")

            # Step 3: Execute tasks
            console.print("[cyan]Step 3: Executing tasks...[/cyan]")
            self.execute_task_plan()

            # Step 4: Check and fix linting errors
            console.print("[cyan]Step 4: Checking for linting errors...[/cyan]")
            self.check_and_fix_linting_errors()

            # Step 5: Generate tests
            console.print("[cyan]Step 5: Generating tests...[/cyan]")
            self.generate_tests()

            # Step 6: Run tests
            console.print("[cyan]Step 6: Running tests...[/cyan]")
            test_results = self.run_tests()

            # Step 7: Generate documentation
            console.print("[cyan]Step 7: Generating documentation...[/cyan]")
            self.generate_documentation()

            # Final summary
            task_summary = self.task_manager.get_task_summary()

            result_summary = f"""
✅ Request processed successfully!

📊 Task Summary:
- Completed: {task_summary['completed']}
- Failed: {task_summary['failed']}
- Pending: {task_summary['pending']}

🧪 Tests: {'✅ Passed' if test_results['success'] else '❌ Failed'}

📚 Documentation: Generated README.md, API_DOCS.md, PROJECT_SUMMARY.md

🎯 All tasks completed successfully!
"""

            console.print(Panel(result_summary, title="Execution Complete", border_style="green"))
            return result_summary

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
            console.print(Panel(error_msg, title="Error", border_style="red"))
            return error_msg

def main():
    """Main function to run the AI Coding Agent"""
    console.print(Panel(
        "[bold blue]AI Coding Agent[/bold blue]\n"
        "Advanced AI assistant for coding tasks\n"
        "Powered by Google Gemini 2.0 Flash",
        title="Welcome", border_style="blue"
    ))

    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("Enter your Google Gemini API key: ").strip()
        if not api_key:
            console.print("[red]API key is required to run the agent[/red]")
            return

    # Get workspace directory
    workspace = input("Enter workspace directory (default: current directory): ").strip()
    if not workspace:
        workspace = "."

    try:
        # Initialize agent
        console.print("[blue]Initializing AI Coding Agent...[/blue]")
        agent = AIAgent(api_key, workspace)

        console.print("[green]✅ Agent initialized successfully![/green]")
        console.print("\n[yellow]Enter your coding request (or 'quit' to exit):[/yellow]")

        while True:
            user_input = input("\n🤖 > ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("[blue]Goodbye! 👋[/blue]")
                break

            if not user_input:
                continue

            # Process the request
            result = agent.process_user_request(user_input)

            # Ask if user wants to continue
            continue_choice = input("\nDo you want to make another request? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                console.print("[blue]Goodbye! 👋[/blue]")
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Agent interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        console.print(f"[red]{traceback.format_exc()}[/red]")

if __name__ == "__main__":
    main()
