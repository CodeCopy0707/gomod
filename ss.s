

#!/usr/bin/env python3
"""
Augment Agent Replica - A comprehensive single-file implementation
Developed by Augment Code

A complete replica of Augment Agent with all core capabilities:
- File operations and code analysis
- Web search and content fetching
- Process management and terminal integration
- Task management and memory system
- Diagnostic tools and package management
- Google Gemini 2.0 Flash integration

Requirements:
    pip install google-generativeai requests beautifulsoup4 psutil

Setup:
    export GEMINI_API_KEY="your_gemini_api_key"
    export GOOGLE_SEARCH_API_KEY="your_google_search_api_key" (optional)
    export GOOGLE_SEARCH_ENGINE_ID="your_search_engine_id" (optional)

Usage:
    python augment_agent_replica.py
"""

import os
import sys
import json
import uuid
import time
import subprocess
import threading
import re
import shutil
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Third-party imports
try:
    import google.generativeai as genai
    import requests
    from bs4 import BeautifulSoup
    import psutil
    import base64
    import hashlib
    import ast
    import tokenize
    from io import StringIO
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install with: pip install google-generativeai requests beautifulsoup4 psutil")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Task management data structure"""
    id: str
    name: str
    description: str
    state: str  # NOT_STARTED, IN_PROGRESS, COMPLETE, CANCELLED
    parent_id: Optional[str] = None
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

@dataclass
class Memory:
    """Memory storage data structure"""
    id: str
    content: str
    created_at: str
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class ProcessManager:
    """Manages system processes and terminal interactions"""
    
    def __init__(self):
        self.processes: Dict[int, subprocess.Popen] = {}
        self.process_outputs: Dict[int, List[str]] = {}
        self.next_terminal_id = 1
    
    def launch_process(self, command: str, cwd: str, wait: bool = False, max_wait_seconds: int = 600) -> Dict[str, Any]:
        """Launch a new process"""
        try:
            terminal_id = self.next_terminal_id
            self.next_terminal_id += 1
            
            if wait:
                # Synchronous execution
                result = subprocess.run(
                    command, shell=True, cwd=cwd, capture_output=True, 
                    text=True, timeout=max_wait_seconds
                )
                output = result.stdout + result.stderr
                self.process_outputs[terminal_id] = [output]
                return {
                    "terminal_id": terminal_id,
                    "output": output,
                    "return_code": result.returncode,
                    "completed": True
                }
            else:
                # Asynchronous execution
                process = subprocess.Popen(
                    command, shell=True, cwd=cwd, 
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, universal_newlines=True
                )
                self.processes[terminal_id] = process
                self.process_outputs[terminal_id] = []
                
                # Start output capture thread
                threading.Thread(
                    target=self._capture_output, 
                    args=(terminal_id, process), 
                    daemon=True
                ).start()
                
                return {
                    "terminal_id": terminal_id,
                    "output": f"Process started with terminal ID {terminal_id}",
                    "return_code": None,
                    "completed": False
                }
        except Exception as e:
            logger.error(f"Failed to launch process: {e}")
            return {"error": str(e)}
    
    def _capture_output(self, terminal_id: int, process: subprocess.Popen):
        """Capture process output in background thread"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.process_outputs[terminal_id].append(line)
        except Exception as e:
            logger.error(f"Error capturing output for terminal {terminal_id}: {e}")
    
    def read_process(self, terminal_id: int, wait: bool = False, max_wait_seconds: int = 60) -> Dict[str, Any]:
        """Read output from a process"""
        if terminal_id not in self.process_outputs:
            return {"error": f"Terminal {terminal_id} not found"}
        
        if wait and terminal_id in self.processes:
            process = self.processes[terminal_id]
            try:
                process.wait(timeout=max_wait_seconds)
            except subprocess.TimeoutExpired:
                pass
        
        output = ''.join(self.process_outputs[terminal_id])
        completed = terminal_id not in self.processes or self.processes[terminal_id].poll() is not None
        
        return {
            "output": output,
            "completed": completed,
            "return_code": self.processes[terminal_id].poll() if terminal_id in self.processes else None
        }
    
    def write_process(self, terminal_id: int, input_text: str) -> Dict[str, Any]:
        """Write input to a process"""
        if terminal_id not in self.processes:
            return {"error": f"Terminal {terminal_id} not found"}
        
        try:
            process = self.processes[terminal_id]
            process.stdin.write(input_text + '\n')
            process.stdin.flush()
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}
    
    def kill_process(self, terminal_id: int) -> Dict[str, Any]:
        """Kill a process"""
        if terminal_id not in self.processes:
            return {"error": f"Terminal {terminal_id} not found"}
        
        try:
            process = self.processes[terminal_id]
            process.terminate()
            process.wait(timeout=5)
            del self.processes[terminal_id]
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}
    
    def list_processes(self) -> List[Dict[str, Any]]:
        """List all managed processes"""
        result = []
        for terminal_id, process in self.processes.items():
            result.append({
                "terminal_id": terminal_id,
                "pid": process.pid,
                "running": process.poll() is None,
                "return_code": process.poll()
            })
        return result

class WebManager:
    """Handles web search and content fetching"""
    
    def __init__(self):
        self.search_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    def web_search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search the web using Google Custom Search API"""
        if not self.search_api_key or not self.search_engine_id:
            logger.warning("Google Search API not configured, using fallback search")
            return self._fallback_search(query, num_results)
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.search_api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
            
            return results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return self._fallback_search(query, num_results)
    
    def _fallback_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Fallback search method when API is not available"""
        return [{
            'title': f'Search results for: {query}',
            'url': f'https://www.google.com/search?q={query.replace(" ", "+")}',
            'snippet': 'Google Search API not configured. Please set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables.'
        }]
    
    def web_fetch(self, url: str) -> str:
        """Fetch and convert webpage content to markdown"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text and convert to basic markdown
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Basic markdown conversion
            title = soup.find('title')
            if title:
                text = f"# {title.get_text()}\n\n{text}"
            
            return text[:10000]  # Limit content length
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return f"Error fetching content from {url}: {str(e)}"
    
    def open_browser(self, url: str) -> Dict[str, Any]:
        """Open URL in default browser"""
        try:
            webbrowser.open(url)
            return {"success": True, "message": f"Opened {url} in browser"}
        except Exception as e:
            return {"error": str(e)}

class FileManager:
    """Handles file operations and code analysis"""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()

    def view_file(self, path: str, view_range: Optional[List[int]] = None,
                  search_query_regex: Optional[str] = None) -> Dict[str, Any]:
        """View file content with optional range or regex search"""
        try:
            file_path = self.workspace_root / path
            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            if file_path.is_dir():
                return self._list_directory(file_path)

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            if search_query_regex:
                return self._search_in_lines(lines, search_query_regex, path)

            if view_range:
                start, end = view_range
                if end == -1:
                    end = len(lines)
                lines = lines[max(0, start-1):end]

            content = ''.join(f"{i+1:4d}: {line}" for i, line in enumerate(lines))
            return {"content": content, "total_lines": len(lines)}

        except Exception as e:
            return {"error": str(e)}

    def _list_directory(self, dir_path: Path) -> Dict[str, Any]:
        """List directory contents"""
        try:
            items = []
            for item in sorted(dir_path.iterdir()):
                if not item.name.startswith('.'):
                    item_type = "directory" if item.is_dir() else "file"
                    items.append(f"{item_type}: {item.name}")
            return {"content": "\n".join(items), "type": "directory"}
        except Exception as e:
            return {"error": str(e)}

    def _search_in_lines(self, lines: List[str], pattern: str, file_path: str) -> Dict[str, Any]:
        """Search for regex pattern in file lines"""
        try:
            import re
            regex = re.compile(pattern, re.IGNORECASE)
            matches = []

            for i, line in enumerate(lines):
                if regex.search(line):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context = ''.join(f"{j+1:4d}: {lines[j]}" for j in range(start, end))
                    matches.append(f"Match at line {i+1}:\n{context}")

            if matches:
                content = "\n...\n".join(matches)
            else:
                content = f"No matches found for pattern: {pattern}"

            return {"content": content, "matches": len(matches)}
        except Exception as e:
            return {"error": f"Regex search failed: {e}"}

    def save_file(self, path: str, content: str) -> Dict[str, Any]:
        """Save content to a new file"""
        try:
            file_path = self.workspace_root / path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.exists():
                return {"error": f"File already exists: {path}. Use edit_file to modify existing files."}

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return {"success": True, "message": f"File saved: {path}"}
        except Exception as e:
            return {"error": str(e)}

    def edit_file(self, path: str, old_str: str, new_str: str,
                  start_line: int, end_line: int) -> Dict[str, Any]:
        """Edit file by replacing specific content"""
        try:
            file_path = self.workspace_root / path
            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Validate line numbers
            if start_line < 1 or end_line > len(lines) or start_line > end_line:
                return {"error": f"Invalid line range: {start_line}-{end_line}"}

            # Extract the target section
            target_lines = lines[start_line-1:end_line]
            target_content = ''.join(target_lines)

            if old_str not in target_content:
                return {"error": f"Old string not found in specified range"}

            # Replace content
            new_content = target_content.replace(old_str, new_str)
            new_lines = new_content.splitlines(keepends=True)

            # Reconstruct file
            result_lines = lines[:start_line-1] + new_lines + lines[end_line:]

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(result_lines)

            return {"success": True, "message": f"File edited: {path}"}
        except Exception as e:
            return {"error": str(e)}

    def remove_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Remove files safely"""
        try:
            removed = []
            errors = []

            for path in file_paths:
                file_path = self.workspace_root / path
                try:
                    if file_path.exists():
                        if file_path.is_file():
                            file_path.unlink()
                        else:
                            shutil.rmtree(file_path)
                        removed.append(path)
                    else:
                        errors.append(f"File not found: {path}")
                except Exception as e:
                    errors.append(f"Failed to remove {path}: {e}")

            return {"removed": removed, "errors": errors}
        except Exception as e:
            return {"error": str(e)}

    def analyze_code(self, path: str) -> Dict[str, Any]:
        """Analyze code structure and extract symbols"""
        try:
            file_path = self.workspace_root / path
            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Basic code analysis
            analysis = {
                "file_type": file_path.suffix,
                "lines": len(content.splitlines()),
                "size": len(content),
                "functions": [],
                "classes": [],
                "imports": []
            }

            # Python-specific analysis
            if file_path.suffix == '.py':
                analysis.update(self._analyze_python(content))
            elif file_path.suffix in ['.js', '.ts']:
                analysis.update(self._analyze_javascript(content))

            return analysis
        except Exception as e:
            return {"error": str(e)}

    def _analyze_python(self, content: str) -> Dict[str, List[str]]:
        """Analyze Python code"""
        functions = re.findall(r'def\s+(\w+)\s*\(', content)
        classes = re.findall(r'class\s+(\w+)\s*[\(:]', content)
        imports = re.findall(r'(?:from\s+\S+\s+)?import\s+([^\n]+)', content)

        return {
            "functions": functions,
            "classes": classes,
            "imports": [imp.strip() for imp in imports]
        }

    def _analyze_javascript(self, content: str) -> Dict[str, List[str]]:
        """Analyze JavaScript/TypeScript code"""
        functions = re.findall(r'function\s+(\w+)\s*\(', content)
        functions.extend(re.findall(r'(\w+)\s*:\s*function\s*\(', content))
        functions.extend(re.findall(r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', content))

        classes = re.findall(r'class\s+(\w+)', content)
        imports = re.findall(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', content)

        return {
            "functions": list(set(functions)),
            "classes": classes,
            "imports": imports
        }

class TaskManager:
    """Manages task lists and project organization"""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_order: List[str] = []

    def add_tasks(self, tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add new tasks to the task list"""
        try:
            added_tasks = []
            for task_data in tasks_data:
                task_id = str(uuid.uuid4())
                task = Task(
                    id=task_id,
                    name=task_data['name'],
                    description=task_data['description'],
                    state=task_data.get('state', 'NOT_STARTED'),
                    parent_id=task_data.get('parent_task_id')
                )

                self.tasks[task_id] = task

                # Handle insertion position
                if 'after_task_id' in task_data:
                    after_id = task_data['after_task_id']
                    if after_id in self.task_order:
                        idx = self.task_order.index(after_id) + 1
                        self.task_order.insert(idx, task_id)
                    else:
                        self.task_order.append(task_id)
                else:
                    self.task_order.append(task_id)

                added_tasks.append(task_id)

            return {"success": True, "added_tasks": added_tasks}
        except Exception as e:
            return {"error": str(e)}

    def update_tasks(self, tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update existing tasks"""
        try:
            updated_tasks = []
            for task_data in tasks_data:
                task_id = task_data['task_id']
                if task_id not in self.tasks:
                    continue

                task = self.tasks[task_id]
                if 'name' in task_data:
                    task.name = task_data['name']
                if 'description' in task_data:
                    task.description = task_data['description']
                if 'state' in task_data:
                    task.state = task_data['state']

                updated_tasks.append(task_id)

            return {"success": True, "updated_tasks": updated_tasks}
        except Exception as e:
            return {"error": str(e)}

    def view_tasklist(self) -> str:
        """Generate markdown representation of task list"""
        if not self.tasks:
            return "No tasks currently defined."

        lines = ["# Task List\n"]

        # Group tasks by hierarchy
        root_tasks = [tid for tid in self.task_order if self.tasks[tid].parent_id is None]

        for task_id in root_tasks:
            lines.extend(self._format_task_tree(task_id, 0))

        return "\n".join(lines)

    def _format_task_tree(self, task_id: str, indent_level: int) -> List[str]:
        """Format task and its subtasks recursively"""
        task = self.tasks[task_id]
        indent = "  " * indent_level

        # Format state symbol
        state_symbols = {
            'NOT_STARTED': '[ ]',
            'IN_PROGRESS': '[/]',
            'COMPLETE': '[x]',
            'CANCELLED': '[-]'
        }
        symbol = state_symbols.get(task.state, '[ ]')

        lines = [f"{indent}- {symbol} **{task.name}** ({task.id})"]
        if task.description:
            lines.append(f"{indent}  {task.description}")

        # Add subtasks
        subtasks = [tid for tid in self.task_order
                   if tid in self.tasks and self.tasks[tid].parent_id == task_id]
        for subtask_id in subtasks:
            lines.extend(self._format_task_tree(subtask_id, indent_level + 1))

        return lines

    def reorganize_tasklist(self, markdown: str) -> Dict[str, Any]:
        """Reorganize task list from markdown representation"""
        try:
            # Parse markdown and rebuild task structure
            # This is a simplified implementation
            lines = markdown.strip().split('\n')
            new_tasks = {}
            new_order = []

            for line in lines:
                if line.strip().startswith('- ['):
                    # Extract task info from markdown
                    match = re.search(r'- \[(.)\] \*\*(.+?)\*\* \((.+?)\)', line)
                    if match:
                        state_char, name, task_id = match.groups()
                        state_map = {' ': 'NOT_STARTED', '/': 'IN_PROGRESS',
                                   'x': 'COMPLETE', '-': 'CANCELLED'}
                        state = state_map.get(state_char, 'NOT_STARTED')

                        if task_id in self.tasks:
                            task = self.tasks[task_id]
                            task.name = name
                            task.state = state
                            new_tasks[task_id] = task
                            new_order.append(task_id)

            self.tasks = new_tasks
            self.task_order = new_order
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

class MemoryManager:
    """Manages long-term memory storage"""

    def __init__(self):
        self.memories: Dict[str, Memory] = {}
        self.memory_file = Path("agent_memories.json")
        self.load_memories()

    def remember(self, content: str, tags: List[str] = None) -> Dict[str, Any]:
        """Store a new memory"""
        try:
            memory_id = str(uuid.uuid4())
            memory = Memory(
                id=memory_id,
                content=content,
                created_at=datetime.now().isoformat(),
                tags=tags or []
            )

            self.memories[memory_id] = memory
            self.save_memories()

            return {"success": True, "memory_id": memory_id}
        except Exception as e:
            return {"error": str(e)}

    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """Search memories by content or tags"""
        results = []
        query_lower = query.lower()

        for memory in self.memories.values():
            if (query_lower in memory.content.lower() or
                any(query_lower in tag.lower() for tag in memory.tags)):
                results.append({
                    "id": memory.id,
                    "content": memory.content,
                    "created_at": memory.created_at,
                    "tags": memory.tags
                })

        return sorted(results, key=lambda x: x['created_at'], reverse=True)

    def save_memories(self):
        """Save memories to file"""
        try:
            data = {mid: asdict(memory) for mid, memory in self.memories.items()}
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")

    def load_memories(self):
        """Load memories from file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)

                for mid, memory_data in data.items():
                    self.memories[mid] = Memory(**memory_data)
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")

class DiagnosticManager:
    """Provides code diagnostics and debugging support"""

    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

    def get_diagnostics(self, file_paths: List[str]) -> Dict[str, Any]:
        """Get diagnostic information for files"""
        diagnostics = {}

        for path in file_paths:
            try:
                file_path = self.file_manager.workspace_root / path
                if not file_path.exists():
                    diagnostics[path] = {"error": "File not found"}
                    continue

                issues = []

                # Basic syntax checking
                if file_path.suffix == '.py':
                    issues.extend(self._check_python_syntax(file_path))
                elif file_path.suffix in ['.js', '.ts']:
                    issues.extend(self._check_javascript_syntax(file_path))

                diagnostics[path] = {"issues": issues}

            except Exception as e:
                diagnostics[path] = {"error": str(e)}

        return diagnostics

    def _check_python_syntax(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check Python syntax"""
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            compile(content, str(file_path), 'exec')
        except SyntaxError as e:
            issues.append({
                "type": "error",
                "line": e.lineno,
                "message": f"Syntax error: {e.msg}",
                "severity": "error"
            })
        except Exception as e:
            issues.append({
                "type": "error",
                "message": f"Compilation error: {str(e)}",
                "severity": "error"
            })

        return issues

    def _check_javascript_syntax(self, file_path: Path) -> List[Dict[str, Any]]:
        """Basic JavaScript syntax checking"""
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Basic checks for common issues
            if content.count('(') != content.count(')'):
                issues.append({
                    "type": "warning",
                    "message": "Mismatched parentheses",
                    "severity": "warning"
                })

            if content.count('{') != content.count('}'):
                issues.append({
                    "type": "warning",
                    "message": "Mismatched braces",
                    "severity": "warning"
                })

        except Exception as e:
            issues.append({
                "type": "error",
                "message": f"Analysis error: {str(e)}",
                "severity": "error"
            })

        return issues

class PackageManager:
    """Handles package management across different languages"""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)

    def detect_package_manager(self) -> Dict[str, str]:
        """Detect available package managers in the workspace"""
        managers = {}

        # Python
        if (self.workspace_root / "requirements.txt").exists():
            managers["python"] = "pip"
        elif (self.workspace_root / "pyproject.toml").exists():
            managers["python"] = "poetry"
        elif (self.workspace_root / "Pipfile").exists():
            managers["python"] = "pipenv"

        # Node.js
        if (self.workspace_root / "package.json").exists():
            if (self.workspace_root / "yarn.lock").exists():
                managers["node"] = "yarn"
            elif (self.workspace_root / "pnpm-lock.yaml").exists():
                managers["node"] = "pnpm"
            else:
                managers["node"] = "npm"

        # Rust
        if (self.workspace_root / "Cargo.toml").exists():
            managers["rust"] = "cargo"

        # Go
        if (self.workspace_root / "go.mod").exists():
            managers["go"] = "go"

        return managers

    def install_package(self, package: str, language: str = None) -> Dict[str, Any]:
        """Install a package using appropriate package manager"""
        managers = self.detect_package_manager()

        if language and language in managers:
            manager = managers[language]
        elif len(managers) == 1:
            manager = list(managers.values())[0]
        else:
            return {"error": "Could not determine package manager. Please specify language."}

        commands = {
            "pip": f"pip install {package}",
            "poetry": f"poetry add {package}",
            "pipenv": f"pipenv install {package}",
            "npm": f"npm install {package}",
            "yarn": f"yarn add {package}",
            "pnpm": f"pnpm add {package}",
            "cargo": f"cargo add {package}",
            "go": f"go get {package}"
        }

        if manager not in commands:
            return {"error": f"Unsupported package manager: {manager}"}

        return {"command": commands[manager], "manager": manager}

class MermaidRenderer:
    """Handles Mermaid diagram rendering and visualization"""

    def __init__(self):
        self.mermaid_cdn = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"

    def render_mermaid(self, diagram_definition: str, title: str = "Mermaid Diagram") -> Dict[str, Any]:
        """Render a Mermaid diagram to HTML and optionally open in browser"""
        try:
            # Validate diagram definition
            if not diagram_definition.strip():
                return {"error": "Empty diagram definition"}

            # Generate HTML with Mermaid
            html_content = self._generate_mermaid_html(diagram_definition, title)

            # Save to temporary file
            temp_file = Path("temp_mermaid_diagram.html")
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Open in browser
            file_url = f"file://{temp_file.absolute()}"
            webbrowser.open(file_url)

            return {
                "success": True,
                "message": f"Mermaid diagram '{title}' rendered and opened in browser",
                "file_path": str(temp_file),
                "url": file_url
            }

        except Exception as e:
            logger.error(f"Failed to render Mermaid diagram: {e}")
            return {"error": str(e)}

    def _generate_mermaid_html(self, diagram_definition: str, title: str) -> str:
        """Generate HTML with embedded Mermaid diagram"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="{self.mermaid_cdn}"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #333;
            margin: 0;
        }}
        .diagram-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        .btn {{
            background: #007acc;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
            font-size: 14px;
        }}
        .btn:hover {{
            background: #005a9e;
        }}
        #mermaid-diagram {{
            margin: 20px auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>Generated by Augment Agent Replica</p>
        </div>

        <div class="controls">
            <button class="btn" onclick="zoomIn()">Zoom In</button>
            <button class="btn" onclick="zoomOut()">Zoom Out</button>
            <button class="btn" onclick="resetZoom()">Reset Zoom</button>
            <button class="btn" onclick="copyDiagram()">Copy Definition</button>
        </div>

        <div class="diagram-container">
            <div id="mermaid-diagram">
                <pre class="mermaid">
{diagram_definition}
                </pre>
            </div>
        </div>
    </div>

    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true
            }}
        }});

        let currentZoom = 1;
        const diagram = document.getElementById('mermaid-diagram');

        function zoomIn() {{
            currentZoom += 0.1;
            diagram.style.transform = `scale(${{currentZoom}})`;
        }}

        function zoomOut() {{
            currentZoom = Math.max(0.1, currentZoom - 0.1);
            diagram.style.transform = `scale(${{currentZoom}})`;
        }}

        function resetZoom() {{
            currentZoom = 1;
            diagram.style.transform = 'scale(1)';
        }}

        function copyDiagram() {{
            const definition = `{diagram_definition}`;
            navigator.clipboard.writeText(definition).then(() => {{
                alert('Diagram definition copied to clipboard!');
            }});
        }}
    </script>
</body>
</html>"""

class CodebaseAnalyzer:
    """Advanced codebase analysis and context retrieval"""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.file_cache = {}
        self.symbol_index = {}
        self.dependency_graph = {}

    def codebase_retrieval(self, information_request: str) -> Dict[str, Any]:
        """Retrieve relevant code snippets based on natural language request"""
        try:
            # Parse the request to understand what's needed
            request_lower = information_request.lower()

            # Build or update index if needed
            self._build_symbol_index()

            # Search for relevant code
            relevant_files = []
            relevant_symbols = []

            # Keyword-based search
            keywords = self._extract_keywords(information_request)

            for keyword in keywords:
                # Search in symbol names
                matching_symbols = self._search_symbols(keyword)
                relevant_symbols.extend(matching_symbols)

                # Search in file contents
                matching_files = self._search_file_contents(keyword)
                relevant_files.extend(matching_files)

            # Rank and return results
            results = self._rank_results(relevant_files, relevant_symbols, information_request)

            return {
                "success": True,
                "results": results,
                "total_files": len(set(relevant_files)),
                "total_symbols": len(set(relevant_symbols))
            }

        except Exception as e:
            logger.error(f"Codebase retrieval failed: {e}")
            return {"error": str(e)}

    def _build_symbol_index(self):
        """Build an index of all symbols in the codebase"""
        try:
            for file_path in self._get_code_files():
                if file_path not in self.file_cache:
                    self._index_file(file_path)
        except Exception as e:
            logger.error(f"Failed to build symbol index: {e}")

    def _get_code_files(self) -> List[Path]:
        """Get all code files in the workspace"""
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.php', '.rb'}
        code_files = []

        for file_path in self.workspace_root.rglob('*'):
            if (file_path.is_file() and
                file_path.suffix in code_extensions and
                not any(part.startswith('.') for part in file_path.parts)):
                code_files.append(file_path)

        return code_files

    def _index_file(self, file_path: Path):
        """Index symbols in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            self.file_cache[file_path] = content

            # Extract symbols based on file type
            if file_path.suffix == '.py':
                symbols = self._extract_python_symbols(content, file_path)
            elif file_path.suffix in ['.js', '.ts']:
                symbols = self._extract_javascript_symbols(content, file_path)
            else:
                symbols = self._extract_generic_symbols(content, file_path)

            self.symbol_index[file_path] = symbols

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")

    def _extract_python_symbols(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract Python symbols using AST"""
        symbols = []
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.append({
                        'type': 'function',
                        'name': node.name,
                        'line': node.lineno,
                        'file': str(file_path),
                        'args': [arg.arg for arg in node.args.args] if hasattr(node.args, 'args') else []
                    })
                elif isinstance(node, ast.ClassDef):
                    symbols.append({
                        'type': 'class',
                        'name': node.name,
                        'line': node.lineno,
                        'file': str(file_path),
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        symbols.append({
                            'type': 'import',
                            'name': alias.name,
                            'line': node.lineno,
                            'file': str(file_path)
                        })
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        symbols.append({
                            'type': 'import_from',
                            'module': node.module,
                            'names': [alias.name for alias in node.names],
                            'line': node.lineno,
                            'file': str(file_path)
                        })
        except SyntaxError:
            # Fallback to regex-based extraction
            symbols = self._extract_generic_symbols(content, file_path)

        return symbols

    def _extract_javascript_symbols(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript symbols using regex"""
        symbols = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Function declarations
            func_match = re.search(r'function\s+(\w+)\s*\(([^)]*)\)', line)
            if func_match:
                symbols.append({
                    'type': 'function',
                    'name': func_match.group(1),
                    'line': i,
                    'file': str(file_path),
                    'args': [arg.strip() for arg in func_match.group(2).split(',') if arg.strip()]
                })

            # Arrow functions
            arrow_match = re.search(r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', line)
            if arrow_match:
                symbols.append({
                    'type': 'function',
                    'name': arrow_match.group(1),
                    'line': i,
                    'file': str(file_path)
                })

            # Class declarations
            class_match = re.search(r'class\s+(\w+)', line)
            if class_match:
                symbols.append({
                    'type': 'class',
                    'name': class_match.group(1),
                    'line': i,
                    'file': str(file_path)
                })

            # Imports
            import_match = re.search(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', line)
            if import_match:
                symbols.append({
                    'type': 'import',
                    'module': import_match.group(1),
                    'line': i,
                    'file': str(file_path)
                })

        return symbols

    def _extract_generic_symbols(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Generic symbol extraction using regex patterns"""
        symbols = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Generic function patterns
            func_patterns = [
                r'def\s+(\w+)\s*\(',  # Python
                r'function\s+(\w+)\s*\(',  # JavaScript
                r'(\w+)\s*\([^)]*\)\s*{',  # C-style
                r'public\s+\w+\s+(\w+)\s*\(',  # Java/C#
            ]

            for pattern in func_patterns:
                match = re.search(pattern, line)
                if match:
                    symbols.append({
                        'type': 'function',
                        'name': match.group(1),
                        'line': i,
                        'file': str(file_path)
                    })
                    break

        return symbols

    def _extract_keywords(self, request: str) -> List[str]:
        """Extract relevant keywords from information request"""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'about', 'how', 'what', 'where', 'when', 'why', 'which', 'who', 'find', 'show', 'get', 'information', 'code', 'function', 'class', 'method'}

        words = re.findall(r'\b\w+\b', request.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords

    def _search_symbols(self, keyword: str) -> List[Dict[str, Any]]:
        """Search for symbols matching keyword"""
        matching_symbols = []

        for file_path, symbols in self.symbol_index.items():
            for symbol in symbols:
                if keyword.lower() in symbol['name'].lower():
                    matching_symbols.append(symbol)

        return matching_symbols

    def _search_file_contents(self, keyword: str) -> List[str]:
        """Search for keyword in file contents"""
        matching_files = []

        for file_path, content in self.file_cache.items():
            if keyword.lower() in content.lower():
                matching_files.append(str(file_path))

        return matching_files

    def _rank_results(self, files: List[str], symbols: List[Dict[str, Any]], request: str) -> List[Dict[str, Any]]:
        """Rank and format results based on relevance"""
        results = []

        # Add symbol results
        for symbol in symbols[:10]:  # Limit to top 10
            results.append({
                'type': 'symbol',
                'name': symbol['name'],
                'symbol_type': symbol['type'],
                'file': symbol['file'],
                'line': symbol.get('line', 0),
                'relevance': self._calculate_relevance(symbol['name'], request)
            })

        # Add file results
        unique_files = list(set(files))[:5]  # Limit to top 5 files
        for file_path in unique_files:
            results.append({
                'type': 'file',
                'file': file_path,
                'relevance': self._calculate_relevance(Path(file_path).name, request)
            })

        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)

        return results

    def _calculate_relevance(self, text: str, request: str) -> float:
        """Calculate relevance score between text and request"""
        text_lower = text.lower()
        request_lower = request.lower()

        # Simple relevance scoring
        score = 0.0

        # Exact match bonus
        if text_lower in request_lower or request_lower in text_lower:
            score += 1.0

        # Word overlap
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        request_words = set(re.findall(r'\b\w+\b', request_lower))

        if text_words and request_words:
            overlap = len(text_words.intersection(request_words))
            score += overlap / len(text_words.union(request_words))

        return score

class AdvancedFileManager(FileManager):
    """Enhanced file manager with advanced features"""

    def __init__(self, workspace_root: str):
        super().__init__(workspace_root)
        self.truncation_cache = {}
        self.next_reference_id = 1

    def view_file_advanced(self, path: str, view_range: Optional[List[int]] = None,
                          search_query_regex: Optional[str] = None,
                          case_sensitive: bool = False,
                          context_lines_before: int = 5,
                          context_lines_after: int = 5) -> Dict[str, Any]:
        """Advanced file viewing with regex search and range support"""
        try:
            file_path = self.workspace_root / path
            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            if file_path.is_dir():
                return self._list_directory(file_path)

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            total_lines = len(lines)

            # Apply view range if specified
            if view_range:
                start, end = view_range
                if end == -1:
                    end = total_lines
                start = max(1, start)
                end = min(total_lines, end)
                lines = lines[start-1:end]
                line_offset = start - 1
            else:
                line_offset = 0

            # Apply regex search if specified
            if search_query_regex:
                return self._search_with_context(lines, search_query_regex, path,
                                               case_sensitive, context_lines_before,
                                               context_lines_after, line_offset)

            # Format content with line numbers
            content = ''.join(f"{i+1+line_offset:4d}: {line}" for i, line in enumerate(lines))

            # Handle truncation for large files
            if len(content) > 50000:  # 50KB limit
                reference_id = f"ref_{self.next_reference_id}"
                self.next_reference_id += 1
                self.truncation_cache[reference_id] = {
                    'content': content,
                    'path': path,
                    'total_lines': total_lines
                }

                truncated_content = content[:50000]
                truncated_content += f"\n\n<response clipped>\nContent truncated. Reference ID: {reference_id}\nTotal lines: {total_lines}\nUse view-range-untruncated or search-untruncated tools to see more."

                return {
                    "content": truncated_content,
                    "total_lines": total_lines,
                    "truncated": True,
                    "reference_id": reference_id
                }

            return {"content": content, "total_lines": total_lines}

        except Exception as e:
            return {"error": str(e)}

    def _search_with_context(self, lines: List[str], pattern: str, file_path: str,
                           case_sensitive: bool, context_before: int, context_after: int,
                           line_offset: int = 0) -> Dict[str, Any]:
        """Search with context lines around matches"""
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)

            matches = []
            match_lines = set()

            # Find all matching lines
            for i, line in enumerate(lines):
                if regex.search(line):
                    match_lines.add(i)

            if not match_lines:
                return {
                    "content": f"No matches found for pattern: {pattern}",
                    "matches": 0,
                    "pattern": pattern
                }

            # Build context blocks
            context_blocks = []
            processed_lines = set()

            for match_line in sorted(match_lines):
                if match_line in processed_lines:
                    continue

                # Determine context range
                start = max(0, match_line - context_before)
                end = min(len(lines), match_line + context_after + 1)

                # Collect context lines
                context_lines = []
                for i in range(start, end):
                    line_num = i + 1 + line_offset
                    marker = ">>> " if i == match_line else "    "
                    context_lines.append(f"{marker}{line_num:4d}: {lines[i]}")
                    processed_lines.add(i)

                context_blocks.append(''.join(context_lines))

            content = "\n...\n".join(context_blocks)

            return {
                "content": content,
                "matches": len(match_lines),
                "pattern": pattern,
                "file": file_path
            }

        except re.error as e:
            return {"error": f"Invalid regex pattern: {e}"}
        except Exception as e:
            return {"error": str(e)}

    def view_range_untruncated(self, reference_id: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """View specific range from truncated content"""
        if reference_id not in self.truncation_cache:
            return {"error": f"Reference ID {reference_id} not found"}

        try:
            cached_data = self.truncation_cache[reference_id]
            full_content = cached_data['content']

            # Split into lines and extract range
            lines = full_content.split('\n')

            if start_line < 1 or end_line > len(lines) or start_line > end_line:
                return {"error": f"Invalid line range: {start_line}-{end_line}"}

            selected_lines = lines[start_line-1:end_line]
            content = '\n'.join(selected_lines)

            return {
                "content": content,
                "range": f"{start_line}-{end_line}",
                "total_lines": cached_data['total_lines']
            }

        except Exception as e:
            return {"error": str(e)}

    def search_untruncated(self, reference_id: str, search_term: str, context_lines: int = 2) -> Dict[str, Any]:
        """Search within truncated content"""
        if reference_id not in self.truncation_cache:
            return {"error": f"Reference ID {reference_id} not found"}

        try:
            cached_data = self.truncation_cache[reference_id]
            full_content = cached_data['content']
            lines = full_content.split('\n')

            matches = []
            for i, line in enumerate(lines):
                if search_term.lower() in line.lower():
                    # Get context
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)

                    context_block = []
                    for j in range(start, end):
                        marker = ">>> " if j == i else "    "
                        context_block.append(f"{marker}{j+1:4d}: {lines[j]}")

                    matches.append('\n'.join(context_block))

            if matches:
                content = "\n...\n".join(matches)
            else:
                content = f"No matches found for: {search_term}"

            return {
                "content": content,
                "matches": len(matches),
                "search_term": search_term
            }

        except Exception as e:
            return {"error": str(e)}

    def insert_content(self, path: str, insert_line: int, new_content: str) -> Dict[str, Any]:
        """Insert content at specific line"""
        try:
            file_path = self.workspace_root / path
            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if insert_line < 0 or insert_line > len(lines):
                return {"error": f"Invalid insert line: {insert_line}"}

            # Insert new content
            new_lines = new_content.splitlines(keepends=True)
            if new_lines and not new_lines[-1].endswith('\n'):
                new_lines[-1] += '\n'

            result_lines = lines[:insert_line] + new_lines + lines[insert_line:]

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(result_lines)

            return {
                "success": True,
                "message": f"Content inserted at line {insert_line} in {path}",
                "lines_added": len(new_lines)
            }

        except Exception as e:
            return {"error": str(e)}

class AdvancedProcessManager(ProcessManager):
    """Enhanced process manager with terminal reading capabilities"""

    def __init__(self):
        super().__init__()
        self.terminal_history = {}

    def read_terminal(self, only_selected: bool = False) -> Dict[str, Any]:
        """Read from the active terminal"""
        try:
            # In a real implementation, this would interface with the actual terminal
            # For this replica, we'll simulate terminal reading from the most recent process

            if not self.processes:
                return {"content": "No active terminal sessions"}

            # Get the most recent terminal
            latest_terminal_id = max(self.processes.keys())

            if only_selected:
                # Simulate selected text (in real implementation, would get actual selection)
                return {"content": "Selected text simulation - not implemented in replica"}

            # Return full terminal output
            output = ''.join(self.process_outputs.get(latest_terminal_id, []))

            return {
                "content": output,
                "terminal_id": latest_terminal_id,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e)}

    def get_terminal_history(self, terminal_id: int) -> Dict[str, Any]:
        """Get complete history for a terminal session"""
        if terminal_id not in self.process_outputs:
            return {"error": f"Terminal {terminal_id} not found"}

        history = self.process_outputs[terminal_id]
        return {
            "history": history,
            "total_lines": len(history),
            "terminal_id": terminal_id
        }

class StructuredToolCallParser:
    """Advanced tool call parsing and execution"""

    def __init__(self, agent):
        self.agent = agent
        self.tool_registry = self._build_tool_registry()

    def _build_tool_registry(self) -> Dict[str, callable]:
        """Build registry of available tools"""
        return {
            # File operations
            'view_file': self.agent.view_file,
            'save_file': self.agent.save_file,
            'edit_file': self.agent.edit_file,
            'remove_files': self.agent.file_manager.remove_files,
            'analyze_code': self.agent.file_manager.analyze_code,

            # Web operations
            'web_search': self.agent.web_search,
            'web_fetch': self.agent.web_fetch,
            'open_browser': self.agent.web_manager.open_browser,

            # Process operations
            'launch_process': self.agent.launch_process,
            'read_process': self.agent.process_manager.read_process,
            'write_process': self.agent.process_manager.write_process,
            'kill_process': self.agent.process_manager.kill_process,
            'list_processes': self.agent.process_manager.list_processes,
            'read_terminal': getattr(self.agent.process_manager, 'read_terminal', None),

            # Task management
            'add_tasks': self.agent.add_tasks,
            'update_tasks': self.agent.update_tasks,
            'view_tasklist': self.agent.view_tasklist,
            'reorganize_tasklist': self.agent.task_manager.reorganize_tasklist,

            # Memory operations
            'remember': self.agent.remember,
            'search_memories': self.agent.search_memories,

            # Advanced features
            'render_mermaid': getattr(self.agent, 'render_mermaid', None),
            'codebase_retrieval': getattr(self.agent, 'codebase_retrieval', None),
            'get_diagnostics': self.agent.get_diagnostics,
            'install_package': self.agent.install_package,
        }

    def parse_and_execute_tools(self, response_text: str, user_input: str) -> str:
        """Parse structured tool calls from response and execute them"""
        try:
            # Look for structured tool calls in the response
            tool_calls = self._extract_tool_calls(response_text)

            if not tool_calls:
                # Fallback to pattern-based parsing
                return self._pattern_based_execution(response_text, user_input)

            # Execute structured tool calls
            results = []
            for tool_call in tool_calls:
                result = self._execute_tool_call(tool_call)
                results.append(result)

            # Format results
            if results:
                formatted_results = self._format_tool_results(results)
                return f"{response_text}\n\n{formatted_results}"

            return response_text

        except Exception as e:
            logger.error(f"Tool call execution failed: {e}")
            return f"{response_text}\n\nTool execution error: {str(e)}"

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured tool calls from text"""
        tool_calls = []

        # Look for XML-style tool calls
        pattern = r'<tool_call name="([^"]+)"(?:\s+([^>]*))?>([^<]*)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            tool_name, params_str, content = match

            # Parse parameters
            params = {}
            if params_str:
                param_matches = re.findall(r'(\w+)="([^"]*)"', params_str)
                params.update(param_matches)

            if content.strip():
                params['content'] = content.strip()

            tool_calls.append({
                'name': tool_name,
                'parameters': params
            })

        return tool_calls

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call"""
        tool_name = tool_call['name']
        params = tool_call['parameters']

        if tool_name not in self.tool_registry:
            return {"error": f"Unknown tool: {tool_name}"}

        tool_func = self.tool_registry[tool_name]
        if tool_func is None:
            return {"error": f"Tool {tool_name} not available"}

        try:
            # Execute the tool with parameters
            result = tool_func(**params)
            return {"tool": tool_name, "result": result, "success": True}
        except Exception as e:
            return {"tool": tool_name, "error": str(e), "success": False}

    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """Format tool execution results"""
        formatted = ["Tool Execution Results:"]

        for i, result in enumerate(results, 1):
            tool_name = result.get('tool', 'unknown')

            if result.get('success'):
                formatted.append(f"\n{i}. {tool_name}: ✅ Success")
                if 'result' in result and result['result']:
                    # Format result based on type
                    res = result['result']
                    if isinstance(res, dict):
                        if 'content' in res:
                            formatted.append(f"   Content: {res['content'][:200]}...")
                        elif 'message' in res:
                            formatted.append(f"   {res['message']}")
                        else:
                            formatted.append(f"   {res}")
                    else:
                        formatted.append(f"   {str(res)[:200]}...")
            else:
                formatted.append(f"\n{i}. {tool_name}: ❌ Error")
                formatted.append(f"   {result.get('error', 'Unknown error')}")

        return '\n'.join(formatted)

    def _pattern_based_execution(self, response_text: str, user_input: str) -> str:
        """Fallback pattern-based tool execution"""
        # Enhanced pattern matching for common requests

        # File viewing patterns
        file_patterns = [
            r'(?:show|view|display|read)\s+(?:me\s+)?(?:the\s+)?(?:contents?\s+of\s+)?["\']?([^"\']+\.[a-zA-Z]+)["\']?',
            r'(?:open|check)\s+["\']?([^"\']+\.[a-zA-Z]+)["\']?',
            r'["\']([^"\']+\.[a-zA-Z]+)["\']'
        ]

        for pattern in file_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                file_path = match.group(1)
                result = self.agent.view_file(file_path)
                if 'content' in result:
                    return f"{response_text}\n\n**File: {file_path}**\n```\n{result['content'][:1000]}...\n```"

        # Web search patterns
        search_patterns = [
            r'search\s+(?:the\s+)?web\s+for\s+["\']?([^"\']+)["\']?',
            r'look\s+up\s+["\']?([^"\']+)["\']?',
            r'find\s+information\s+about\s+["\']?([^"\']+)["\']?'
        ]

        for pattern in search_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                query = match.group(1)
                results = self.agent.web_search(query, 3)
                if results:
                    search_results = "\n".join([f"• {r['title']}: {r['url']}" for r in results])
                    return f"{response_text}\n\n**Web Search Results for '{query}':**\n{search_results}"

        # Task management patterns
        if re.search(r'(?:create|add|make)\s+(?:a\s+)?task', user_input, re.IGNORECASE):
            task_match = re.search(r'task\s+(?:to\s+|for\s+)?["\']?([^"\']+)["\']?', user_input, re.IGNORECASE)
            if task_match:
                task_desc = task_match.group(1)
                result = self.agent.add_tasks([{"name": task_desc, "description": f"Task created from: {user_input}"}])
                return f"{response_text}\n\n**Task Created:** {task_desc}"

        if re.search(r'(?:show|view|list)\s+tasks?', user_input, re.IGNORECASE):
            tasklist = self.agent.view_tasklist()
            return f"{response_text}\n\n**Current Tasks:**\n{tasklist}"

        # Memory patterns
        remember_match = re.search(r'remember\s+(?:that\s+)?["\']?([^"\']+)["\']?', user_input, re.IGNORECASE)
        if remember_match:
            memory_content = remember_match.group(1)
            result = self.agent.remember(memory_content)
            return f"{response_text}\n\n**Remembered:** {memory_content}"

        return response_text

class AugmentAgent:
    """Main Augment Agent class - comprehensive AI assistant replica"""

    def __init__(self, workspace_root: str = None):
        # Initialize workspace
        self.workspace_root = workspace_root or os.getcwd()

        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Initialize enhanced managers
        self.file_manager = AdvancedFileManager(self.workspace_root)
        self.process_manager = AdvancedProcessManager()
        self.web_manager = WebManager()
        self.task_manager = TaskManager()
        self.memory_manager = MemoryManager()
        self.diagnostic_manager = DiagnosticManager(self.file_manager)
        self.package_manager = PackageManager(self.workspace_root)

        # Initialize advanced features
        self.mermaid_renderer = MermaidRenderer()
        self.codebase_analyzer = CodebaseAnalyzer(self.workspace_root)
        self.tool_parser = StructuredToolCallParser(self)

        # Agent identity and system prompt
        self.identity = """You are Augment Agent developed by Augment Code, an agentic coding AI assistant
        based on the Claude Sonnet 4 model by Anthropic, with access to the developer's codebase through
        Augment's world-leading context engine and integrations. You have comprehensive capabilities for
        file operations, code analysis, web search, process management, task organization, and more."""

        self.system_prompt = """You are a helpful, professional AI coding assistant. You:
        - Always ask for permission before potentially damaging actions
        - Use appropriate package managers instead of manually editing config files
        - Focus on doing exactly what the user asks
        - Suggest testing code changes
        - Are conservative with code changes and respect the existing codebase
        - Provide clear, actionable responses
        - Use proper error handling and logging
        """

    def process_command(self, user_input: str) -> str:
        """Process user command and return response"""
        try:
            # Check for direct tool execution patterns first
            direct_result = self._check_direct_execution(user_input)
            if direct_result:
                return direct_result

            # Prepare context with available tools and current state
            context = self._build_context()

            # Enhanced system prompt with tool information
            enhanced_system_prompt = f"""{self.system_prompt}

Available Tools:
- File operations: view_file, save_file, edit_file, analyze_code
- Web operations: web_search, web_fetch, open_browser
- Process management: launch_process, read_terminal, list_processes
- Task management: add_tasks, update_tasks, view_tasklist
- Memory operations: remember, search_memories
- Advanced features: render_mermaid, codebase_retrieval, get_diagnostics
- Package management: install_package

When you need to use tools, you can either:
1. Describe what you want to do and I'll execute the appropriate tools
2. Use structured tool calls: <tool_call name="tool_name" param="value">content</tool_call>
"""

            # Create full prompt
            full_prompt = f"{enhanced_system_prompt}\n\nContext:\n{context}\n\nUser: {user_input}\n\nAssistant:"

            # Generate response
            response = self.model.generate_content(full_prompt)

            # Process tool calls using advanced parser
            processed_response = self.tool_parser.parse_and_execute_tools(response.text, user_input)

            return processed_response

        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return f"I encountered an error: {str(e)}. Please try rephrasing your request."

    def _check_direct_execution(self, user_input: str) -> Optional[str]:
        """Check for commands that should be executed directly without AI processing"""
        user_lower = user_input.lower().strip()

        # Direct file viewing
        if user_lower.startswith(('show ', 'view ', 'cat ', 'display ')):
            # Extract file path
            words = user_input.split()
            for word in words[1:]:  # Skip the command word
                if '.' in word and not word.startswith('-'):
                    result = self.view_file(word)
                    if 'content' in result:
                        return f"**File: {word}**\n\n<augment_code_snippet path=\"{word}\" mode=\"EXCERPT\">\n````\n{result['content'][:2000]}...\n````\n</augment_code_snippet>"
                    else:
                        return f"Error viewing file {word}: {result.get('error', 'Unknown error')}"

        # Direct task list viewing
        if user_lower in ['tasks', 'show tasks', 'list tasks', 'view tasks']:
            return self.view_tasklist()

        # Direct memory search
        if user_lower.startswith('memories ') or user_lower.startswith('search memories '):
            query = user_input.split(' ', 1)[1] if ' ' in user_input else ''
            if query:
                memories = self.search_memories(query)
                if memories:
                    result = "**Found Memories:**\n"
                    for mem in memories[:5]:
                        result += f"- {mem['content'][:100]}...\n"
                    return result
                else:
                    return f"No memories found for: {query}"

        return None

    def _build_context(self) -> str:
        """Build context information for the AI model"""
        context_parts = []

        # Workspace info
        context_parts.append(f"Workspace: {self.workspace_root}")

        # Available package managers
        managers = self.package_manager.detect_package_manager()
        if managers:
            context_parts.append(f"Package managers: {managers}")

        # Active processes
        processes = self.process_manager.list_processes()
        if processes:
            context_parts.append(f"Active processes: {len(processes)}")

        # Current tasks
        if self.task_manager.tasks:
            context_parts.append(f"Tasks: {len(self.task_manager.tasks)} defined")

        # Recent memories
        recent_memories = list(self.memory_manager.memories.values())[-3:]
        if recent_memories:
            context_parts.append("Recent memories:")
            for memory in recent_memories:
                context_parts.append(f"- {memory.content}")

        return "\n".join(context_parts)

    def _process_tool_calls(self, response_text: str, user_input: str) -> str:
        """Process any tool calls mentioned in the response"""
        # This is a simplified implementation
        # In a full implementation, you would parse structured tool calls

        # Check for common patterns and execute appropriate tools
        if "view file" in user_input.lower() or "show me" in user_input.lower():
            # Extract file path if mentioned
            words = user_input.split()
            for word in words:
                if '.' in word and '/' in word:
                    result = self.file_manager.view_file(word)
                    if 'content' in result:
                        return f"{response_text}\n\nFile content:\n```\n{result['content']}\n```"

        if "search" in user_input.lower() and "web" in user_input.lower():
            # Extract search query
            query = user_input.replace("search", "").replace("web", "").strip()
            if query:
                results = self.web_manager.web_search(query)
                search_results = "\n".join([f"- {r['title']}: {r['url']}" for r in results[:3]])
                return f"{response_text}\n\nSearch results:\n{search_results}"

        return response_text

    def run_interactive(self):
        """Run interactive CLI mode"""
        print("🤖 Augment Agent Replica - Interactive Mode")
        print("Developed by Augment Code")
        print(f"Workspace: {self.workspace_root}")
        print("Type 'help' for commands, 'quit' to exit\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! 👋")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if not user_input:
                    continue

                print("🤔 Thinking...")
                response = self.process_command(user_input)
                print(f"\n🤖 Agent: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _show_help(self):
        """Show comprehensive help information"""
        help_text = """
🤖 AUGMENT AGENT REPLICA - COMPREHENSIVE CAPABILITIES

📁 FILE OPERATIONS:
• View files: "show main.py", "view src/utils.py lines 10-50"
• Edit files: "edit config.py line 25 to change timeout"
• Create files: "create a new file called helpers.py"
• Search in files: "search for 'function' in main.py"
• Code analysis: "analyze the structure of my project"

🌐 WEB & RESEARCH:
• Web search: "search the web for Python async patterns"
• Fetch content: "get content from https://example.com/docs"
• Open browser: "open GitHub repository in browser"

⚙️ PROCESS & TERMINAL:
• Run commands: "run pytest in terminal"
• Launch processes: "start the development server"
• Read terminal: "show me the terminal output"
• Manage processes: "list running processes"

📋 TASK MANAGEMENT:
• Create tasks: "add task to implement user auth"
• View tasks: "show current tasks" or just "tasks"
• Update tasks: "mark database setup as complete"
• Organize: "reorganize task priorities"

🧠 MEMORY SYSTEM:
• Remember: "remember we're using PostgreSQL database"
• Search memories: "what did I tell you about the API?"
• Long-term storage: Automatically saves important information

📊 ADVANCED FEATURES:
• Mermaid diagrams: "create a flowchart showing the user flow"
• Codebase analysis: "find all functions related to authentication"
• Diagnostics: "check for errors in my Python files"
• Package management: "install the requests library"

🔧 DIRECT COMMANDS:
• tasks - Show task list
• memories [query] - Search memories
• show [file] - View file contents
• help - Show this help
• quit/exit - Exit agent

💡 TIPS:
• Be specific about file paths and line numbers
• Use natural language - I understand context
• Ask for explanations of code or concepts
• Request step-by-step guidance for complex tasks

🚀 EXAMPLE WORKFLOWS:
• "Analyze my codebase and create tasks for refactoring"
• "Search for React hooks best practices and remember key points"
• "Show me main.py, then create a test file for it"
• "Install pytest, run tests, and show me any failures"
        """
        print(help_text)

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Augment Agent Replica')
    parser.add_argument('--workspace', '-w', default=None, help='Workspace directory')
    parser.add_argument('--command', '-c', help='Single command to execute')
    parser.add_argument('--example', action='store_true', help='Run example usage')

    args = parser.parse_args()

    try:
        workspace = args.workspace or os.getcwd()
        agent = AugmentAgent(workspace)

        if args.example:
            # Run example
            print("🚀 Augment Agent Replica - Example Usage")
            result = agent.save_file("test.py", "print('Hello World')")
            print(f"Created file: {result}")

            content = agent.view_file("test.py")
            print(f"File content: {content}")

            agent.file_manager.remove_files(["test.py"])
            print("Example completed!")

        elif args.command:
            response = agent.process_command(args.command)
            print(response)
        else:
            agent.run_interactive()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

    # Tool integration methods for direct access
    def view_file(self, path: str, **kwargs) -> Dict[str, Any]:
        """Direct access to file viewing"""
        return self.file_manager.view_file(path, **kwargs)

    def save_file(self, path: str, content: str) -> Dict[str, Any]:
        """Direct access to file saving"""
        return self.file_manager.save_file(path, content)

    def edit_file(self, path: str, old_str: str, new_str: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """Direct access to file editing"""
        return self.file_manager.edit_file(path, old_str, new_str, start_line, end_line)

    def web_search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Direct access to web search"""
        return self.web_manager.web_search(query, num_results)

    def web_fetch(self, url: str) -> str:
        """Direct access to web content fetching"""
        return self.web_manager.web_fetch(url)

    def launch_process(self, command: str, cwd: str = None, wait: bool = False, max_wait_seconds: int = 600) -> Dict[str, Any]:
        """Direct access to process launching"""
        if cwd is None:
            cwd = self.workspace_root
        return self.process_manager.launch_process(command, cwd, wait, max_wait_seconds)

    def add_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Direct access to task creation"""
        return self.task_manager.add_tasks(tasks)

    def update_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Direct access to task updates"""
        return self.task_manager.update_tasks(tasks)

    def view_tasklist(self) -> str:
        """Direct access to task list viewing"""
        return self.task_manager.view_tasklist()

    def remember(self, content: str, tags: List[str] = None) -> Dict[str, Any]:
        """Direct access to memory storage"""
        return self.memory_manager.remember(content, tags)

    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """Direct access to memory search"""
        return self.memory_manager.search_memories(query)

    def get_diagnostics(self, file_paths: List[str]) -> Dict[str, Any]:
        """Direct access to diagnostics"""
        return self.diagnostic_manager.get_diagnostics(file_paths)

    def install_package(self, package: str, language: str = None) -> Dict[str, Any]:
        """Direct access to package installation"""
        return self.package_manager.install_package(package, language)

    # Advanced feature access methods
    def render_mermaid(self, diagram_definition: str, title: str = "Mermaid Diagram") -> Dict[str, Any]:
        """Direct access to Mermaid diagram rendering"""
        return self.mermaid_renderer.render_mermaid(diagram_definition, title)

    def codebase_retrieval(self, information_request: str) -> Dict[str, Any]:
        """Direct access to codebase analysis and retrieval"""
        return self.codebase_analyzer.codebase_retrieval(information_request)

    def view_range_untruncated(self, reference_id: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """Direct access to viewing untruncated content ranges"""
        return self.file_manager.view_range_untruncated(reference_id, start_line, end_line)

    def search_untruncated(self, reference_id: str, search_term: str, context_lines: int = 2) -> Dict[str, Any]:
        """Direct access to searching untruncated content"""
        return self.file_manager.search_untruncated(reference_id, search_term, context_lines)

    def read_terminal(self, only_selected: bool = False) -> Dict[str, Any]:
        """Direct access to terminal reading"""
        return self.process_manager.read_terminal(only_selected)

    def insert_content(self, path: str, insert_line: int, new_content: str) -> Dict[str, Any]:
        """Direct access to content insertion"""
        return self.file_manager.insert_content(path, insert_line, new_content)

    # Enhanced file operations
    def view_file(self, path: str, view_range: Optional[List[int]] = None,
                  search_query_regex: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Enhanced file viewing with advanced features"""
        return self.file_manager.view_file_advanced(path, view_range, search_query_regex, **kwargs)

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Augment Agent Replica - Comprehensive AI Assistant')
    parser.add_argument('--workspace', '-w', default=None, help='Workspace root directory')
    parser.add_argument('--command', '-c', help='Single command to execute')
    parser.add_argument('--file', '-f', help='Execute commands from file')
    parser.add_argument('--version', action='version', version='Augment Agent Replica 1.0')

    args = parser.parse_args()

    try:
        # Initialize agent
        workspace = args.workspace or os.getcwd()
        agent = AugmentAgent(workspace)

        if args.command:
            # Execute single command
            response = agent.process_command(args.command)
            print(response)
        elif args.file:
            # Execute commands from file
            with open(args.file, 'r') as f:
                commands = f.read().strip().split('\n')

            for cmd in commands:
                if cmd.strip():
                    print(f"Executing: {cmd}")
                    response = agent.process_command(cmd)
                    print(f"Response: {response}\n")
        else:
            # Interactive mode
            agent.run_interactive()

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        sys.exit(1)

# Example usage and testing functions
def example_usage():
    """Demonstrate key capabilities"""
    print("🚀 Augment Agent Replica - Example Usage\n")

    # Initialize agent
    agent = AugmentAgent()

    # File operations
    print("1. File Operations:")
    result = agent.save_file("test_file.py", "print('Hello from Augment Agent!')")
    print(f"   Save file: {result}")

    result = agent.view_file("test_file.py")
    print(f"   View file: {result.get('content', 'Error')[:50]}...")

    # Code analysis
    print("\n2. Code Analysis:")
    result = agent.file_manager.analyze_code("test_file.py")
    print(f"   Analysis: {result}")

    # Task management
    print("\n3. Task Management:")
    tasks = [
        {"name": "Setup project", "description": "Initialize the project structure"},
        {"name": "Write tests", "description": "Create comprehensive test suite"}
    ]
    result = agent.add_tasks(tasks)
    print(f"   Add tasks: {result}")
    print(f"   Task list:\n{agent.view_tasklist()}")

    # Memory system
    print("\n4. Memory System:")
    result = agent.remember("This is a test project using Python", ["python", "test"])
    print(f"   Remember: {result}")

    memories = agent.search_memories("python")
    print(f"   Search memories: {len(memories)} found")

    # Web capabilities (if configured)
    print("\n5. Web Capabilities:")
    if os.getenv('GOOGLE_SEARCH_API_KEY'):
        results = agent.web_search("Python best practices", 2)
        print(f"   Web search: {len(results)} results")
    else:
        print("   Web search: API not configured")

    # Process management
    print("\n6. Process Management:")
    result = agent.launch_process("echo 'Hello from process'", wait=True, max_wait_seconds=5)
    print(f"   Process output: {result.get('output', 'Error')}")

    # Clean up
    agent.file_manager.remove_files(["test_file.py"])
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    # Check if running as example
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        example_usage()
    else:
        main()
    
