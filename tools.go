package main

// -----------------------------------------------------------------------------
// COMPREHENSIVE TOOLS DEFINITION - 25+ ADVANCED CODING TOOLS
// -----------------------------------------------------------------------------

var tools = []Tool{
	// 1. FILE SYSTEM OPERATIONS
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "read_file",
			Description: "Read the content of a single file from the filesystem with intelligent encoding detection",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path": {Type: "string", Description: "The path to the file to read"},
					"encoding":  {Type: "string", Description: "Optional encoding (utf-8, ascii, etc.). Auto-detected if not specified"},
					"max_lines": {Type: "integer", Description: "Optional maximum number of lines to read (for large files)"},
				},
				Required: []string{"file_path"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "read_multiple_files",
			Description: "Read the content of multiple files efficiently with batch processing",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_paths": {
						Type:        "array",
						Items:       &struct{ Type string `json:"type"` }{Type: "string"},
						Description: "Array of file paths to read",
					},
					"max_total_size": {Type: "integer", Description: "Maximum total size in bytes to read (default: 100KB)"},
				},
				Required: []string{"file_paths"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "create_file",
			Description: "Create or overwrite a file with intelligent directory creation and backup",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path":    {Type: "string", Description: "Path for the file"},
					"content":      {Type: "string", Description: "Content for the file"},
					"create_dirs":  {Type: "boolean", Description: "Create parent directories if they don't exist (default: true)"},
					"backup_existing": {Type: "boolean", Description: "Create backup of existing file (default: false)"},
					"encoding":     {Type: "string", Description: "File encoding (default: utf-8)"},
				},
				Required: []string{"file_path", "content"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "create_multiple_files",
			Description: "Create multiple files in a single operation with atomic transactions",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"files": {
						Type: "array",
						Items: &struct{ Type string `json:"type"` }{Type: "object"},
						Description: "Array of files to create (path, content)",
						Properties: &map[string]struct {
							Type string `json:"type"`
						}{
							"path":    {Type: "string"},
							"content": {Type: "string"},
						},
						Required: []string{"path", "content"},
					},
					"atomic": {Type: "boolean", Description: "All files created or none (rollback on failure)"},
				},
				Required: []string{"files"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "edit_file",
			Description: "Edit a file by replacing a snippet with advanced fuzzy matching and context awareness",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path":        {Type: "string", Description: "Path to the file"},
					"original_snippet": {Type: "string", Description: "Snippet to replace (supports fuzzy matching)"},
					"new_snippet":      {Type: "string", Description: "Replacement snippet"},
					"fuzzy_threshold":  {Type: "integer", Description: "Fuzzy matching threshold (0-100, default: 75)"},
					"context_lines":    {Type: "integer", Description: "Number of context lines to consider (default: 3)"},
					"backup":          {Type: "boolean", Description: "Create backup before editing (default: true)"},
				},
				Required: []string{"file_path", "original_snippet", "new_snippet"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "delete_file",
			Description: "Delete files or directories with safety checks and optional backup",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"path":      {Type: "string", Description: "Path to file or directory to delete"},
					"recursive": {Type: "boolean", Description: "Delete directories recursively (default: false)"},
					"backup":    {Type: "boolean", Description: "Create backup before deletion (default: true)"},
					"force":     {Type: "boolean", Description: "Force deletion without confirmation (default: false)"},
				},
				Required: []string{"path"},
			},
		},
	},

	// 2. CODE ANALYSIS & SEARCH TOOLS
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "code_finder",
			Description: "Smart search for functions, variables, blocks using keywords or regex with AST analysis",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path":     {Type: "string", Description: "The path to the file to search in"},
					"pattern":       {Type: "string", Description: "The keyword or regex pattern to search for"},
					"is_regex":      {Type: "boolean", Description: "True if the pattern is a regex, false for keyword search"},
					"search_type":   {Type: "string", Description: "Type of search: 'all', 'functions', 'classes', 'variables', 'imports'"},
					"context_lines": {Type: "integer", Description: "Number of context lines to include (default: 3)"},
					"case_sensitive": {Type: "boolean", Description: "Case sensitive search (default: false)"},
				},
				Required: []string{"file_path", "pattern", "is_regex"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "grep_plus_plus",
			Description: "Ultra-smart grep across files with filters, patterns, and intelligent ranking",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"directory":     {Type: "string", Description: "The directory to start searching from"},
					"pattern":       {Type: "string", Description: "The regex pattern to search for"},
					"file_filter":   {Type: "string", Description: "Glob pattern to filter files (e.g., '*.go', '*.js')"},
					"recursive":     {Type: "boolean", Description: "True for recursive search, false for top-level only"},
					"max_results":   {Type: "integer", Description: "Maximum number of results to return (default: 100)"},
					"context_lines": {Type: "integer", Description: "Number of context lines around matches (default: 2)"},
					"exclude_dirs":  {Type: "array", Items: &struct{ Type string `json:"type"` }{Type: "string"}, Description: "Directories to exclude from search"},
					"case_sensitive": {Type: "boolean", Description: "Case sensitive search (default: false)"},
				},
				Required: []string{"directory", "pattern"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "string_replacer",
			Description: "Mass-change code lines or strings with advanced pattern matching and validation",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path":     {Type: "string", Description: "The path to the file to modify"},
					"old_string":    {Type: "string", Description: "The string or regex to find"},
					"new_string":    {Type: "string", Description: "The string to replace with"},
					"is_regex":      {Type: "boolean", Description: "True if old_string is a regex, false for literal string"},
					"all_matches":   {Type: "boolean", Description: "True to replace all occurrences, false for first only"},
					"validate_syntax": {Type: "boolean", Description: "Validate syntax after replacement (default: true)"},
					"backup":        {Type: "boolean", Description: "Create backup before replacement (default: true)"},
					"dry_run":       {Type: "boolean", Description: "Show what would be replaced without making changes"},
				},
				Required: []string{"file_path", "old_string", "new_string", "is_regex", "all_matches"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "long_file_indexer",
			Description: "Break up and index massive files for context-aware editing with smart chunking",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path":      {Type: "string", Description: "The path to the large file to index"},
					"chunk_size":     {Type: "integer", Description: "The desired size of each chunk in lines"},
					"overlap_lines":  {Type: "integer", Description: "Number of overlapping lines between chunks (default: 5)"},
					"smart_chunking": {Type: "boolean", Description: "Use intelligent chunking based on code structure (default: true)"},
					"create_index":   {Type: "boolean", Description: "Create searchable index of chunks (default: true)"},
				},
				Required: []string{"file_path", "chunk_size"},
			},
		},
	},

	// 3. GIT & VERSION CONTROL TOOLS
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "git_smart_commit",
			Description: "Intelligent Git commit with automatic message generation and staging",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"message":        {Type: "string", Description: "Commit message (auto-generated if not provided)"},
					"files":          {Type: "array", Items: &struct{ Type string `json:"type"` }{Type: "string"}, Description: "Specific files to commit (all changes if not specified)"},
					"auto_stage":     {Type: "boolean", Description: "Automatically stage files before commit (default: true)"},
					"generate_message": {Type: "boolean", Description: "Generate commit message from changes (default: true if no message)"},
					"push":           {Type: "boolean", Description: "Push after commit (default: false)"},
					"create_branch":  {Type: "string", Description: "Create and switch to new branch before commit"},
				},
				Required: []string{},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "git_branch_manager",
			Description: "Advanced Git branch management with conflict detection and merging",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"action":         {Type: "string", Description: "Action: 'create', 'switch', 'merge', 'delete', 'list', 'status'"},
					"branch_name":    {Type: "string", Description: "Name of the branch"},
					"source_branch":  {Type: "string", Description: "Source branch for create/merge operations"},
					"force":          {Type: "boolean", Description: "Force operation (use with caution)"},
					"auto_resolve":   {Type: "boolean", Description: "Attempt to auto-resolve merge conflicts"},
				},
				Required: []string{"action"},
			},
		},
	},

	// 4. TERMINAL & COMMAND EXECUTION
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "execute_command",
			Description: "Execute shell commands with security checks, timeout, and intelligent output parsing",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"command":        {Type: "string", Description: "The command to execute"},
					"working_dir":    {Type: "string", Description: "Working directory for command execution"},
					"timeout":        {Type: "integer", Description: "Timeout in seconds (default: 30)"},
					"capture_output": {Type: "boolean", Description: "Capture and return command output (default: true)"},
					"shell":          {Type: "string", Description: "Shell to use: 'bash', 'zsh', 'powershell', 'cmd'"},
					"environment":    {Type: "object", Description: "Environment variables to set"},
					"confirm_dangerous": {Type: "boolean", Description: "Require confirmation for potentially dangerous commands"},
				},
				Required: []string{"command"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "dependency_manager",
			Description: "Intelligent package management across multiple ecosystems (npm, pip, go mod, etc.)",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"action":         {Type: "string", Description: "Action: 'install', 'update', 'remove', 'list', 'audit', 'outdated'"},
					"package_name":   {Type: "string", Description: "Name of the package"},
					"version":        {Type: "string", Description: "Specific version to install"},
					"ecosystem":      {Type: "string", Description: "Package ecosystem: 'npm', 'pip', 'go', 'cargo', 'gem', 'composer'"},
					"dev_dependency": {Type: "boolean", Description: "Install as development dependency"},
					"global":         {Type: "boolean", Description: "Install globally"},
					"auto_detect":    {Type: "boolean", Description: "Auto-detect ecosystem from project files (default: true)"},
				},
				Required: []string{"action"},
			},
		},
	},

	// 5. AI-POWERED CODE ANALYSIS & DEBUGGING
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "code_debugger",
			Description: "Advanced AI-powered code debugging with error analysis and fix suggestions",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path":      {Type: "string", Description: "The path to the file to debug"},
					"error_message":  {Type: "string", Description: "The error message or stack trace"},
					"language":       {Type: "string", Description: "The programming language of the code"},
					"context_lines":  {Type: "integer", Description: "Number of context lines around error (default: 10)"},
					"auto_fix":       {Type: "boolean", Description: "Attempt to automatically fix the error (default: false)"},
					"suggest_tests":  {Type: "boolean", Description: "Suggest test cases to prevent similar errors"},
				},
				Required: []string{"file_path", "error_message", "language"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "code_refactor",
			Description: "Intelligent code refactoring with function extraction, duplication removal, and modularization",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path":      {Type: "string", Description: "The path to the file to refactor"},
					"refactor_type":  {Type: "string", Description: "Type: 'extract_function', 'remove_duplication', 'modularize', 'optimize', 'clean'"},
					"details":        {Type: "string", Description: "Specific details for the refactoring"},
					"preserve_behavior": {Type: "boolean", Description: "Ensure behavior is preserved (default: true)"},
					"generate_tests": {Type: "boolean", Description: "Generate tests for refactored code (default: true)"},
					"backup":         {Type: "boolean", Description: "Create backup before refactoring (default: true)"},
				},
				Required: []string{"file_path", "refactor_type"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "code_profiler",
			Description: "Analyze code for performance bottlenecks with optimization suggestions",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path":        {Type: "string", Description: "Path to the code file to profile"},
					"language":         {Type: "string", Description: "Programming language of the code"},
					"profile_type":     {Type: "string", Description: "Type: 'cpu', 'memory', 'io', 'complexity', 'all'"},
					"suggest_optimizations": {Type: "boolean", Description: "Provide optimization suggestions (default: true)"},
					"benchmark":        {Type: "boolean", Description: "Run performance benchmarks if possible"},
				},
				Required: []string{"file_path", "language"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "security_scanner",
			Description: "Comprehensive security scanning for vulnerabilities, secrets, and insecure patterns",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"scan_path":      {Type: "string", Description: "Path to the directory or file to scan"},
					"scan_type":      {Type: "string", Description: "Type: 'code', 'dependencies', 'secrets', 'all'"},
					"severity_filter": {Type: "string", Description: "Minimum severity: 'low', 'medium', 'high', 'critical'"},
					"auto_fix":       {Type: "boolean", Description: "Attempt to auto-fix issues where possible"},
					"generate_report": {Type: "boolean", Description: "Generate detailed security report"},
				},
				Required: []string{"scan_path", "scan_type"},
			},
		},
	},

	// 6. TESTING & QUALITY ASSURANCE
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "test_runner",
			Description: "Generate and execute comprehensive tests with coverage analysis and failure resolution",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"file_path":       {Type: "string", Description: "Path to the code file to test"},
					"test_framework":  {Type: "string", Description: "Testing framework: 'jest', 'pytest', 'go test', 'junit', 'auto'"},
					"test_type":       {Type: "string", Description: "Type: 'unit', 'integration', 'e2e', 'all'"},
					"coverage_target": {Type: "number", Description: "Target coverage percentage (default: 80)"},
					"generate_tests":  {Type: "boolean", Description: "Generate missing test cases (default: true)"},
					"fix_failures":    {Type: "boolean", Description: "Attempt to fix test failures automatically"},
					"parallel":        {Type: "boolean", Description: "Run tests in parallel (default: true)"},
				},
				Required: []string{"file_path"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "code_translator",
			Description: "Translate code between programming languages with optimization for target language idioms",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"source_code":     {Type: "string", Description: "The code to translate"},
					"source_language": {Type: "string", Description: "The original language of the code"},
					"target_language": {Type: "string", Description: "The language to translate the code to"},
					"optimize":        {Type: "boolean", Description: "Optimize for target language best practices (default: true)"},
					"preserve_comments": {Type: "boolean", Description: "Preserve and translate comments (default: true)"},
					"generate_tests":  {Type: "boolean", Description: "Generate tests for translated code"},
				},
				Required: []string{"source_code", "source_language", "target_language"},
			},
		},
	},

	// 7. WEB SEARCH & INFORMATION RETRIEVAL
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "web_search",
			Description: "Search the web for programming solutions, documentation, and code examples",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"query":           {Type: "string", Description: "The search query"},
					"search_type":     {Type: "string", Description: "Type: 'code', 'documentation', 'tutorial', 'stackoverflow', 'github'"},
					"language_filter": {Type: "string", Description: "Filter by programming language"},
					"max_results":     {Type: "integer", Description: "Maximum number of results (default: 10)"},
					"include_code":    {Type: "boolean", Description: "Include code snippets in results (default: true)"},
					"recent_only":     {Type: "boolean", Description: "Search only recent results (last 2 years)"},
				},
				Required: []string{"query"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "documentation_lookup",
			Description: "Fetch and summarize official documentation for libraries, frameworks, and APIs",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"library_name":    {Type: "string", Description: "Name of the library or framework"},
					"topic":           {Type: "string", Description: "Specific topic or function to look up"},
					"language":        {Type: "string", Description: "Programming language"},
					"version":         {Type: "string", Description: "Specific version (latest if not specified)"},
					"include_examples": {Type: "boolean", Description: "Include code examples (default: true)"},
					"summarize":       {Type: "boolean", Description: "Provide a summary of the documentation"},
				},
				Required: []string{"library_name"},
			},
		},
	},

	// 8. TASK MANAGEMENT & PROJECT PLANNING
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "auto_task_planner",
			Description: "Break down complex requests into manageable sub-tasks with intelligent execution planning",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"main_task_description": {Type: "string", Description: "The main task to break down"},
					"complexity_level":      {Type: "string", Description: "Complexity: 'simple', 'medium', 'complex', 'enterprise'"},
					"time_constraint":       {Type: "string", Description: "Time constraint for completion"},
					"priority":              {Type: "string", Description: "Priority: 'low', 'medium', 'high', 'critical'"},
					"dependencies":          {Type: "array", Items: &struct{ Type string `json:"type"` }{Type: "string"}, Description: "External dependencies or requirements"},
					"auto_execute":          {Type: "boolean", Description: "Automatically execute tasks after planning"},
				},
				Required: []string{"main_task_description"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "task_manager",
			Description: "Manage tasks in the task bucket with progress tracking and intelligent scheduling",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"action":      {Type: "string", Description: "Action: 'view', 'add', 'update', 'complete', 'delete', 'reorder'"},
					"task_id":     {Type: "string", Description: "ID of the task to operate on"},
					"task_data":   {Type: "object", Description: "Task data for add/update operations"},
					"filter":      {Type: "string", Description: "Filter tasks by status, priority, or tag"},
					"sort_by":     {Type: "string", Description: "Sort tasks by: 'priority', 'created', 'deadline', 'progress'"},
				},
				Required: []string{"action"},
			},
		},
	},

	// 9. PROJECT ANALYSIS & DOCUMENTATION
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "project_analyzer",
			Description: "Comprehensive project analysis including structure, dependencies, and health metrics",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"project_path":    {Type: "string", Description: "Path to the project directory"},
					"analysis_depth":  {Type: "string", Description: "Depth: 'basic', 'detailed', 'comprehensive'"},
					"include_metrics": {Type: "boolean", Description: "Include code quality metrics (default: true)"},
					"generate_report": {Type: "boolean", Description: "Generate detailed analysis report"},
					"suggest_improvements": {Type: "boolean", Description: "Suggest project improvements"},
				},
				Required: []string{"project_path"},
			},
		},
	},

	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "api_doc_generator",
			Description: "Generate comprehensive API documentation with examples and interactive features",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"source_path":     {Type: "string", Description: "Path to the source code or API specification"},
					"output_format":   {Type: "string", Description: "Format: 'openapi', 'swagger', 'markdown', 'html', 'postman'"},
					"include_examples": {Type: "boolean", Description: "Include code examples (default: true)"},
					"interactive":     {Type: "boolean", Description: "Generate interactive documentation"},
					"output_path":     {Type: "string", Description: "Output path for generated documentation"},
				},
				Required: []string{"source_path", "output_format"},
			},
		},
	},

	// 10. UTILITY & HELPER TOOLS
	{
		Type: "function",
		Function: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			} `json:"parameters"`
		}{
			Name:        "input_fixer",
			Description: "Auto-fix malformed code from screenshots, PDFs, or corrupted sources with intelligent parsing",
			Parameters: struct {
				Type       string `json:"type"`
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				} `json:"properties"`
				Required []string `json:"required"`
			}{
				Type: "object",
				Properties: map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
					Items       *struct {
						Type string `json:"type"`
					} `json:"items,omitempty"`
					Properties *map[string]struct {
						Type string `json:"type"`
					} `json:"properties,omitempty"`
					Required []string `json:"required,omitempty"`
				}{
					"malformed_code": {Type: "string", Description: "The malformed code snippet to fix"},
					"language":       {Type: "string", Description: "Programming language of the code"},
					"source_type":    {Type: "string", Description: "Source: 'screenshot', 'pdf', 'ocr', 'copy_paste', 'unknown'"},
					"fix_indentation": {Type: "boolean", Description: "Fix indentation issues (default: true)"},
					"fix_syntax":     {Type: "boolean", Description: "Fix syntax errors (default: true)"},
					"add_missing":    {Type: "boolean", Description: "Add missing imports/declarations (default: true)"},
				},
				Required: []string{"malformed_code", "language"},
			},
		},
	},
}
