package main

import (
	"fmt"
	"log"
	"os"
	"testing"
	"time"
)

// Test basic functionality of the AI coding agent
func TestAgentInitialization(t *testing.T) {
	// Test that the agent can initialize without errors
	err := initializeAgent()
	if err != nil {
		t.Fatalf("Agent initialization failed: %v", err)
	}
	
	fmt.Println("✅ Agent initialization test passed")
}

func TestToolExecution(t *testing.T) {
	// Test basic tool execution
	args := map[string]interface{}{
		"file_path": "test_file.txt",
		"content":   "Hello, World!",
	}
	
	result, err := executeCreateFile(args)
	if err != nil {
		t.Fatalf("Tool execution failed: %v", err)
	}
	
	if result == "" {
		t.Fatal("Tool execution returned empty result")
	}
	
	// Clean up
	os.Remove("test_file.txt")
	
	fmt.Println("✅ Tool execution test passed")
}

func TestFileOperations(t *testing.T) {
	// Test file creation
	createArgs := map[string]interface{}{
		"file_path": "test_operations.txt",
		"content":   "Test content for operations",
	}
	
	_, err := executeCreateFile(createArgs)
	if err != nil {
		t.Fatalf("File creation failed: %v", err)
	}
	
	// Test file reading
	readArgs := map[string]interface{}{
		"file_path": "test_operations.txt",
	}
	
	content, err := executeReadFile(readArgs)
	if err != nil {
		t.Fatalf("File reading failed: %v", err)
	}
	
	if content == "" {
		t.Fatal("File reading returned empty content")
	}
	
	// Test file deletion
	deleteArgs := map[string]interface{}{
		"path":  "test_operations.txt",
		"force": true,
	}
	
	_, err = executeDeleteFile(deleteArgs)
	if err != nil {
		t.Fatalf("File deletion failed: %v", err)
	}
	
	fmt.Println("✅ File operations test passed")
}

func TestCodeAnalysis(t *testing.T) {
	// Create a test file for analysis
	testCode := `package main

import "fmt"

func main() {
	fmt.Println("Hello, World!")
}

func add(a, b int) int {
	return a + b
}
`
	
	createArgs := map[string]interface{}{
		"file_path": "test_analysis.go",
		"content":   testCode,
	}
	
	_, err := executeCreateFile(createArgs)
	if err != nil {
		t.Fatalf("Test file creation failed: %v", err)
	}
	
	// Test code finder
	finderArgs := map[string]interface{}{
		"file_path": "test_analysis.go",
		"pattern":   "func",
		"is_regex":  false,
	}
	
	result, err := executeCodeFinder(finderArgs)
	if err != nil {
		t.Fatalf("Code finder failed: %v", err)
	}
	
	if result == "" {
		t.Fatal("Code finder returned empty result")
	}
	
	// Clean up
	os.Remove("test_analysis.go")
	
	fmt.Println("✅ Code analysis test passed")
}

func TestTaskManagement(t *testing.T) {
	// Test task creation
	taskArgs := map[string]interface{}{
		"action": "add",
		"task_data": map[string]interface{}{
			"title":       "Test Task",
			"description": "This is a test task",
			"priority":    1,
		},
	}
	
	result, err := executeTaskManager(taskArgs)
	if err != nil {
		t.Fatalf("Task creation failed: %v", err)
	}
	
	if result == "" {
		t.Fatal("Task creation returned empty result")
	}
	
	// Test task viewing
	viewArgs := map[string]interface{}{
		"action": "view",
	}
	
	result, err = executeTaskManager(viewArgs)
	if err != nil {
		t.Fatalf("Task viewing failed: %v", err)
	}
	
	fmt.Println("✅ Task management test passed")
}

func TestSecurityScanner(t *testing.T) {
	// Create a test file with potential security issues
	testCode := `package main

import "fmt"

func main() {
	password := "hardcoded_password_123"
	fmt.Println("Password:", password)
}
`
	
	createArgs := map[string]interface{}{
		"file_path": "test_security.go",
		"content":   testCode,
	}
	
	_, err := executeCreateFile(createArgs)
	if err != nil {
		t.Fatalf("Test file creation failed: %v", err)
	}
	
	// Test security scanner
	scanArgs := map[string]interface{}{
		"scan_path":       "test_security.go",
		"scan_type":       "code",
		"severity_filter": "low",
	}
	
	result, err := executeSecurityScanner(scanArgs)
	if err != nil {
		t.Fatalf("Security scanner failed: %v", err)
	}
	
	if result == "" {
		t.Fatal("Security scanner returned empty result")
	}
	
	// Clean up
	os.Remove("test_security.go")
	
	fmt.Println("✅ Security scanner test passed")
}

func TestProjectAnalyzer(t *testing.T) {
	// Test project analysis
	analyzeArgs := map[string]interface{}{
		"project_path":        ".",
		"analysis_depth":      "basic",
		"include_metrics":     true,
		"suggest_improvements": true,
	}
	
	result, err := executeProjectAnalyzer(analyzeArgs)
	if err != nil {
		t.Fatalf("Project analyzer failed: %v", err)
	}
	
	if result == "" {
		t.Fatal("Project analyzer returned empty result")
	}
	
	fmt.Println("✅ Project analyzer test passed")
}

func TestInputFixer(t *testing.T) {
	// Test input fixer with malformed code
	malformedCode := `func main(){
fmt.Println("Hello")
}`
	
	fixArgs := map[string]interface{}{
		"malformed_code": malformedCode,
		"language":       "go",
		"source_type":    "manual",
	}
	
	result, err := executeInputFixer(fixArgs)
	if err != nil {
		t.Fatalf("Input fixer failed: %v", err)
	}
	
	if result == "" {
		t.Fatal("Input fixer returned empty result")
	}
	
	fmt.Println("✅ Input fixer test passed")
}

// Run all tests
func RunAllTests() {
	fmt.Println("🧪 Running Advanced AI Coding Agent Tests...")
	fmt.Println("=" * 50)
	
	tests := []struct {
		name string
		fn   func(*testing.T)
	}{
		{"Agent Initialization", TestAgentInitialization},
		{"Tool Execution", TestToolExecution},
		{"File Operations", TestFileOperations},
		{"Code Analysis", TestCodeAnalysis},
		{"Task Management", TestTaskManagement},
		{"Security Scanner", TestSecurityScanner},
		{"Project Analyzer", TestProjectAnalyzer},
		{"Input Fixer", TestInputFixer},
	}
	
	passed := 0
	failed := 0
	
	for _, test := range tests {
		fmt.Printf("\n🔍 Running test: %s\n", test.name)
		
		t := &testing.T{}
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("❌ Test %s failed with panic: %v\n", test.name, r)
					failed++
				}
			}()
			
			test.fn(t)
			if t.Failed() {
				fmt.Printf("❌ Test %s failed\n", test.name)
				failed++
			} else {
				fmt.Printf("✅ Test %s passed\n", test.name)
				passed++
			}
		}()
	}
	
	fmt.Println("\n" + "=" * 50)
	fmt.Printf("🎯 Test Results: %d passed, %d failed\n", passed, failed)
	
	if failed == 0 {
		fmt.Println("🎉 All tests passed! The AI Coding Agent is ready for use.")
	} else {
		fmt.Printf("⚠️  %d tests failed. Please review the issues above.\n", failed)
	}
}

func main() {
	// Set up minimal environment for testing
	os.Setenv("DEEPSEEK_API_KEY", "test_key")
	
	// Initialize the agent
	log.SetOutput(os.Stdout)
	
	// Run tests
	RunAllTests()
}
