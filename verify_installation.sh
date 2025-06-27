#!/bin/bash
# Make this script executable: chmod +x verify_installation.sh

# Advanced AI Coding Agent - Installation Verification Script
# This script verifies that the agent is properly installed and functional

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Print welcome message
print_welcome() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🤖 ADVANCED AI CODING TERMINAL AGENT                     ║
║                                                                              ║
║                         INSTALLATION VERIFICATION                           ║
║                                                                              ║
║  🔍 Verifying that your AI coding assistant is ready for action...          ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Verify Go installation
verify_go() {
    print_header "🔍 Verifying Go Installation..."
    
    if command_exists go; then
        GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
        print_success "Go is installed (version: $GO_VERSION)"
        return 0
    else
        print_error "Go is not installed"
        return 1
    fi
}

# Verify project files
verify_files() {
    print_header "📁 Verifying Project Files..."
    
    required_files=(
        "main.go"
        "tools.go"
        "go.mod"
        "go.sum"
        "README.md"
        ".env.template"
        "install.sh"
    )
    
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            print_success "Found: $file"
        else
            print_warning "Missing: $file"
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        print_success "All required files are present"
        return 0
    else
        print_error "Missing ${#missing_files[@]} required files"
        return 1
    fi
}

# Verify dependencies
verify_dependencies() {
    print_header "📦 Verifying Dependencies..."
    
    print_status "Checking Go modules..."
    if go mod verify; then
        print_success "All dependencies verified"
        return 0
    else
        print_error "Dependency verification failed"
        return 1
    fi
}

# Verify build
verify_build() {
    print_header "🔨 Verifying Build Process..."
    
    print_status "Attempting to build the agent..."
    if go build -o ai-agent-test main.go; then
        print_success "Build successful"
        
        # Clean up test binary
        rm -f ai-agent-test
        return 0
    else
        print_error "Build failed"
        return 1
    fi
}

# Verify configuration
verify_config() {
    print_header "⚙️  Verifying Configuration..."
    
    if [ -f ".env" ]; then
        print_success "Configuration file (.env) exists"
        
        # Check for API keys
        if grep -q "DEEPSEEK_API_KEY=" .env; then
            if grep -q "DEEPSEEK_API_KEY=your_" .env; then
                print_warning "DeepSeek API key not configured (using template value)"
            else
                print_success "DeepSeek API key configured"
            fi
        else
            print_warning "DeepSeek API key not found in .env"
        fi
        
        return 0
    else
        print_warning "Configuration file (.env) not found"
        print_status "You can create it from the template: cp .env.template .env"
        return 1
    fi
}

# Run basic functionality test
verify_functionality() {
    print_header "🧪 Verifying Basic Functionality..."
    
    print_status "Running basic functionality tests..."
    
    # Create a simple test
    cat > test_basic.go << 'EOF'
package main

import (
    "fmt"
    "os"
)

func main() {
    // Test basic initialization
    os.Setenv("DEEPSEEK_API_KEY", "test_key")
    
    fmt.Println("Testing basic agent functionality...")
    
    // Test tool execution
    args := map[string]interface{}{
        "file_path": "test_verify.txt",
        "content":   "Verification test content",
    }
    
    result, err := executeCreateFile(args)
    if err != nil {
        fmt.Printf("❌ Basic functionality test failed: %v\n", err)
        os.Exit(1)
    }
    
    fmt.Printf("✅ Basic functionality test passed: %s\n", result)
    
    // Clean up
    os.Remove("test_verify.txt")
    
    fmt.Println("🎉 All basic functionality tests passed!")
}
EOF
    
    if go run test_basic.go; then
        print_success "Basic functionality tests passed"
        rm -f test_basic.go test_verify.txt
        return 0
    else
        print_error "Basic functionality tests failed"
        rm -f test_basic.go test_verify.txt
        return 1
    fi
}

# Run comprehensive tests
run_comprehensive_tests() {
    print_header "🔬 Running Comprehensive Tests..."
    
    if [ -f "test_agent.go" ]; then
        print_status "Running comprehensive test suite..."
        if go run test_agent.go; then
            print_success "Comprehensive tests passed"
            return 0
        else
            print_error "Some comprehensive tests failed"
            return 1
        fi
    else
        print_warning "Comprehensive test suite not found (test_agent.go)"
        return 1
    fi
}

# Print final results
print_results() {
    local total_checks=$1
    local passed_checks=$2
    local failed_checks=$((total_checks - passed_checks))
    
    print_header "📊 Verification Results"
    echo
    echo -e "${GREEN}✅ Passed: $passed_checks${NC}"
    echo -e "${RED}❌ Failed: $failed_checks${NC}"
    echo -e "${BLUE}📈 Success Rate: $(( passed_checks * 100 / total_checks ))%${NC}"
    echo
    
    if [ $failed_checks -eq 0 ]; then
        echo -e "${GREEN}🎉 VERIFICATION COMPLETE!${NC}"
        echo -e "${GREEN}Your Advanced AI Coding Agent is ready for use!${NC}"
        echo
        echo -e "${CYAN}🚀 To start the agent:${NC}"
        echo "   ./ai-agent"
        echo
        echo -e "${CYAN}📚 For help and documentation:${NC}"
        echo "   Type '/help' in the agent"
        echo "   Check README.md for detailed instructions"
        echo
    else
        echo -e "${RED}⚠️  VERIFICATION INCOMPLETE${NC}"
        echo -e "${YELLOW}Please address the failed checks above before using the agent.${NC}"
        echo
        echo -e "${CYAN}💡 Common solutions:${NC}"
        echo "   - Run 'go mod tidy' to fix dependency issues"
        echo "   - Copy .env.template to .env and configure API keys"
        echo "   - Check that Go 1.21+ is installed"
        echo
    fi
}

# Main verification flow
main() {
    print_welcome
    
    local total_checks=0
    local passed_checks=0
    
    # Run all verification checks
    checks=(
        "verify_go"
        "verify_files"
        "verify_dependencies"
        "verify_build"
        "verify_config"
        "verify_functionality"
        "run_comprehensive_tests"
    )
    
    for check in "${checks[@]}"; do
        total_checks=$((total_checks + 1))
        if $check; then
            passed_checks=$((passed_checks + 1))
        fi
        echo
    done
    
    print_results $total_checks $passed_checks
}

# Run main function
main "$@"
