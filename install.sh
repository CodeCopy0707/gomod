#!/bin/bash

# Advanced AI Coding Agent - Installation Script
# This script sets up the complete development environment

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
║                            INSTALLATION SCRIPT                              ║
║                                                                              ║
║  🚀 Setting up your production-ready AI coding assistant...                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    print_header "🔍 Checking System Requirements..."
    
    # Check Go installation
    if command_exists go; then
        GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
        print_success "Go is installed (version: $GO_VERSION)"
        
        # Check if Go version is 1.21 or higher
        if [[ $(echo "$GO_VERSION 1.21" | tr " " "\n" | sort -V | head -n1) == "1.21" ]]; then
            print_success "Go version is compatible"
        else
            print_warning "Go version should be 1.21 or higher for best compatibility"
        fi
    else
        print_error "Go is not installed. Please install Go 1.21 or higher first."
        echo "Visit: https://golang.org/doc/install"
        exit 1
    fi
    
    # Check Git installation (optional)
    if command_exists git; then
        print_success "Git is installed"
    else
        print_warning "Git is not installed. Some features will be limited."
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    if [ "$AVAILABLE_SPACE" -gt 1000000 ]; then
        print_success "Sufficient disk space available"
    else
        print_warning "Low disk space. At least 1GB recommended."
    fi
    
    echo
}

# Install dependencies
install_dependencies() {
    print_header "📦 Installing Dependencies..."
    
    print_status "Downloading Go modules..."
    if go mod tidy; then
        print_success "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
    
    echo
}

# Setup configuration
setup_configuration() {
    print_header "⚙️  Setting up Configuration..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp .env.template .env
            print_success "Created .env file from template"
        else
            print_warning ".env.template not found, creating basic .env file"
            cat > .env << EOF
# AI API Keys
MISTRAL_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=

# Web Search API Keys (Optional)
GOOGLE_SEARCH_API_KEY=
GOOGLE_SEARCH_ENGINE_ID=

# Database Configuration
DATABASE_PATH=./agent_memory.db

# Agent Configuration
DEFAULT_MODEL=deepseek-chat
REASONER_MODEL=deepseek-reasoner
GEMINI_MODEL=gemini-pro
MISTRAL_MODEL=mistral-large-latest
EOF
        fi
    else
        print_status ".env file already exists"
    fi
    
    # Prompt for API keys
    echo
    print_status "API Key Configuration:"
    echo "The agent requires at least one AI API key to function."
    echo "You can configure these now or edit the .env file later."
    echo
    
    read -p "Do you want to configure API keys now? (y/N): " configure_keys
    if [[ $configure_keys =~ ^[Yy]$ ]]; then
        configure_api_keys
    else
        print_warning "Remember to configure API keys in the .env file before running the agent"
    fi
    
    echo
}

# Configure API keys
configure_api_keys() {
    print_status "Configuring API keys..."
    
    # DeepSeek API Key (recommended)
    echo
    echo "🤖 DeepSeek API (Recommended - Fast and cost-effective)"
    echo "Get your API key from: https://platform.deepseek.com/"
    read -p "Enter DeepSeek API key (or press Enter to skip): " deepseek_key
    if [ ! -z "$deepseek_key" ]; then
        sed -i.bak "s/DEEPSEEK_API_KEY=.*/DEEPSEEK_API_KEY=$deepseek_key/" .env
        print_success "DeepSeek API key configured"
    fi
    
    # Gemini API Key
    echo
    echo "🧠 Google Gemini API (Optional - Advanced reasoning)"
    echo "Get your API key from: https://makersuite.google.com/app/apikey"
    read -p "Enter Gemini API key (or press Enter to skip): " gemini_key
    if [ ! -z "$gemini_key" ]; then
        sed -i.bak "s/GEMINI_API_KEY=.*/GEMINI_API_KEY=$gemini_key/" .env
        print_success "Gemini API key configured"
    fi
    
    # Mistral API Key
    echo
    echo "⚡ Mistral API (Optional - European AI)"
    echo "Get your API key from: https://console.mistral.ai/"
    read -p "Enter Mistral API key (or press Enter to skip): " mistral_key
    if [ ! -z "$mistral_key" ]; then
        sed -i.bak "s/MISTRAL_API_KEY=.*/MISTRAL_API_KEY=$mistral_key/" .env
        print_success "Mistral API key configured"
    fi
    
    # Google Search API (optional)
    echo
    echo "🔍 Google Search API (Optional - Web search capabilities)"
    echo "Get your API key from: https://developers.google.com/custom-search/v1/introduction"
    read -p "Enter Google Search API key (or press Enter to skip): " search_key
    if [ ! -z "$search_key" ]; then
        sed -i.bak "s/GOOGLE_SEARCH_API_KEY=.*/GOOGLE_SEARCH_API_KEY=$search_key/" .env
        read -p "Enter Google Search Engine ID: " engine_id
        if [ ! -z "$engine_id" ]; then
            sed -i.bak "s/GOOGLE_SEARCH_ENGINE_ID=.*/GOOGLE_SEARCH_ENGINE_ID=$engine_id/" .env
            print_success "Google Search API configured"
        fi
    fi
    
    # Clean up backup files
    rm -f .env.bak
}

# Build the application
build_application() {
    print_header "🔨 Building Application..."
    
    print_status "Compiling the AI coding agent..."
    if go build -o ai-agent main.go tools.go; then
        print_success "Application built successfully"
        
        # Make executable
        chmod +x ai-agent
        print_success "Made executable"
    else
        print_error "Failed to build application"
        exit 1
    fi
    
    echo
}

# Create desktop shortcut (Linux/macOS)
create_shortcut() {
    print_header "🔗 Creating Shortcuts..."
    
    CURRENT_DIR=$(pwd)
    
    # Create shell script wrapper
    cat > run-ai-agent.sh << EOF
#!/bin/bash
cd "$CURRENT_DIR"
./ai-agent
EOF
    chmod +x run-ai-agent.sh
    print_success "Created run script: run-ai-agent.sh"
    
    # Add to PATH suggestion
    echo
    print_status "To use the agent from anywhere, add this to your shell profile:"
    echo "export PATH=\"$CURRENT_DIR:\$PATH\""
    echo
}

# Run tests
run_tests() {
    print_header "🧪 Running Tests..."
    
    print_status "Running basic functionality tests..."
    
    # Test Go compilation
    if go build -o test-build main.go tools.go; then
        print_success "Compilation test passed"
        rm -f test-build
    else
        print_error "Compilation test failed"
        return 1
    fi
    
    # Test configuration loading
    if [ -f ".env" ]; then
        print_success "Configuration file test passed"
    else
        print_warning "Configuration file not found"
    fi
    
    print_success "All tests passed!"
    echo
}

# Print final instructions
print_final_instructions() {
    print_header "🎉 Installation Complete!"
    
    echo
    echo -e "${GREEN}✅ The Advanced AI Coding Agent is ready to use!${NC}"
    echo
    echo "🚀 To start the agent:"
    echo "   ./ai-agent"
    echo "   or"
    echo "   ./run-ai-agent.sh"
    echo
    echo "📚 Quick start commands:"
    echo "   /help          - Show all available commands"
    echo "   /status        - Check agent status"
    echo "   /tools         - List all available tools"
    echo "   /config        - Show configuration"
    echo
    echo "💡 Example natural language commands:"
    echo '   "Read the main.go file"'
    echo '   "Create a new Python web scraper"'
    echo '   "Find all TODO comments in the project"'
    echo '   "Generate unit tests for the user service"'
    echo
    echo "🔧 Configuration:"
    echo "   Edit .env file to update API keys and settings"
    echo
    echo "📖 Documentation:"
    echo "   Check README.md for detailed usage instructions"
    echo
    echo -e "${CYAN}Happy coding with your new AI assistant! 🤖✨${NC}"
    echo
}

# Main installation flow
main() {
    print_welcome
    
    print_status "Starting installation process..."
    echo
    
    check_requirements
    install_dependencies
    setup_configuration
    build_application
    create_shortcut
    run_tests
    print_final_instructions
}

# Run main function
main "$@"
