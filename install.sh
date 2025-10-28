#!/bin/bash

# STORM CLI Installation Script

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌪️  STORM CLI - Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python version
echo "➤ Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found"
echo ""

# Check pip
echo "➤ Checking pip..."
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed"
    echo "   Please install pip"
    exit 1
fi
echo "✓ pip found"
echo ""

# Install dependencies
echo "➤ Installing dependencies..."
echo "  - dspy-ai"
echo "  - pydantic"
echo ""

pip install -q dspy-ai pydantic || {
    echo "❌ Failed to install dependencies"
    exit 1
}

echo "✓ Dependencies installed"
echo ""

# Make executable
echo "➤ Making STORM CLI executable..."
chmod +x storm
echo "✓ STORM CLI is now executable"
echo ""

# Check for API key
echo "➤ Checking OpenRouter API key..."
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  OPENROUTER_API_KEY is not set"
    echo ""
    echo "   To set it, run:"
    echo "   export OPENROUTER_API_KEY='your-key-here'"
    echo ""
    echo "   Or add to ~/.bashrc or ~/.zshrc:"
    echo "   echo 'export OPENROUTER_API_KEY=\"your-key-here\"' >> ~/.bashrc"
    echo ""
else
    echo "✓ OPENROUTER_API_KEY is set"
    echo ""
fi

# Optional: Add to PATH
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Installation complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Usage:"
echo "  python storm \"Your Topic\"                    # Basic usage"
echo "  python storm \"Topic\" --words 1500           # Long article"
echo "  python storm \"Topic\" -o output.md --format md  # Save to file"
echo ""
echo "For more options:"
echo "  python storm --help"
echo ""
echo "Documentation:"
echo "  cat CLI_README.md"
echo ""

# Test run
echo "Would you like to test the installation? (y/n)"
read -r REPLY
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo ""
        echo "⚠️  Cannot test without OPENROUTER_API_KEY"
        echo "   Please set it first:"
        echo "   export OPENROUTER_API_KEY='your-key-here'"
        echo ""
    else
        echo ""
        echo "Running test: python storm \"Test Topic\" --words 200"
        echo ""
        python storm "Test Topic" --words 200
    fi
fi

echo ""
echo "Happy writing! 🌪️"
