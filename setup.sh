#!/bin/bash

# setup.sh - Automated setup script for PDF RAG Solution
# This script creates a virtual environment, installs dependencies, and runs the application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "Starting PDF RAG Solution setup..."
print_status "Working directory: $SCRIPT_DIR"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
print_status "Found Python version: $PYTHON_VERSION"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in the current directory."
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    print_error "app.py not found in the current directory."
    exit 1
fi

# Check if source_pdf directory exists
if [ ! -d "source_pdf" ]; then
    print_warning "source_pdf directory not found. Creating it..."
    mkdir -p source_pdf
    print_status "Created source_pdf directory. Please add your PDF files to this directory."
fi

# Virtual environment name
VENV_NAME="venv"

# Remove existing virtual environment if it exists
if [ -d "$VENV_NAME" ]; then
    print_warning "Existing virtual environment found. Removing it..."
    rm -rf "$VENV_NAME"
fi

# Create virtual environment
print_status "Creating virtual environment..."
python3 -m venv "$VENV_NAME"

if [ ! -d "$VENV_NAME" ]; then
    print_error "Failed to create virtual environment."
    exit 1
fi

print_success "Virtual environment created successfully."

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing requirements from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Requirements installed successfully."
else
    print_error "Failed to install requirements."
    exit 1
fi

# Check if there are PDF files in source_pdf directory
PDF_COUNT=$(find source_pdf -name "*.pdf" -type f 2>/dev/null | wc -l)
if [ "$PDF_COUNT" -eq 0 ]; then
    print_warning "No PDF files found in source_pdf directory."
    print_status "Please add PDF files to the source_pdf directory before running the application."
    print_status "You can run the application later with: ./run.sh"
    exit 0
fi

print_status "Found $PDF_COUNT PDF file(s) in source_pdf directory."

# Start the application
print_status "Starting the PDF RAG application..."
print_status "Running: python app.py --data_directory source_pdf"
echo ""
print_success "Setup completed! Starting the application..."
echo ""

# Run the application with the virtual environment Python
python app.py --data_directory source_pdf
