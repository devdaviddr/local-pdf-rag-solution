#!/bin/bash

# run.sh - Script to run the PDF RAG application using the existing virtual environment
# Use this script after running setup.sh for the first time

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Virtual environment name
VENV_NAME="venv"

# Check if virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
    print_error "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    print_error "app.py not found in the current directory."
    exit 1
fi

# Check if source_pdf directory exists and has PDF files
if [ ! -d "source_pdf" ]; then
    print_error "source_pdf directory not found."
    exit 1
fi

PDF_COUNT=$(find source_pdf -name "*.pdf" -type f 2>/dev/null | wc -l)
if [ "$PDF_COUNT" -eq 0 ]; then
    print_error "No PDF files found in source_pdf directory."
    print_status "Please add PDF files to the source_pdf directory."
    exit 1
fi

print_status "Found $PDF_COUNT PDF file(s) in source_pdf directory."

# Activate virtual environment and run the application
print_status "Activating virtual environment and starting the application..."
source "$VENV_NAME/bin/activate"

print_status "Running: python app.py --data_directory source_pdf"
print_success "Starting the PDF RAG application..."
echo ""

python app.py --data_directory source_pdf
