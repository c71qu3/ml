#!/bin/bash

# Function to compare python versions
version_ge() {
  [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

# Try python3 then python
if command -v python3 &> /dev/null; then
    PYTHON_BIN=python3
elif command -v python &> /dev/null; then
    PYTHON_BIN=python
else
    echo "Error: Python 3.12+ is required and not found."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
REQUIRED_VERSION="3.12.0"

if ! version_ge "$PYTHON_VERSION" "$REQUIRED_VERSION"; then
    echo "Error: Python version >= 3.12.0 is required, but found $PYTHON_VERSION."
    exit 1
fi

echo "Using $PYTHON_BIN (version $PYTHON_VERSION)"

# Create virtual environment in .venv directory
$PYTHON_BIN -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip (recommended)
pip install --upgrade pip

# Install the required libraries from requirements.txt
pip install -r requirements.txt

echo "Virtual environment setup complete."