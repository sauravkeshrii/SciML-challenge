#!/bin/bash
# Launcher script for Gen-SHM complete demonstration

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f "gen-shm-env/bin/activate" ]; then
    source gen-shm-env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

# Run the complete demonstration
echo "🚀 Starting Gen-SHM Complete Working Demonstration..."
python complete_working_demo.py