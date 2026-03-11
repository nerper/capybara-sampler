#!/bin/bash
# Run the Word Familiarity API locally
# Usage: ./run.sh [dev]
#   dev = development mode with hot reload on port 8000
#   (no args) = production mode on port 7000

cd "$(dirname "$0")"

# Use venv if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

# Poetry: poetry run python main.py
# Or: python main.py

if [ "$1" = "dev" ]; then
  export DEV_MODE=true
  echo "Starting in DEVELOPMENT mode (port 8000, hot reload)..."
else
  echo "Starting in PRODUCTION mode (port 7000)..."
fi

python main.py
