"""
Utility functions for logging, validation, and helpers.
UPDATED: Added UTF-8 encoding for file writes to handle Unicode characters.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(level=logging.INFO):
    """Configure logging with timestamp and level."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def save_output(df, filename, output_dir="output"):
    """Save DataFrame to CSV."""
    ensure_directory(output_dir)
    filepath = Path(output_dir) / filename
    df.to_csv(filepath, index=False, encoding='utf-8')
    return filepath

def save_text(content, filename, output_dir="output"):
    """Save text content to file with UTF-8 encoding."""
    ensure_directory(output_dir)
    filepath = Path(output_dir) / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath

def save_json(data, filename, output_dir="output"):
    """Save JSON data to file with UTF-8 encoding."""
    import json
    ensure_directory(output_dir)
    filepath = Path(output_dir) / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath