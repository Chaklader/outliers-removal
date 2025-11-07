#!/usr/bin/env python3
"""
Find the largest COLMAP sparse model by images.bin file size.
Usage: python3 find_largest_model.py <sparse_dir>
"""

import os
import sys
from pathlib import Path


def find_largest_model(sparse_dir: str) -> str:
    """
    Find the largest model in a COLMAP sparse directory.
    
    Args:
        sparse_dir: Path to directory containing model subdirectories (0, 1, 2, ...)
    
    Returns:
        Name of the largest model directory (e.g., "0", "1")
    """
    sparse_path = Path(sparse_dir)
    
    if not sparse_path.exists():
        print("0", file=sys.stderr)
        return "0"
    
    # Find all model directories (numeric names)
    models = [
        d.name for d in sparse_path.iterdir() 
        if d.is_dir() and d.name.isdigit()
    ]
    
    if not models:
        return "0"
    
    # Find largest by images.bin file size
    def get_size(model_name: str) -> int:
        images_bin = sparse_path / model_name / "images.bin"
        return images_bin.stat().st_size if images_bin.exists() else 0
    
    largest = max(models, key=get_size)
    return largest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 find_largest_model.py <sparse_dir>", file=sys.stderr)
        sys.exit(1)
    
    sparse_dir = sys.argv[1]
    largest_model = find_largest_model(sparse_dir)
    print(largest_model)
