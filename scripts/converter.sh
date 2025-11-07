#!/usr/bin/env bash
set -euo pipefail

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_ply_path> <output_spz_path>"
    echo ""
    echo "Example:"
    echo "  $0 assets/Unit_C/splat.ply exports/NeRF.spz"
    exit 1
fi

PLY_FILE="$1"
SPZ_OUTPUT="$2"

echo "=========================================="
echo "PLY to SPZ Converter"
echo "=========================================="
echo "PLY Input:     $PLY_FILE"
echo "SPZ Output:    $SPZ_OUTPUT"
echo "=========================================="

echo ""
echo ">>> Checking for input PLY file..."

if [ ! -f "$PLY_FILE" ]; then
    echo "❌ ERROR: PLY file not found at: $PLY_FILE"
    echo ""
    echo "Please provide a valid PLY file path."
    exit 1
fi

echo "✓ Found PLY file: $PLY_FILE"
PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
echo "  Size: $PLY_SIZE"

echo ""
echo ">>> Converting PLY to SPZ format..."

# Ensure output directory exists
OUTPUT_DIR=$(dirname "$SPZ_OUTPUT")
mkdir -p "$OUTPUT_DIR"

python3 ./spz_converter.py "$PLY_FILE" "$SPZ_OUTPUT"

if [ $? -eq 0 ] && [ -f "$SPZ_OUTPUT" ]; then
    echo ""
    echo "=========================================="
    echo "✅ Conversion Complete!"
    echo "=========================================="
    echo "SPZ file: $SPZ_OUTPUT"
    SPZ_SIZE=$(du -h "$SPZ_OUTPUT" | cut -f1)
    echo "Size:     $SPZ_SIZE"
    echo ""
    echo "Compression ratio:"
    ls -lh "$PLY_FILE" "$SPZ_OUTPUT"
    echo "=========================================="
else
    echo ""
    echo "❌ Conversion failed. Check the error messages above."
    exit 1
fi
