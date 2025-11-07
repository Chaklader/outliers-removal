#!/usr/bin/env bash
set -euo pipefail

# Fixed paths for NeRF processing
PLY_FILE="assets/NeRF.ply"
SPZ_OUTPUT="exports/NeRF.spz"
ROTATION_OUTPUT="exports/NeRF_matrix_4_4.json"
BOUNDARY_OUTPUT="exports/NeRF_boundary.json"

echo "=========================================="
echo "NeRF Processing Pipeline"
echo "=========================================="
echo "PLY Input:        $PLY_FILE"
echo "SPZ Output:       $SPZ_OUTPUT"
echo "Rotation Output:  $ROTATION_OUTPUT"
echo "Boundary Output:  $BOUNDARY_OUTPUT"
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
echo ">>> Converting PLY to SPZ..."
mkdir -p exports
python3 ./spz_converter.py "$PLY_FILE" "$SPZ_OUTPUT"

echo ""
echo ">>> Generating rotation matrix..."
python3 ./rotation_corrction.py "$PLY_FILE" "$ROTATION_OUTPUT"

echo ""
echo ">>> Generating boundary..."
python3 ./boundary.py "$PLY_FILE" "$ROTATION_OUTPUT" "$BOUNDARY_OUTPUT"

echo ""
echo "✅ Done!"
