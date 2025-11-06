#!/usr/bin/env bash
set -euo pipefail

# Prioritize conda libraries to match build environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

EXPERIMENT_NAME="my_experiment"
OUTPUT_DIR="./exports"

echo "=========================================="
echo "Gaussian Splat Export"
echo "=========================================="
echo "Experiment:       $EXPERIMENT_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

echo ""
echo ">>> Searching for trained model config..."

# Find the config file from nerfstudio outputs
CFGFILE=$(find ./outputs/ -name "config.yml" -path "*/${EXPERIMENT_NAME}/*" | head -1)

if [ -z "$CFGFILE" ]; then
    echo "❌ ERROR: No config.yml found for experiment '$EXPERIMENT_NAME'"
    echo "   Make sure training completed successfully."
    echo "   Looking in: ./outputs/*/${EXPERIMENT_NAME}/*/config.yml"
    exit 1
fi

echo "✓ Found config: $CFGFILE"

echo ""
echo ">>> Exporting Gaussian Splat..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Export the model
ns-export gaussian-splat \
    --load-config "$CFGFILE" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "✅ Export Complete!"
echo "=========================================="
echo "Exported files location: $OUTPUT_DIR"
echo ""
echo "Contents:"
ls -lh "$OUTPUT_DIR"
echo "=========================================="
