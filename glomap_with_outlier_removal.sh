#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# GLoMAP Pipeline with Outlier Removal (No Bundle Adjustment)
# =========================================================================
# This script:
# 1. Runs feature extraction and sequential matching
# 2. Runs GLoMAP mapper (global SfM)
# 3. Applies simple median-distance outlier removal (5x threshold)
# 4. Creates cleaned database and sparse model
# 5. Moves outlier images for nerfstudio training
# =========================================================================

# --- Activate conda environment ---
source /home/azureuser/miniconda3/etc/profile.d/conda.sh

# Try difixcrw first, fallback to Difix3D-Charles
if conda activate difixcrw 2>/dev/null; then
    echo "[info] Activated conda environment: difixcrw"
elif conda activate Difix3D-Charles 2>/dev/null; then
    echo "[info] Activated conda environment: Difix3D-Charles"
else
    echo "[error] Could not activate conda environment"
    exit 1
fi

# Verify colmap and glomap are available
if ! command -v colmap &> /dev/null; then
    echo "[error] colmap not found in PATH"
    exit 1
fi

if ! command -v glomap &> /dev/null; then
    echo "[error] glomap not found in PATH"
    exit 1
fi

# --- your paths ---
PROJECT="/mnt/data/UnitA"
IMG_DIR="jpgs_sequential"
REMOVE_OUTLIERS_SCRIPT="/home/azureuser/github/Difix3D-Charles/remove_outlier_poses.py"

# --- experiment name to keep DBs / outputs separate ---
# usage: EXP_NAME=mytest bash glomap_with_outlier_removal.sh
EXP_NAME="${EXP_NAME:-$(date +%Y%m%d_%H%M%S)}"

# --- derived paths ---
cd "$PROJECT"
DB="database_${EXP_NAME}.db"
DB_CLEAN="database_${EXP_NAME}_clean.db"
OUT_GLO="sparse_glomap_${EXP_NAME}"
OUT_CLEAN="sparse_clean_${EXP_NAME}"
OUTLIER_DIR="outlier_filter_${EXP_NAME}"
IMAGES_OUTLIERS="images_outliers_${EXP_NAME}"

# --- clean slate ---
rm -f "$DB" "$DB_CLEAN"
rm -rf "$OUT_GLO" "$OUT_CLEAN" "$OUTLIER_DIR" "$IMAGES_OUTLIERS"
mkdir -p "$OUT_GLO"

echo "=========================================="
echo "GLoMAP + Outlier Removal Pipeline"
echo "=========================================="
echo "Project:          $PROJECT"
echo "Images:           $IMG_DIR"
echo "EXP_NAME:         $EXP_NAME"
echo "Database:         $DB"
echo "Clean Database:   $DB_CLEAN"
echo "Output (GLoMAP):  $OUT_GLO/0"
echo "Output (Clean):   $OUT_CLEAN"
echo "Outlier Filter:   $OUTLIER_DIR"
echo "Outlier Images:   $IMAGES_OUTLIERS"
echo "=========================================="

# =========================================================================
# STEP 1: Feature extraction
# =========================================================================
echo ""
echo ">>> [1/4] Feature extraction"
colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$IMG_DIR" \
  --ImageReader.single_camera true \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 1 \
  --SiftExtraction.gpu_index 0 \
  --SiftExtraction.max_image_size 1600 \
  --SiftExtraction.first_octave -1 \
  --SiftExtraction.domain_size_pooling false \
  --SiftExtraction.estimate_affine_shape 0 \
  --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.peak_threshold 0.006 \
  --log_to_stderr 1 --log_level 2 \
|| colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$IMG_DIR" \
  --ImageReader.single_camera true \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 0 \
  --SiftExtraction.max_image_size 1600 \
  --SiftExtraction.first_octave -1 \
  --SiftExtraction.domain_size_pooling false \
  --SiftExtraction.estimate_affine_shape 0 \
  --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.peak_threshold 0.006 \
  --log_to_stderr 1 --log_level 2

# =========================================================================
# STEP 2: Sequential matching
# =========================================================================
echo ""
echo ">>> [2/4] Sequential matching"
colmap sequential_matcher \
  --database_path "$DB" \
  --SequentialMatching.overlap 20 \
  --SequentialMatching.loop_detection 0 \
  --SiftMatching.guided_matching 1 \
  --SiftMatching.use_gpu 1 \
  --SiftMatching.gpu_index 0 \
  --log_to_stderr 1 --log_level 2 \
|| colmap sequential_matcher \
  --database_path "$DB" \
  --SequentialMatching.overlap 20 \
  --SequentialMatching.loop_detection 0 \
  --SiftMatching.guided_matching 1 \
  --SiftMatching.use_gpu 0 \
  --log_to_stderr 1 --log_level 2

# =========================================================================
# STEP 3: Global SfM (GLoMAP) - NO BUNDLE ADJUSTMENT
# =========================================================================
echo ""
echo ">>> [3/4] GLoMAP mapper (Global SfM)"
glomap mapper \
  --database_path "$DB" \
  --image_path "$IMG_DIR" \
  --output_path "$OUT_GLO"

# Quick stats
echo ""
echo ">>> Model statistics (before outlier removal):"
colmap model_analyzer --path "$OUT_GLO/0" || true

# =========================================================================
# STEP 4: Outlier removal with simple median-distance method
# =========================================================================
echo ""
echo ">>> [4/4] Outlier removal (5x median distance threshold)"

# Run outlier detection to get manifest and create cleaned sparse model
python3 "$REMOVE_OUTLIERS_SCRIPT" \
  --sparse_dir "$OUT_GLO/0" \
  --database_path "$DB" \
  --out_dir "$OUTLIER_DIR" \
  --method median_distance \
  --median_distance_multiplier 5.0 \
  --apply model \
  --max_outlier_frac 0.2 \
  --force \
  --plot advanced \
  --json_report \
  --images_dir "$IMG_DIR" \
  --outliers_dir "$IMAGES_OUTLIERS" \
  --no_move_images

echo ""
echo ">>> Organizing cleaned outputs..."

# Read outlier info
OUTLIER_IDS_FILE="$OUTLIER_DIR/outliers_ids.txt"
INLIER_FILE="$OUTLIER_DIR/inliers.txt"

if [ -f "$OUTLIER_IDS_FILE" ]; then
    NUM_OUTLIERS=$(wc -l < "$OUTLIER_IDS_FILE" || echo "0")
    NUM_INLIERS=$(wc -l < "$INLIER_FILE" || echo "0") 
    echo "[info] Outliers detected: $NUM_OUTLIERS"
    echo "[info] Inliers remaining: $NUM_INLIERS"
else
    NUM_OUTLIERS=0
    NUM_INLIERS=0
    echo "[warn] No outliers file found"
fi

# Copy cleaned sparse model from outlier filter output
mkdir -p "$OUT_CLEAN/0"
if [ -d "$OUTLIER_DIR/sparse/0_cleaned" ]; then
    echo "[info] Copying cleaned sparse model"
    cp -r "$OUTLIER_DIR/sparse/0_cleaned"/* "$OUT_CLEAN/0/"
else
    echo "[warn] No cleaned model found, copying original"
    cp -r "$OUT_GLO/0"/* "$OUT_CLEAN/0/"
fi

# Copy and clean the database
echo ""
echo ">>> Creating cleaned database..."
if [ -f "$DB" ]; then
    cp "$DB" "$DB_CLEAN"
    echo "[info] Copied database: $DB -> $DB_CLEAN"
    
    # Remove outliers from the database copy if we have outlier IDs
    if [ -f "$OUTLIER_IDS_FILE" ] && [ $NUM_OUTLIERS -gt 0 ]; then
        echo "[info] Removing $NUM_OUTLIERS outlier images from database..."
        OUTLIER_IDS_CSV=$(cat "$OUTLIER_IDS_FILE" | tr '\n' ',' | sed 's/,$//')
        
        if [ -n "$OUTLIER_IDS_CSV" ]; then
            colmap image_deleter \
              --database_path "$DB_CLEAN" \
              --image_ids "$OUTLIER_IDS_CSV" \
            && echo "[info] ✓ Outliers removed from database" \
            || echo "[warn] Failed to remove outliers from database"
        fi
    else
        echo "[info] No outliers to remove from database"
    fi
else
    echo "[warn] Original database not found: $DB"
fi

# Stats after cleaning
if [ -d "$OUT_CLEAN/0" ]; then
    echo ""
    echo ">>> Model statistics (after outlier removal):"
    colmap model_analyzer --path "$OUT_CLEAN/0" || true
fi

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo "Original Database:    $DB"
echo "Cleaned Database:     $DB_CLEAN"
echo "Original Model:       $OUT_GLO/0"
echo "Cleaned Model:        $OUT_CLEAN/0"
echo "Outlier Info:         $OUTLIER_DIR/"
echo "  - inliers.txt       (images to keep)"
echo "  - outliers.txt      (images removed)"
echo "  - outliers_ids.txt  (COLMAP image IDs)"
echo "  - report.json       (detailed stats)"
echo "  - outliers_before_after.png (visualization)"
echo ""
echo "For nerfstudio training, use:"
echo "  Images:  $IMG_DIR"
echo "  Model:   $OUT_CLEAN/0"
echo "  Or reference outliers.txt to skip those images"
echo "=========================================="

