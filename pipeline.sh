#!/usr/bin/env bash
set -euo pipefail

# Prioritize conda libraries to match GLoMAP build environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

IMG_DIR="images"

EXP_NAME="outlier_removal"
DB="database.db"
DB_CLEAN="database_clean.db"
OUT_GLO="colmap/sparse"
OUT_CLEAN="colmap_clean/sparse"
OUTLIER_DIR="outlier_filter"
IMAGES_OUTLIERS="images_outliers"

rm -f "$DB" "$DB_CLEAN"
rm -rf "$OUT_GLO" "$OUT_CLEAN" "$OUTLIER_DIR" "$IMAGES_OUTLIERS"
mkdir -p "$OUT_GLO"

echo "=========================================="
echo "GLoMAP + Outlier Removal Pipeline"
echo "=========================================="
echo "Project:          $(pwd)"
echo "Images:           $IMG_DIR"
echo "EXP_NAME:         $EXP_NAME"
echo "Database:         $DB"
echo "Clean Database:   $DB_CLEAN"
echo "Output (GLoMAP):  $OUT_GLO/0"
echo "Output (Clean):   $OUT_CLEAN"
echo "Outlier Filter:   $OUTLIER_DIR"
echo "Outlier Images:   $IMAGES_OUTLIERS"
echo "=========================================="

echo ""
echo ">>> [1/5] Feature extraction"
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
  --log_to_stderr 1 --log_level 2

echo ""
echo ">>> [2/5] Sequential matching"
colmap sequential_matcher \
  --database_path "$DB" \
  --SequentialMatching.overlap 20 \
  --SequentialMatching.loop_detection 0 \
  --SiftMatching.guided_matching 1 \
  --SiftMatching.use_gpu 1 \
  --SiftMatching.gpu_index 0 \
  --log_to_stderr 1 --log_level 2

echo ""
echo ">>> [3/5] COLMAP mapper (Incremental SfM)"
colmap mapper \
  --database_path "$DB" \
  --image_path "$IMG_DIR" \
  --output_path "$OUT_GLO"

echo ""
echo ">>> Model statistics (before outlier removal):"
colmap model_analyzer --path "$OUT_GLO/0" || true

echo ""
echo ">>> [4/5] Outlier removal (5x median distance threshold)"

python3 ./minimal_pose_filter.py \
  --sparse_dir "$OUT_GLO/0" \
  --out_dir "$OUTLIER_DIR" \
  --apply manifest \
  --max_outlier_frac 0.2 \
  --force

echo ""
echo ">>> Organizing cleaned outputs..."

echo "[info] Creating cleaned COLMAP model..."
mkdir -p "$OUT_CLEAN/0"

colmap model_converter \
  --input_path "$OUT_GLO/0" \
  --output_path "$OUT_CLEAN/0" \
  --output_type BIN \
  --image_list_path "$OUTLIER_DIR/inliers.txt"

echo ""
echo ">>> Creating cleaned database..."

cp "$DB" "$DB_CLEAN"
OUTLIER_IDS_CSV=$(cat "$OUTLIER_DIR/outliers_ids.txt" | tr '\n' ',' | sed 's/,$//')

colmap image_deleter \
  --database_path "$DB_CLEAN" \
  --image_ids "$OUTLIER_IDS_CSV"

echo ""
echo ">>> Model statistics (after outlier removal):"
colmap model_analyzer --path "$OUT_CLEAN/0"


echo "=========================================="
echo "Original Model:       $OUT_GLO/0"
echo "Cleaned Model:        $OUT_CLEAN/0"
echo "Outlier Info:         $OUTLIER_DIR/"
echo "=========================================="

echo ""
echo ">>> [5/5] Training Gaussian Splatting with cleaned data"

ns-train splatfacto \
    --machine.num-devices 1 \
    --vis viewer+wandb \
    --viewer.quit-on-train-completion True \
    --log-gradients False \
    --pipeline.datamanager.images-on-gpu True \
    --pipeline.datamanager.train-cameras-sampling-strategy fps \
    --pipeline.model.use-bilateral-grid True \
    --experiment-name "my_experiment" \
    --project-name "gaussian_splatting" \
    --use-grad-scaler False \
    --load-scheduler False \
    colmap \
    --data . \
    --colmap-path $OUT_CLEAN/0 \
    --auto-scale-poses False \
    --downscale-factor 1

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="

