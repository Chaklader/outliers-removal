# Setup Instructions

## Prerequisites
- Conda or Miniconda installed
- CUDA-capable GPU (recommended for COLMAP and nerfstudio)
- GLoMAP binary

## Installation Steps

### 1. Create and activate conda environment
```bash
conda env create -f environment.yaml
conda activate outliers_removal
```

### 2. Install GLoMAP
GLoMAP is not available via conda/pip. Install it manually:

**Option A: From source**
```bash
git clone https://github.com/colmap/glomap.git
cd glomap
mkdir build && cd build
cmake .. -GNinja
ninja
sudo ninja install
```

**Option B: Download pre-built binary**
Download from the GLoMAP releases page and add to your PATH:
https://github.com/colmap/glomap/releases

### 3. Verify installations
```bash
# Check COLMAP
colmap -h

# Check GLoMAP
glomap -h

# Check pycolmap
python -c "import pycolmap; print(pycolmap.__version__)"

# Check nerfstudio
ns-train --help
```

## Running the Pipeline

### Basic usage
```bash
cd /path/to/your/project
bash pipeline.sh
```

### With custom experiment name
```bash
EXP_NAME=my_experiment bash pipeline.sh
```

### With custom viewer and project name
```bash
EXP_NAME=scene1 PROJECT_NAME=my_project VIEWER=viewer bash pipeline.sh
```

## Directory Structure Expected

Your project directory should contain:
```
project_root/
├── jpgs_sequential/          # Your input images
├── pipeline.sh               # The main pipeline script
├── minimal_pose_filter.py    # Outlier removal script
└── environment.yaml          # This conda environment file
```

## Outputs

After running the pipeline, you'll get:
- `database_*.db` - Original COLMAP database
- `database_*_clean.db` - Cleaned database (outliers removed)
- `sparse_glomap_*/0` - Original GLoMAP reconstruction
- `sparse_clean_*/0` - Cleaned reconstruction (for nerfstudio)
- `outlier_filter_*/` - Outlier analysis files
  - `inliers.txt` - Images kept
  - `outliers.txt` - Images removed
  - `outliers_ids.txt` - COLMAP IDs of removed images
  - `report.json` - Detailed statistics

## Environment Variables

- `EXP_NAME` - Experiment name (default: timestamp)
- `PROJECT_NAME` - Project name for nerfstudio (default: gaussian_splatting)
- `VIEWER` - Nerfstudio viewer type (default: viewer+wandb)

## Notes

- The script uses `jpgs_sequential` as the default image directory
- Modify `IMG_DIR` in `pipeline.sh` if your images are in a different folder
- The script tries GPU first, falls back to CPU for COLMAP operations
- Outlier removal uses 5x median distance threshold by default

