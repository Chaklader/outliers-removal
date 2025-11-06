# Setup Guide - Outlier Removal Pipeline

This guide covers the complete setup process for the GLoMAP + Outlier Removal pipeline.

## System Requirements

- **OS**: Ubuntu 24.04 LTS (or similar Linux distribution)
- **GPU**: NVIDIA GPU with CUDA support
- **CPU**: Multi-core processor (tested on Intel Xeon Platinum 8470)
- **RAM**: 16GB+ recommended
- **Disk**: 100GB+ free space

## Prerequisites

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  libboost-all-dev \
  libeigen3-dev
```

**Note**: We only install minimal system dependencies. COLMAP, glog, OpenSSL, and other libraries will be managed through conda to ensure version consistency.

## Installation Steps

### 2. Create Conda Environment

```bash
# Create environment from environment.yaml
conda env create -f environment.yaml

# Activate the environment
conda activate outliers_removal
```

**Note**: The environment includes COLMAP, glog, and all necessary Python packages to ensure consistent library versions.

### 3. Install GLoMAP

GLoMAP is a global Structure-from-Motion tool that provides faster and more robust reconstruction than traditional
incremental methods.

#### Build from Source

```bash
# 1. Activate conda environment
conda activate outliers_removal

# 2. Clone GLoMAP repository
cd ~/Projects
git clone https://github.com/colmap/glomap.git
cd glomap

# 3. Create and enter build directory
mkdir build
cd build

# 4. Configure with CMake (using conda environment)
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

# 5. Build (this will take 10-20 minutes)
make -j$(nproc)

# 6. Install
sudo make install
```

**Important**: GLoMAP must be built with the conda environment activated to ensure it links against the correct library versions (glog, OpenSSL, etc.).

#### Verify Installation

```bash
glomap --help
```

You should see:

```
GLOMAP -- Global Structure-from-Motion

This version was compiled with CUDA!

Usage:
  glomap mapper --database_path DATABASE --output_path MODEL
  glomap mapper_resume --input_path MODEL_INPUT --output_path MODEL_OUTPUT
Available commands:
  help
  mapper
  mapper_resume
  rotation_averager
```

### 4. Verify Python Dependencies

```bash
# Activate environment
conda activate outliers_removal

# Check key packages
python -c "import pycolmap; print('pycolmap:', pycolmap.__version__)"
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import nerfstudio; print('nerfstudio installed')"
```

## Troubleshooting

### GLoMAP Build Issues

**Problem**: CMake can't find glog or OpenSSL

```
cannot find -lglog::glog: No such file or directory
```

**Solution**: Ensure conda environment is activated during build

```bash
conda activate outliers_removal
cd ~/Projects/glomap/build
rm -rf *
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make -j$(nproc)
sudo make install
```

**Problem**: Flag conflicts at runtime

```
ERROR: flag 'timestamp_in_logfile_name' was defined more than once
```

**Solution**: Ensure the pipeline script sets library path correctly (already configured in `pipeline.sh` line 5)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### COLMAP Issues

**Problem**: COLMAP not found

**Solution**: COLMAP is included in the conda environment

```bash
conda activate outliers_removal
which colmap
# Should show: ~/.conda/envs/outliers_removal/bin/colmap
```

### GPU Issues

**Problem**: CUDA not available

**Solution**: Verify CUDA installation

```bash
nvidia-smi
nvcc --version
```

**Problem**: CUDA architecture mismatch

**Solution**: Rebuild GLoMAP with your GPU's architecture

```bash
# Find your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild with specific architecture (e.g., 80 for A100, 86 for RTX 3090)
cd ~/Projects/glomap/build
rm -rf *
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make -j$(nproc)
sudo make install
```

## Directory Structure

After setup, your project should look like:

```
outliers_removal/
├── SETUP.md                    # This file
├── environment.yaml            # Conda environment specification
├── pipeline.sh                 # Main pipeline script
├── minimal_pose_filter.py      # Outlier detection script
├── images/                     # Input images (you provide)
└── [generated during run]
    ├── database.db
    ├── database_clean.db
    ├── colmap/
    │   └── sparse/0/
    ├── colmap_clean/
    │   └── sparse/0/
    └── outlier_filter/
```

## Next Steps

1. Place your images in the `images/` directory
2. Review and customize `pipeline.sh` if needed
3. Run the pipeline: `./pipeline.sh`

See the main README for usage instructions.

## Environment Details

- **Python**: 3.10
- **CUDA Toolkit**: 11.8
- **PyTorch**: 2.0.0+
- **COLMAP**: Installed via conda-forge
- **glog**: Installed via conda-forge (required for COLMAP and GLoMAP)
- **GLoMAP**: Built from source (using conda libraries)
- **Nerfstudio**: Latest version
- **pycolmap**: For Python COLMAP bindings

**Key Design Decision**: All dependencies (COLMAP, glog, Python packages) are managed through conda to ensure consistent library versions and avoid conflicts. GLoMAP is built from source but uses the conda environment's libraries.

## Useful Commands

```bash
# Activate environment
conda activate outliers_removal

# Deactivate environment
conda deactivate

# Update environment
conda env update -f environment.yaml

# Remove environment
conda env remove -n outliers_removal

# Check GLoMAP version
glomap --help

# Check COLMAP version
colmap --version
```

## Support

For issues with:

- **GLoMAP**: https://github.com/colmap/glomap/issues
- **COLMAP**: https://github.com/colmap/colmap/issues
- **Nerfstudio**: https://github.com/nerfstudio-project/nerfstudio/issues
