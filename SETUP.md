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
  libeigen3-dev \
  libgoogle-glog-dev \
  libgoogle-glog0v6 \
  libssl-dev \
  libcrypto++-dev
```

## Installation Steps

### 2. Create Conda Environment

```bash
# Create environment from environment.yaml
conda env create -f environment.yaml

# Activate the environment
conda activate outliers_removal
```

### 3. Install GLoMAP

GLoMAP is a global Structure-from-Motion tool that provides faster and more robust reconstruction than traditional
incremental methods.

#### Build from Source

```bash
# 1. Install system dependencies for GLoMAP
sudo apt install -y libgoogle-glog-dev libgoogle-glog0v6

# 2. Activate conda environment and install OpenSSL
conda activate outliers_removal
conda install -c conda-forge openssl=3.2.0 -y

# 3. Clone GLoMAP repository
cd ~/Projects
git clone https://github.com/colmap/glomap.git
cd glomap

# 4. Create and enter build directory
mkdir build
cd build

# 5. Configure with CMake (using conda OpenSSL)
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DOPENSSL_ROOT_DIR=$CONDA_PREFIX \
  -DOPENSSL_INCLUDE_DIR=$CONDA_PREFIX/include \
  -DOPENSSL_CRYPTO_LIBRARY=$CONDA_PREFIX/lib/libcrypto.so \
  -DOPENSSL_SSL_LIBRARY=$CONDA_PREFIX/lib/libssl.so \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

# 6. Build (this will take 10-20 minutes)
make -j$(nproc)

# 7. Install
sudo make install
```

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

**Problem**: OpenSSL linking errors during build

```
undefined reference to `X509_STORE_CTX_init_rpk@OPENSSL_3.2.0'
```

**Solution**: Install system glog library

```bash
sudo apt install libgoogle-glog-dev libgoogle-glog0v6
```

**Problem**: Missing `libglog.so.2` at runtime

```
glomap: error while loading shared libraries: libglog.so.2
```

**Solution**: Install system glog (same as above)

```bash
sudo apt install libgoogle-glog-dev libgoogle-glog0v6
```

### COLMAP Issues

**Problem**: COLMAP not found

**Solution**: COLMAP is included in the conda environment

```bash
conda activate outliers_removal
which colmap
```

### GPU Issues

**Problem**: CUDA not available

**Solution**: Verify CUDA installation

```bash
nvidia-smi
nvcc --version
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
- **COLMAP**: Installed via conda
- **GLoMAP**: Built from source
- **Nerfstudio**: Latest version
- **pycolmap**: For Python COLMAP bindings

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
