# Installation (EmbodMocap)

**Language / 语言切换:** [English](install.md) | [中文](install_zh.md)

This guide covers the open-source main pipeline setup and the required dependencies / checkpoints.

## 1) Clone Repository

```bash
git clone --recurse-submodules https://github.com/WenjiaWang0312/EmbodMocap
cd EmbodMocap
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

## 2) Create Python Environment

```bash
conda create -n embodmocap python=3.11 -y
conda activate embodmocap
```

Install PyTorch according to your CUDA runtime. Two examples are shown below:

```bash
# CUDA 12.4 example
pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124

# CUDA 12.8 example
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

## 3) Install Core Dependencies

```bash
pip install -r requirements.txt
pip install -e embod_mocap
```

## 4) Third-party Modules (Submodules)

Third-party dependencies are managed as Git submodules rather than vendored code.

The submodules can be added with:

```bash
cd embod_mocap
git submodule add https://github.com/luca-medeiros/lang-segment-anything thirdparty/lang_sam
git submodule add https://github.com/Robbyant/lingbot-depth thirdparty/lingbot_depth
git submodule add https://github.com/ViTAE-Transformer/ViTPose thirdparty/ViTPose
```

Common modules:

- `embod_mocap/thirdparty/lingbot_depth`
- `embod_mocap/thirdparty/lang_sam`
- `embod_mocap/thirdparty/ViTPose`

If editable install is needed:

```bash
pip install -e embod_mocap/thirdparty/lingbot_depth
pip install -e embod_mocap/thirdparty/lang_sam
pip install -e embod_mocap/thirdparty/ViTPose
```

## 5) COLMAP

COLMAP install guide:

- [COLMAP Installation Guide](https://colmap.github.io/install.html)

In our setup, installing COLMAP with the system package manager worked well:

```bash
sudo apt install colmap
```

## 6) Other Dependencies

### [torch-scatter](https://github.com/rusty1s/pytorch_scatter) (for selected training/eval paths)

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### [pytorch3d](https://github.com/facebookresearch/pytorch3d) (optional, for camera-space rendering)

Choose a pytorch3d package that matches your CUDA / PyTorch setup. For example, you can download a package from the [Tsinghua mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/) and then install it locally. The rendering wrapper used in this project is [torch3d_render](https://github.com/WenjiaWang0312/torch3d_render.git).

```bash
conda install -c iopath iopath
conda install -c bottler nvidiacub
conda install --use-local xxx.tar.bz2
pip install git+https://github.com/WenjiaWang0312/torch3d_render.git
```

### Custom downloads (optional)

If you prefer to manually choose individual downloads instead of using the bundled scripts, the original checkpoint and COLMAP vocab tree sources are listed below:

- [VGGT model checkpoint](https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt)
- [SAM2.1 Hiera Large checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
- [SAM2.1 Hiera Small checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [VIMO checkpoint (Google Drive)](https://drive.google.com/file/d/1fdeUxn_hK4ERGFwuksFpV_-_PHZJuoiW/view?usp=share_link)
- [YOLOv8x checkpoint (Google Drive)](https://drive.google.com/uc?id=1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo&export=download&confirm=t)
- [ViTPose-H Multi-COCO checkpoint (Google Drive)](https://drive.google.com/uc?id=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso&export=download&confirm=t)
- [COLMAP releases](https://github.com/colmap/colmap/releases/) for vocab tree files

## 7) Troubleshooting

### COLMAP

#### Installation

- [COLMAP issue #2464](https://github.com/colmap/colmap/issues/2464)

#### `No CMAKE_CUDA_COMPILER could be found`

- [jetsonhacks/buildLibrealsense2TX issue #13](https://github.com/jetsonhacks/buildLibrealsense2TX/issues/13)

#### `FAILED: src/colmap/mvs/CMakeFiles/xxx`

- [COLMAP issue #2091](https://github.com/colmap/colmap/issues/2091)

#### `libcudart.so` error

- [vllm issue #1369](https://github.com/vllm-project/vllm/issues/1369)
- Example:

```bash
export LD_LIBRARY_PATH=/home/wwj/miniconda3/envs/droidenv/lib/:$LD_LIBRARY_PATH
```

#### Registration issues

For COLMAP registration and localization issues, see:

- [COLMAP FAQ: Register / localize new images into an existing reconstruction](https://colmap.github.io/faq.html#register-localize-new-images-into-an-existing-reconstruction)

### NumPy

#### `ImportError: cannot import name 'bool' from 'numpy'`

Try:

```bash
pip install git+https://github.com/mattloper/chumpy
```

#### `floating point exception`

Try:

```bash
pip install numpy==1.23.1
```

You may also need:

```bash
pip install --force-reinstall charset-normalizer==3.1.0
```

#### `ValueError: numpy.dtype size changed`

Example error:

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject.
```

### Isaac Gym

Example `LD_LIBRARY_PATH` settings:

```bash
export LD_LIBRARY_PATH=/home/wenjiawang/miniconda3/envs/gym/lib/libpython3.8.so.1.0:/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=/home/wenjiawang/miniconda3/pkgs/python-3.8.20-he870216_0/lib/libpython3.8.so.1.0:/usr/lib/x86_64-linux-gnu
```
