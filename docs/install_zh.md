# 安装说明（EmbodMocap）

**语言切换 / Language:** [中文](install_zh.md) | [English](install.md)

本文覆盖开源主流程所需的安装步骤，以及必要的依赖 / checkpoints 配置。

## 1）克隆仓库

```bash
git clone --recurse-submodules https://github.com/WenjiaWang0312/EmbodMocap
cd EmbodMocap
```

如果已克隆但还没有拉取 submodule：

```bash
git submodule update --init --recursive
```

## 2）创建 Python 环境

```bash
conda create -n embodmocap python=3.11 -y
conda activate embodmocap
```

根据你的 CUDA 运行时安装 PyTorch。下面给出两个示例：

```bash
# CUDA 12.4 示例
pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124

# CUDA 12.8 示例
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

## 3）安装核心依赖

```bash
pip install -r requirements.txt
pip install -e embod_mocap
```

## 4）第三方模块（Submodule）

第三方依赖通过 Git submodule 管理，而不是直接内置在仓库中。

这些 submodule 可以通过以下命令添加：

```bash
cd embodmocap
git submodule add https://github.com/luca-medeiros/lang-segment-anything thirdparty/lang_sam
git submodule add https://github.com/Robbyant/lingbot-depth thirdparty/lingbot_depth
git submodule add https://github.com/ViTAE-Transformer/ViTPose thirdparty/ViTPose
```

常见模块：

- `embod_mocap/thirdparty/lingbot_depth`
- `embod_mocap/thirdparty/lang_sam`
- `embod_mocap/thirdparty/ViTPose`

如果需要可编辑安装：

```bash
pip install -e embod_mocap/thirdparty/lingbot_depth
pip install -e embod_mocap/thirdparty/lang_sam
pip install -e embod_mocap/thirdparty/ViTPose
```

## 5）COLMAP

COLMAP 安装指南：

- [COLMAP 安装指南](https://colmap.github.io/install.html)

在我们的环境里，直接使用系统包管理器安装 COLMAP 效果良好：

```bash
sudo apt install colmap
```

## 6）其他依赖

### [torch-scatter](https://github.com/rusty1s/pytorch_scatter)（部分训练 / 评估路径需要）

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### [pytorch3d](https://github.com/facebookresearch/pytorch3d) / 渲染栈（可选，用于相机空间渲染）

请选择与你的 CUDA / PyTorch 匹配的 pytorch3d 包。例如，你可以从[清华镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/)下载对应的包，然后再进行本地安装。本项目使用的渲染封装为 [torch3d_render](https://github.com/WenjiaWang0312/torch3d_render.git)。

```bash
conda install -c iopath iopath
conda install -c bottler nvidiacub
conda install --use-local xxx.tar.bz2
pip install git+https://github.com/WenjiaWang0312/torch3d_render.git
```

### 自定义下载（可选）

如果你不想使用打包好的下载脚本，而是希望手动选择单个文件下载，可以参考下面这些原始 checkpoint 与 COLMAP vocab tree 链接：

- [VGGT 模型权重](https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt)
- [SAM2.1 Hiera Large 权重](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
- [SAM2.1 Hiera Small 权重](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [VIMO 权重（Google Drive）](https://drive.google.com/file/d/1fdeUxn_hK4ERGFwuksFpV_-_PHZJuoiW/view?usp=share_link)
- [YOLOv8x 权重（Google Drive）](https://drive.google.com/uc?id=1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo&export=download&confirm=t)
- [ViTPose-H Multi-COCO 权重（Google Drive）](https://drive.google.com/uc?id=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso&export=download&confirm=t)
- [COLMAP releases 页面](https://github.com/colmap/colmap/releases/)（vocab tree 文件请自行查找）

## 7）排障

### COLMAP

#### 安装

- [COLMAP issue #2464](https://github.com/colmap/colmap/issues/2464)

#### `No CMAKE_CUDA_COMPILER could be found`

- [jetsonhacks/buildLibrealsense2TX issue #13](https://github.com/jetsonhacks/buildLibrealsense2TX/issues/13)

#### `FAILED: src/colmap/mvs/CMakeFiles/xxx`

- [COLMAP issue #2091](https://github.com/colmap/colmap/issues/2091)

#### `libcudart.so` 错误

- [vllm issue #1369](https://github.com/vllm-project/vllm/issues/1369)
- 示例：

```bash
export LD_LIBRARY_PATH=/home/wwj/miniconda3/envs/droidenv/lib/:$LD_LIBRARY_PATH
```

#### 配准问题

关于 COLMAP 配准与定位问题，参考：

- [COLMAP FAQ：向已有重建中注册 / 定位新图像](https://colmap.github.io/faq.html#register-localize-new-images-into-an-existing-reconstruction)

### NumPy

#### `ImportError: cannot import name 'bool' from 'numpy'`

尝试：

```bash
pip install git+https://github.com/mattloper/chumpy
```

#### `floating point exception`

尝试：

```bash
pip install numpy==1.26.4
```

你可能还需要：

```bash
pip install --force-reinstall charset-normalizer==3.1.0
```

#### `ValueError: numpy.dtype size changed`

错误示例：

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject.
```

推荐版本：

```ini
numpy==1.26.4
```

### Isaac Gym

`LD_LIBRARY_PATH` 设置示例：

```bash
export LD_LIBRARY_PATH=/home/wenjiawang/miniconda3/envs/gym/lib/libpython3.8.so.1.0:/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=/home/wenjiawang/miniconda3/pkgs/python-3.8.20-he870216_0/lib/libpython3.8.so.1.0:/usr/lib/x86_64-linux-gnu
```
