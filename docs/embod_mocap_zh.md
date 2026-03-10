# EmbodMocap 主流程

**语言切换 / Language:** [中文](embod_mocap_zh.md) | [English](embod_mocap.md)

本项目统一入口为：`embod_mocap/run_stages.py`。

## 快速开始

运行主流程前，先下载所需资源：

```bash
# checkpoints
bash embod_mocap/tools/download_ckpt.sh

# SMPL 人体模型资源
bash embod_mocap/tools/download_body_models.sh

# demo 数据 + release_demo.xlsx
bash embod_mocap/tools/download_demo_data.sh
```

如果你需要完整 benchmark / 全量发布数据，再执行 `download_full_data.sh`。

```bash
bash embod_mocap/tools/download_full_data.sh
```

### 先查看我们提供的结果

如果你想在跑主流程前先浏览发布的 demo 结果，可以启动交互式 Viser 可视化：

```bash
cd embod_mocap
python tools/visualize_viser.py --xlsx ../datasets/release_demo.xlsx --data_root ../datasets/dataset_demo --stride 2 --scene_mesh simple --mesh_level 1
```


### 使用示例数据跑通主流程

你可以直接在我们提供的 demo 数据上运行主流程。

```bash
cd embod_mocap

# 使用提供的示例数据运行演示
python run_stages.py ../datasets/release_demo.xlsx --data_root ../datasets/dataset_demo --config config_fast.yaml --steps 1-15 --mode overwrite
```

演示数据包含预填充的同步索引（`v1_start`/`v2_start`），因此您可以直接运行完整流程。

### 从头开始

如果你要处理自己采集的数据，先把文件组织成下面这样：

```text
datasets/
└── my_capture/
    └── livingroom1/
        ├── calibration.json
        ├── data.jsonl
        ├── data.mov
        ├── frames2/
        ├── metadata.json
        ├── seq0/
        │   ├── recording_2026-01-25_17-51-07.zip
        │   └── recording_2026-01-25_17-51-08.zip
        └── seq1/
            ├── recording_2026-01-25_18-10-11.zip
            └── recording_2026-01-25_18-10-11(1).zip
```

保留原始 zip 文件名即可。后续脚本会自动解压，并将两个视角整理/重命名为 `raw1/` 和 `raw2/`。

这两个录制 zip 应该来自同一次采集，因此时间戳通常会非常接近，甚至可能相同。

时序对齐方面，我们采集时会用激光笔作为同步信号。在 step 6 之前，请先检查 `raw1/` 和 `raw2/` 中解压出来的图像，找到各自画面里激光光斑消失的那一帧，并将对应帧号填写到 xlsx 的 `v1_start` 和 `v2_start`。你也可以用类似的同步方案，例如手电筒、手机闪光灯等明显的瞬时光信号。

更详细的目录结构和所需文件说明，请直接看下面的 `文件布局` 小节。

建议在 `embod_mocap/` 目录中执行：

```bash
cd embod_mocap

# 1) 先自动生成 xlsx（步骤0，如果文件已存在会报错）
python run_stages.py seq_info.xlsx --data_root /path/to/data --steps 0

# 2) 先填写基础字段（in_door / vertical / FAILED 等）
#    `FAILED` 表示该序列已经过人工检查，并被判断为不可用

# 3) 先跑 scene 与前半预处理
python run_stages.py seq_info.xlsx --data_root /path/to/data --config config_fast.yaml --steps 1-5 --mode overwrite
```

然后人工检查两个视角，确定同步关系，并在 xlsx 中填写 `v1_start` / `v2_start`。

```bash
# 4) 对齐索引填写后，再继续主流程
python run_stages.py seq_info.xlsx --data_root /path/to/data --config config_fast.yaml --steps 6-15 --mode overwrite

# 5) 可选最终步骤：仅当标注了 `contacts` 时运行接触对齐
python run_stages.py seq_info.xlsx --data_root /path/to/data --config config_fast.yaml --steps 16 --mode overwrite

# 如果使用 `standard` 模式，将 `config_fast.yaml` 替换为 `config.yaml`
# `fast`：针对 mesh + motion 任务优化，迭代更快
# `standard`：保留/生成更完整的 RGBD + mask 资产，用于数据/模型训练
```

## 数据集下载

<details>
<summary><strong>展开数据下载说明</strong></summary>

对大多数用户来说，前面 Quick Start 里的下载脚本已经足够。

只有当你想手动下载发布文件时，再看下面这些链接：

- HuggingFace：[EmbodMocap_release](https://huggingface.co/datasets/WenjiaWang/EmbodMocap_release)
- OneDrive：[EmbodMocap OneDrive Data](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wwj2022_connect_hku_hk/IgAh_tLK24aLT61TePApWqk1AdpvlVBHvyttzmO61fegoC0?e=ikzCTO)

</details>

## 文件布局

<details>
<summary><strong>展开推荐文件布局</strong></summary>

以下是一个简单示例，展示如何在本仓库周围组织 checkpoints、body models 和 datasets：

```text
EmbodMocap/
├── checkpoints/
│   ├── vggt.pt
│   ├── sam2.1_hiera_large.pt
│   ├── sam2.1_hiera_small.pt
│   ├── vimo_checkpoint.pth.tar
│   ├── yolov8x.pt
│   ├── vitpose-h-multi-coco.pth
│   ├── vocab_tree_flickr100K_words32K.bin
│   └── vocab_tree_faiss_flickr100K_words1M.bin
├── body_models/
│   └── smpl/
│       ├── SMPL_NEUTRAL.pkl
│       ├── J_regressor_extra.npy
│       ├── J_regressor_h36m.npy
│       └── mesh_downsampling.npz
├── datasets/
│   └── my_capture/
│       └── livingroom1/
│           ├── calibration.json（原始文件）
│           ├── data.jsonl（原始文件）
│           ├── data.mov（原始文件）
│           ├── frames2/（原始文件）
│           ├── metadata.json（原始文件）
│           ├── seq0/
│           │   ├── recording_2026-01-25_17-51-07.zip（iPhone A 原始文件）
│           │   ├── recording_2026-01-25_17-51-08.zip（iPhone B 原始文件）
│           │   ├── raw1/
│           │   │   ├── data.mov（原始文件）
│           │   │   ├── data.jsonl（原始文件）
│           │   │   ├── calibration.json（原始文件）
│           │   │   ├── metadata.json（原始文件）
│           │   │   └── frames2/（原始文件）
│           │   ├── raw2/
│           │   │   └── ...（与 raw1 相同）
│           │   ├── v1/
│           │   │   ├── images/（Step 6 输出）
│           │   │   ├── depths/（Step 10 输出，standard 模式）
│           │   │   ├── depths_refined/（Step 10 输出，standard 模式）
│           │   │   └── masks/（Step 10 输出，standard 模式）
│           │   ├── v2/
│           │   │   └── ...（与 v1 相同）
│           │   └── optim_params.npz（Step 15 输出）
│           ├── seq1/
│           │   └── ...
│           ├── transforms.json（Step 1 输出）
│           └── mesh_simplified.ply（Step 2 输出）
└── embod_mocap/
```

建议用法：

- 将 checkpoints 放在 `checkpoints/`。
- 将 SMPL/SMPL-X body-model 资产放在 `body_models/`。
- 将捕获的 scene 放在 `datasets/` 下，并将该根目录传递给 `--data_root`。
- 对于自己采集的数据，只需要把两个 `recording_*.zip` 放进每个 `seq*` 文件夹；脚本会自动整理成 `raw1/` 和 `raw2/`。

</details>

### 多 GPU 处理

使用多 GPU 加速处理，使用 `run_stages_mp.py`：

```bash
cd embod_mocap

# 使用多个 GPU（例如 GPU 0, 1, 2）
python run_stages_mp.py seq_info.xlsx --data_root /path/to/data --config config.yaml --steps 1-15 --mode overwrite --gpu_ids 0,1,2
```

`run_stages_mp.py` 支持与 `run_stages.py` 相同的所有参数，另外还有：
- `--gpu_ids`：指定 GPU ID（例如 `0,1,2`）。当数量 > 1 时自动启用多 GPU 模式
- `--worker_poll_interval`：工作队列轮询间隔（秒），默认：1.0
- `--max_retries`：每个任务的最大重试次数，默认：1

多 GPU 版本会自动将 seq 分配到可用的 GPU 上进行并行处理。

## Standard / Fast 模式

两种模式共享相同的步骤定义。`standard` 和 `fast` 的主要区别在于输出完整度：

- `fast`：针对 mesh + motion 任务优化，迭代更快。
- `standard`：保留/生成更完整的 RGBD + mask 资产，用于数据/模型训练。

## 步骤总览（1-16）

<details>
<summary><strong>展开全部 16 个 Step</strong></summary>

### Stage 1：scene 重建

**Step 1: `sai`**

- 目标：scene 关键帧与相机初始化。
- 输出：`transforms.json`。
- 说明：`in_door`/`out_door` 策略影响关键帧间隔。

**Step 2: `recon_scene`**

- 目标：从 RGBD/深度线索重建 scene 网格。
- 输出：`mesh_raw.ply`、`mesh_simplified.ply`。
- 说明：体素/深度截断参数权衡质量与速度。

**Step 3: `rebuild_colmap`**

- 目标：重建 scene 的 COLMAP 数据库和模型，用于后续配准。
- 输出：`colmap/database.db`、稀疏模型文件。
- 说明：配准质量不稳定时检查 COLMAP 环境。

### Stage 2：seq 预处理

**Step 4: `get_frames`**

- 目标：从原始录制中提取每视角图像。
- 输出：`raw1/images`、`raw2/images`。

**Step 5: `smooth_camera`**

- 目标：平滑原始捕获的相机轨迹。
- 输出：平滑后的每视角相机文件。

**Step 6: `slice_views`**

- 目标：使用同步索引切分对齐的 `v1/v2` 片段。
- 输出：包含切片图像/相机的 `v1/` 和 `v2/` 文件夹。
- 说明：继续之前必须正确填写 `v1_start`/`v2_start`。

**Step 7: `process_smpl`**

- 目标：估计人体状态（SMPL/姿态相关中间结果）。
- 输出：视角侧 `smpl_params.npz`（中间文件）。

**Step 8: `colmap_human_cam`**

- 目标：将人体视角相机配准到 scene / 世界坐标系。
- 输出：`v1/cameras_colmap.npz`、`v2/cameras_colmap.npz`。

### Stage 3：几何与相机优化

**Step 9: `generate_keyframes`**

- 目标：构建优化关键帧集。
- 输出：关键帧索引/元数据。

**Step 10: `process_depth_mask`**

- 目标：生成/细化深度和人体掩码。
- 输出：`images/`、`depths_refined/`、`masks/`（策略取决于模式）。
- 说明：这是 `standard` 与 `fast` 完整度差异的主要部分。

**Step 11: `vggt_track`**

- 目标：为后续优化生成跟踪约束。
- 输出：跟踪元数据。

**Step 12: `align_cameras`**

- 目标：将 SAI/COLMAP 相机系统对齐到一致的尺度/坐标系。
- 输出：对齐的相机文件。

**Step 13: `unproj_human`**

- 目标：将人体视角几何反投影到点云。
- 输出：相机优化使用的点云文件。

**Step 14: `optim_human_cam`**

- 目标：优化人体视角相机轨迹。
- 输出：优化/对齐的相机参数（中间/最终）。

### Stage 4：运动与接触

**Step 15: `optim_motion`**

- 目标：优化世界坐标系运动参数。
- 核心输出：`optim_params.npz`。
- 说明：这是可视化和下游任务所需的关键产物。

**Step 16: `align_contact`**

- 目标：可选的接触感知全局对齐。
- 要求：XLSX 中有效的 `contacts`。
- 输出：`optim_params_aligned.npz`、对齐的相机/`kp3d` 产物。


</details>

## XLSX 关键字段

- `scene_folder`, `seq_name`
- `in_door`, `vertical`
- `v1_start`, `v2_start`
- `FAILED`
- `contacts`（仅 Step 16 使用）
- `skills`（导出 `plan.json` 时使用）

### `contacts`（Step 16）

只有当 `contacts` 有效时才执行 Step 16。若为空或 `nan`，该步骤自动跳过。

## 清理（--clean）

当前支持：`--clean standard|fast|all`，并要求提供 `xlsx + data_root`。

```bash
# 先预览理论删除列表（不真正删除）
python run_stages.py seq_info.xlsx --data_root /path/to/data --clean fast --clean_dry_run

# 真正执行
python run_stages.py seq_info.xlsx --data_root /path/to/data --clean fast
```

当前语义：

- `all`：只保留 raw 安全文件
- `fast`：保留核心结果 + 轻量可视化产物
- `standard`：在 `fast` 基础上额外保留 `images/depths_refined/masks`

## 完成检查

检查 xlsx 中每个 seq 的步骤完成情况：

```bash
python run_stages.py seq_info.xlsx --data_root /path/to/data --steps 1-15 --check --config config.yaml
```

该命令会扫描所有 seq 并报告每个步骤的完成状态，不会执行任何处理。适用于跟踪多个 scene 的进度。
