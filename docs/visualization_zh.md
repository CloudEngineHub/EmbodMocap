# 可视化说明

**语言切换 / Language:** [中文](visualization_zh.md) | [English](visualization.md)

## 1）通过 `tools/visualize.py` 生成视频可视化

从已处理完成的序列生成可视化视频。该工具会生成拼接视频，展示输入帧、带 SMPL 叠加的处理结果，以及优化后的相机视角。

**单序列模式**（使用 `--seq_path`）：

```bash
cd embod_mocap
python tools/visualize.py --seq_path /path/to/scene/seq0 --input --processed --optim_cam --downscale 2 --mode overwrite
```

**批量模式**（使用 `--xlsx`）：

```bash
cd embod_mocap
python tools/visualize.py --xlsx seq_info.xlsx --data_root /path/to/data --input --processed --optim_cam --downscale 2 --mode overwrite
```

注意：`--seq_path` 和 `--xlsx` 二选一，不能同时使用。在 `--xlsx` 模式下，`--data_root` 是必填项。

### 参数：

**输入模式（选择其一）：**
- `--seq_path`：单个序列文件夹路径
- `--xlsx`：xlsx 清单路径（批量模式，默认跳过标记为 `FAILED` 的行）
- `--data_root`：必填根路径，用于与 xlsx 中的 `scene_folder` 拼接
- `--force_all`：在 xlsx 批量模式下，包含标记为 `FAILED` 的行

**可视化选项（至少选择一个）：**
- `--input`：可视化输入帧（生成 `concat_input.mp4`）
- `--processed`：可视化带 SMPL 叠加的处理结果（生成 `concat_processed.mp4`）
- `--optim_cam`：可视化优化相机视角下的 SMPL 渲染（生成 `concat_optimized.mp4`）

**其他选项：**
- `--device`：使用的设备（默认：`cuda:0`）
- `--downscale`：可视化缩放比例（默认：`2`）
- `--mode`：`overwrite` 或 `skip`，控制是否覆盖已有视频（默认：`overwrite`）
- `--vis_chunk`：SMPL 可视化分块大小（默认：`60`）

### 输出：

该工具会在序列目录下生成 MP4 视频：
- `concat_input.mp4`：`v1` 和 `v2` 输入帧的左右拼接视图
- `concat_processed.mp4`：带 SMPL 姿态叠加的输入帧
- `concat_optimized.mp4`：优化相机视角下渲染的 SMPL 结果

## 2）通过 `tools/visualize_viser.py` 进行交互式可视化

使用 Viser 在 3D 中交互浏览场景网格、SMPL 动作与相机。

**单场景模式**（使用 `--scene_path`）：

```bash
cd embod_mocap
python tools/visualize_viser.py --scene_path /path/to/scene --port 8080 --max_frames -1 --stride 2 --mesh_level 1 --scene_mesh simple
```

**多场景模式**（使用 `--xlsx`）：

```bash
cd embod_mocap
python tools/visualize_viser.py --xlsx seq_info.xlsx --data_root /path/to/data --port 8080 --max_frames -1 --stride 2 --mesh_level 1 --scene_mesh simple
```

注意：`--scene_path` 和 `--xlsx` 二选一，不能同时使用。

### 参数：

**输入模式（选择其一）：**
- `--scene_path`：单个场景文件夹路径（包含 `seq*` 子文件夹）
- `--xlsx`：xlsx 清单路径（多场景模式，跳过 `FAILED` 行）
- `--data_root`：可选根路径前缀，用于与 xlsx 中的 `scene_folder` 拼接

**可视化选项：**
- `--port`：Web UI 端口（默认：`8080`）
- `--max_frames`：每个序列加载的最大帧数；`-1` 表示加载全部帧（默认：`-1`）
- `--stride`：帧采样步长，例如 `2` 表示每隔 2 帧加载一帧（默认：`1`）
- `--mesh_level`：SMPL 网格降采样级别 - 0=完整，1=降采样（约 1723 顶点），2=更粗（默认：`1`）
- `--scene_mesh`：场景网格模式 - `simple`=优先使用 `mesh_simplified.ply`，否则回退到 `mesh_raw.ply`；`raw`=仅使用 `mesh_raw.ply`；`no`=禁用场景网格（默认：`simple`）
- `--hq`：启用高质量渲染，包含多光源和阴影

### 使用说明：

- 该脚本依赖序列级输出（例如 `optim_params.npz`）以及场景网格（`mesh_simplified.ply` 或 `mesh_raw.ply`）。
- 启动后，根据终端打印的本地 URL 在浏览器中打开页面。
- 可在网页界面中切换场景 / 序列并控制播放。
