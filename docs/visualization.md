# Visualization Guide

**Language / 语言切换:** [English](visualization.md) | [中文](visualization_zh.md)

## 1) Video Visualization via `tools/visualize.py`

Generate visualization videos from processed sequences. This tool creates concatenated videos showing input frames, processed results with SMPL overlays, and optimized camera views.

**Single-sequence mode** (using `--seq_path`):

```bash
cd embod_mocap
python tools/visualize.py --seq_path /path/to/scene/seq0 --input --processed --optim_cam --downscale 2 --mode overwrite
```

**Batch mode** (using `--xlsx`):

```bash
cd embod_mocap
python tools/visualize.py --xlsx seq_info.xlsx --data_root /path/to/data --input --processed --optim_cam --downscale 2 --mode overwrite
```

Note: Choose either `--seq_path` or `--xlsx`, not both. In `--xlsx` mode, `--data_root` is required.

### Parameters:

**Input mode (choose one):**
- `--seq_path`: single sequence folder path
- `--xlsx`: xlsx manifest path (batch mode, skips FAILED rows by default)
- `--data_root`: required root path prefix to join with `scene_folder` from xlsx
- `--force_all`: include rows marked as `FAILED` in xlsx batch mode

**Visualization options (at least one required):**
- `--input`: visualize input frames (generates `concat_input.mp4`)
- `--processed`: visualize processed results with SMPL overlay (generates `concat_processed.mp4`)
- `--optim_cam`: visualize optimized camera view with SMPL rendering (generates `concat_optimized.mp4`)

**Other options:**
- `--device`: device to use (default: `cuda:0`)
- `--downscale`: downscale factor for visualization (default: `2`)
- `--mode`: `overwrite` or `skip` existing videos (default: `overwrite`)
- `--vis_chunk`: SMPL visualization chunk size (default: `60`)

### Output:

The tool generates MP4 videos in the sequence folder:
- `concat_input.mp4`: side-by-side view of v1 and v2 input frames
- `concat_processed.mp4`: input frames with SMPL pose overlay
- `concat_optimized.mp4`: rendered SMPL in optimized camera views

## 2) Interactive Visualization via `tools/visualize_viser.py`

Use Viser for interactive 3D browsing of scene mesh, SMPL motion, and cameras.

**Single-scene mode** (using `--scene_path`):

```bash
cd embod_mocap
python tools/visualize_viser.py --scene_path /path/to/scene --port 8080 --max_frames -1 --stride 2 --mesh_level 1 --scene_mesh simple
```

**Multi-scene mode** (using `--xlsx`):

```bash
cd embod_mocap
python tools/visualize_viser.py --xlsx seq_info.xlsx --data_root /path/to/data --port 8080 --max_frames -1 --stride 2 --mesh_level 1 --scene_mesh simple
```

Note: Choose either `--scene_path` or `--xlsx`, not both.

### Parameters:

**Input mode (choose one):**
- `--scene_path`: single scene folder path (contains seq* subfolders)
- `--xlsx`: xlsx manifest path (multi-scene mode, skips FAILED rows)
- `--data_root`: optional root path prefix to join with scene_folder from xlsx

**Visualization options:**
- `--port`: web UI port (default: `8080`)
- `--max_frames`: maximum frames to load per sequence; -1 means all frames (default: `-1`)
- `--stride`: frame sampling stride, e.g., 2 loads every 2nd frame (default: `1`)
- `--mesh_level`: SMPL mesh downsampling level - 0=full, 1=downsample (~1723 verts), 2=coarser (default: `1`)
- `--scene_mesh`: scene mesh mode - `simple`=prefer mesh_simplified.ply with fallback to mesh_raw.ply, `raw`=use mesh_raw.ply only, `no`=disable scene mesh (default: `simple`)
- `--hq`: enable high-quality rendering with multiple lights and shadows

### Usage:

- This script expects sequence-level outputs such as `optim_params.npz` and scene mesh (`mesh_simplified.ply` or `mesh_raw.ply`).
- Open your browser to the printed local URL after startup.
- Use the web interface to switch between scenes/sequences and control playback.
