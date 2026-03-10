"""
使用 viser 可视化 SMPL 运动 + 场景网格
支持时序拖动、相机显示、场景静态显示

使用方法：
    # 单场景模式（加载场景下的 seq*）
    python visualize_viser.py --scene_path path/to/scene
    
    # 指定端口
    python visualize_viser.py --scene_path path/to/scene --port 8888
    
    # 多场景模式（从 xlsx 清单读取）
    python visualize_viser.py --xlsx seq_info.xlsx --data_root /path/to/data
    
    # 使用步长（每 5 帧加载一帧）
    python visualize_viser.py --scene_path path/to/scene --stride 5
    
    # 组合使用
    python visualize_viser.py --xlsx seq_info.xlsx --data_root /path/to/data --max_frames 500 --stride 10 --port 8080

参数说明：
    --scene_path: 单场景文件夹路径（包含 seq* 子文件夹）
    --xlsx: xlsx 清单路径（多场景模式）
    --data_root: 与 xlsx 中 scene_folder 拼接的可选根目录
    --port: viser 服务器端口（默认 8080）
    --max_frames: 最大加载帧数（默认加载所有帧）
    --stride: 帧步长（默认 1，即每帧都加载）
    
示例：
    # 加载单场景
    python visualize_viser.py --scene_path datasets/dataset_raw/0505_capture/0505apartment1
    
    # 快速预览：每 10 帧加载一帧
    python visualize_viser.py --scene_path datasets/dataset_raw/0505_capture/0505apartment1 --stride 10
"""
import argparse
import sys
import numpy as np
import viser
import viser.transforms as tf
import time
import trimesh
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embod_mocap.utils.mesh_sampler import SMPLMeshSampler


def load_smpl_model():
    """加载 SMPL 模型"""
    import sys
    from pathlib import Path
    
    # 添加项目根目录到 sys.path
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    from embod_mocap.human.configs import BMODEL
    from embod_mocap.human.smpl import SMPL
    
    body_model = SMPL(
        model_path=BMODEL.FLDR,
        gender='neutral',
        extra_joints_regressor=BMODEL.JOINTS_REGRESSOR_EXTRA,
        create_transl=False,
    )
    return body_model


def load_motion_data(optim_params_path):
    """只加载序列的 optim_params.npz（不包含场景网格）。"""
    optim_params = np.load(optim_params_path, allow_pickle=True)

    transl = optim_params['transl']
    global_orient = optim_params['global_orient']
    body_pose = optim_params['body_pose']
    betas = optim_params['betas']

    K1 = optim_params['K1']
    K2 = optim_params['K2']
    R1 = optim_params['R1']
    R2 = optim_params['R2']
    T1 = optim_params['T1']
    T2 = optim_params['T2']

    return {
        'transl': transl,
        'global_orient': global_orient,
        'body_pose': body_pose,
        'betas': betas,
        'K1': K1,
        'K2': K2,
        'R1': R1,
        'R2': R2,
        'T1': T1,
        'T2': T2,
        'num_frames': len(transl),
    }


def resolve_scene_mesh_path(scene_path, scene_mesh='simple'):
    """根据配置选择场景网格路径。"""
    scene_path = Path(scene_path)

    if scene_mesh == 'no':
        return None
    if scene_mesh == 'raw':
        mesh_path = scene_path / 'mesh_raw.ply'
        return mesh_path if mesh_path.exists() else None
    if scene_mesh == 'simple':
        mesh_simplified_path = scene_path / 'mesh_simplified.ply'
        if mesh_simplified_path.exists():
            return mesh_simplified_path
        mesh_raw_path = scene_path / 'mesh_raw.ply'
        if mesh_raw_path.exists():
            print(f"Warning: {mesh_simplified_path} not found, using {mesh_raw_path.name}")
            return mesh_raw_path
        return None
    raise ValueError(f"Unsupported scene_mesh mode: {scene_mesh}")


def load_scene_data(mesh_path):
    """加载场景网格。"""
    mesh_path = Path(mesh_path)

    print(f"Loading mesh from: {mesh_path}")
    scene_mesh = trimesh.load(str(mesh_path))

    vertices = scene_mesh.vertices
    faces = scene_mesh.faces

    if hasattr(scene_mesh.visual, 'vertex_colors') and scene_mesh.visual.vertex_colors is not None:
        vertex_colors = scene_mesh.visual.vertex_colors[:, :3]
        print(f"Loaded mesh with {len(vertices)} vertices and vertex colors")
    else:
        vertex_colors = None
        print(f"Loaded mesh with {len(vertices)} vertices (no colors)")

    return {
        'scene_vertices': vertices,
        'scene_faces': faces,
        'scene_colors': vertex_colors,
    }


def load_data(seq_path, mesh_path=None, no_scene=False):
    """兼容旧调用：加载 optim_params.npz + 可选场景网格。"""
    motion = load_motion_data(seq_path)
    if no_scene or mesh_path is None:
        scene = {'scene_vertices': None, 'scene_faces': None, 'scene_colors': None}
    else:
        scene = load_scene_data(mesh_path)
    motion.update(scene)
    return motion


def discover_seq_optim(scene_path: Path):
    """发现 scene 下所有含 optim_params.npz 的 seq* 目录。"""
    seq_items = []

    for child in scene_path.iterdir():
        if not child.is_dir() or not child.name.startswith('seq'):
            continue

        optim_path = child / 'optim_params.npz'
        if optim_path.exists():
            seq_items.append((child.name, optim_path))

    def _sort_key(item):
        name = item[0]
        suffix = name[3:]
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, name)

    seq_items.sort(key=_sort_key)
    return seq_items


def get_bool_from_excel(row, column, default=False):
    value = row.get(column, default)
    if value is None:
        return default
    try:
        import pandas as pd

        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value).upper() in ["TRUE", "1.0", "1", "YES"]


def build_manifest_from_scene(scene_path: Path):
    seq_items = discover_seq_optim(scene_path)
    if not seq_items:
        return {}
    scene_key = str(scene_path)
    return {
        scene_key: {
            "scene_path": scene_path,
            "scene_label": scene_path.name,
            "seq_items": seq_items,
        }
    }


def build_manifest_from_xlsx(xlsx_path, data_root=None):
    import pandas as pd

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"xlsx not found: {xlsx_path}")

    xl = pd.ExcelFile(xlsx_path)
    if len(xl.sheet_names) > 1:
        dfs = [pd.read_excel(xlsx_path, sheet_name=sn) for sn in xl.sheet_names]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_excel(xlsx_path)

    manifest = {}
    filtered_failed = 0
    missing_optim = 0
    missing_seq_dir = 0

    for _, row in df.iterrows():
        scene_rel = str(row.get("scene_folder", "")).strip()
        seq_name = str(row.get("seq_name", "")).strip()
        if not scene_rel or not seq_name:
            continue

        if get_bool_from_excel(row, "FAILED", default=False):
            filtered_failed += 1
            continue

        scene_path = Path(data_root) / scene_rel if data_root else Path(scene_rel)
        seq_path = scene_path / seq_name
        optim_path = seq_path / "optim_params.npz"

        if not seq_path.exists():
            missing_seq_dir += 1
            continue
        if not optim_path.exists():
            missing_optim += 1
            continue

        scene_key = scene_rel
        if scene_key not in manifest:
            manifest[scene_key] = {
                "scene_path": scene_path,
                "scene_label": scene_rel,
                "seq_items": [],
            }
        manifest[scene_key]["seq_items"].append((seq_name, optim_path))

    def _seq_sort_key(item):
        name = item[0]
        suffix = name[3:]
        if name.startswith("seq") and suffix.isdigit():
            return (0, int(suffix))
        return (1, name)

    for scene_key in list(manifest.keys()):
        uniq = {}
        for seq_name, optim_path in manifest[scene_key]["seq_items"]:
            uniq[seq_name] = optim_path
        seq_items = sorted(uniq.items(), key=_seq_sort_key)
        if seq_items:
            manifest[scene_key]["seq_items"] = seq_items
        else:
            del manifest[scene_key]

    print(f"Manifest from xlsx: scenes={len(manifest)}")
    print(f"Filtered FAILED rows: {filtered_failed}")
    if missing_seq_dir > 0:
        print(f"Skipped missing sequence dirs: {missing_seq_dir}")
    if missing_optim > 0:
        print(f"Skipped missing optim_params.npz: {missing_optim}")
    return manifest


def compute_smpl_vertices(body_model, transl, global_orient, body_pose, betas):
    """计算 SMPL 顶点"""
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    body_model = body_model.to(device)
    
    num_frames = len(transl)
    
    # 转换为 torch tensor
    transl = torch.from_numpy(transl).float().to(device)  # [T, 3]
    global_orient = torch.from_numpy(global_orient).float().to(device)  # [T, 1, 3, 3]
    body_pose = torch.from_numpy(body_pose).float().to(device)  # [T, 23, 3, 3]
    betas = torch.from_numpy(betas).float().to(device)  # [1, 10] or [T, 10]
    
    # 如果 betas 只有一个，扩展到所有帧
    if betas.shape[0] == 1:
        betas = betas.repeat(num_frames, 1)  # [T, 10]
    
    # Reshape
    transl = transl.view(-1, 3)
    global_orient = global_orient.view(-1, 1, 3, 3)
    body_pose = body_pose.view(-1, 23, 3, 3)
    
    # 批量计算所有帧
    with torch.no_grad():
        output = body_model(
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            pose2rot=False,
        )
        vertices = output.vertices.cpu().numpy()
    
    return vertices, body_model.faces


class MotionSceneViewer:
    def __init__(
        self,
        manifest,
        body_model,
        port=8080,
        max_frames=-1,
        stride=1,
        high_quality=False,
        scene_mesh='simple',
        mesh_level=1,
    ):
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.port = port

        self.manifest = manifest
        self.scene_keys = sorted(self.manifest.keys())

        self.body_model = body_model
        self.max_frames = max_frames
        self.stride = stride
        self.high_quality = high_quality
        self.scene_mesh = scene_mesh
        self.mesh_level = mesh_level

        self.current_scene_key = None
        self.scene_path = None
        self.scene_name = None
        self.seq_items = []
        self.seq_map = {}

        self.scene_data = {
            "scene_vertices": None,
            "scene_faces": None,
            "scene_colors": None,
        }
        self.scene_static_handles = []

        self.data = None
        self.frame_indices = []
        self.num_frames = 0
        self.current_seq = None

        self.smpl_vertices = None
        self.smpl_faces = None

        self.frame_nodes = []
        self.mesh_handles = []
        self.cam1_handle = None
        self.cam2_handle = None
        self.is_playing = False
        self.is_switching = False
        self.pending_scene_key = None
        self.pending_seq_name = None
        self.suppress_gui_events = False
        self.enforce_zero_frame = False
        self.enforce_zero_frame_until = 0.0

        self.default_camera_position = np.array([0.0, -2.0, 1.5])
        self.default_look_at = np.array([0.0, 0.0, 1.0])

        self.mesh_sampler = None
        if self.mesh_level > 0:
            self.mesh_sampler = SMPLMeshSampler()
            if self.mesh_level >= self.mesh_sampler.num_levels:
                raise ValueError(
                    f"mesh_level={self.mesh_level} out of range, "
                    f"supported=[0, {self.mesh_sampler.num_levels - 1}]"
                )

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = self.default_camera_position
            client.camera.look_at = self.default_look_at

        self.setup_gui()
        self.setup_dataset_selectors()
        self.frames_root = self.server.scene.add_frame("/frames", show_axes=False)

        self.load_scene(self.scene_keys[0], reset_timestep=True, force=True)

    def _compose_data(self, motion_data):
        data = dict(motion_data)
        data.update(self.scene_data)
        return data

    def _compute_frame_indices(self, total_frames):
        max_frames = total_frames if self.max_frames == -1 else min(self.max_frames, total_frames)
        stride = max(1, self.stride)
        return list(range(0, max_frames, stride))

    def _set_default_camera_from_motion(self):
        transl = self.data["transl"][self.frame_indices]
        transl_min = transl.min(axis=0)
        transl_max = transl.max(axis=0)
        transl_center = (transl_min + transl_max) / 2
        transl_range = np.linalg.norm(transl_max - transl_min)

        print(f"Human motion bounds: min={transl_min}, max={transl_max}")
        print(f"Human motion center: {transl_center}")
        print(f"Human motion range: {transl_range}")

        camera_distance = max(transl_range * 1.2, 1.0)
        camera_height = max(transl_range * 0.8, 0.8)

        self.default_camera_position = np.array(
            [transl_center[0], transl_center[1] - camera_distance, transl_center[2] + camera_height]
        )
        self.default_look_at = transl_center
        print(f"Setting camera at: {self.default_camera_position}, looking at: {self.default_look_at}")

    def _set_scene_context(self, scene_key):
        item = self.manifest[scene_key]
        self.current_scene_key = scene_key
        self.scene_path = Path(item["scene_path"])
        self.scene_name = item["scene_label"]
        self.seq_items = item["seq_items"]
        self.seq_map = {name: path for name, path in self.seq_items}

    def _refresh_selectors(self, selected_seq=None):
        self.suppress_gui_events = True
        try:
            self.gui_scene_selector.value = self.current_scene_key
            self.gui_current_scene.value = self.scene_name

            seq_options = [name for name, _ in self.seq_items]
            self.gui_seq_selector.options = seq_options

            if selected_seq is None or selected_seq not in self.seq_map:
                selected_seq = seq_options[0]

            self.gui_seq_selector.value = selected_seq
            self.gui_current_seq.value = selected_seq
        finally:
            self.suppress_gui_events = False

    def queue_scene_switch(self, scene_key):
        if scene_key not in self.manifest:
            return
        self.pending_scene_key = scene_key
        self.pending_seq_name = None

    def queue_sequence_switch(self, seq_name):
        if seq_name not in self.seq_map:
            return
        self.pending_seq_name = seq_name

    def clear_scene_static_nodes(self):
        for handle in self.scene_static_handles:
            handle.remove()
        self.scene_static_handles = []

    def load_scene(self, scene_key, reset_timestep=True, force=False):
        if scene_key not in self.manifest:
            print(f"Error: scene not found: {scene_key}")
            return
        if not force and scene_key == self.current_scene_key:
            return

        self.is_playing = False
        self.gui_play_pause.name = "Play"

        self.clear_dynamic_nodes()
        self.clear_scene_static_nodes()

        self._set_scene_context(scene_key)
        self.current_seq = None
        self.scene_data = {
            "scene_vertices": None,
            "scene_faces": None,
            "scene_colors": None,
        }

        mesh_path = resolve_scene_mesh_path(self.scene_path, self.scene_mesh)
        if mesh_path is not None:
            self.scene_data = load_scene_data(mesh_path)
            self.add_scene_mesh()
            self.add_grid_floor()
            if self.high_quality:
                self.setup_lighting()
        elif self.scene_mesh != 'no':
            print(f"Warning: scene mesh not found in {self.scene_path}, loading without mesh.")

        first_seq = self.seq_items[0][0]
        self._refresh_selectors(selected_seq=first_seq)
        self.load_sequence(first_seq, reset_timestep=reset_timestep, force=True)
        print(f"Switched scene to: {self.scene_name}")

    def load_sequence(self, seq_name, reset_timestep=False, force=False):
        if seq_name not in self.seq_map:
            print(f"Error: sequence not found: {seq_name}")
            return
        if not force and seq_name == self.current_seq:
            return

        optim_path = self.seq_map[seq_name]
        print(f"Loading motion from: {optim_path}")

        motion_data = load_motion_data(str(optim_path))
        self.data = self._compose_data(motion_data)

        self.frame_indices = self._compute_frame_indices(self.data["num_frames"])
        self.num_frames = len(self.frame_indices)
        if self.num_frames == 0:
            print(f"Warning: {seq_name} has no frames after sampling, fallback to first frame")
            self.frame_indices = [0]
            self.num_frames = 1

        print(
            f"[{self.scene_name}/{seq_name}] Loading {self.num_frames} frames "
            f"(max_frames={self.max_frames}, stride={self.stride}, total={self.data['num_frames']})"
        )

        print("Computing SMPL vertices...")
        self.smpl_vertices, self.smpl_faces = compute_smpl_vertices(
            self.body_model,
            self.data["transl"][self.frame_indices],
            self.data["global_orient"][self.frame_indices],
            self.data["body_pose"][self.frame_indices],
            self.data["betas"],
        )
        if self.mesh_sampler is not None and self.mesh_level > 0:
            num_v_before = int(self.smpl_vertices.shape[1])
            self.smpl_vertices = self.mesh_sampler.downsample(
                self.smpl_vertices,
                from_level=0,
                to_level=self.mesh_level,
            )
            ds_faces = self.mesh_sampler.get_faces(level=self.mesh_level)
            if ds_faces is not None:
                self.smpl_faces = ds_faces
            num_v_after = int(self.smpl_vertices.shape[1])
            print(
                f"Downsampled SMPL mesh with mesh_level={self.mesh_level}: "
                f"{num_v_before} -> {num_v_after} vertices"
            )
        print(f"Computed {len(self.smpl_vertices)} frames")

        self.clear_dynamic_nodes()
        self.create_frames()

        self.current_seq = seq_name
        self._set_default_camera_from_motion()

        self.refresh_timeline(reset_timestep=reset_timestep)
        if reset_timestep:
            self.enforce_zero_frame = True
            self.enforce_zero_frame_until = time.time() + 1.0
            self.gui_timestep.value = 0.0
        self.update_frame(0.0 if reset_timestep else self.gui_timestep.value)
        self.gui_current_seq.value = seq_name
        if self.gui_seq_selector.value != seq_name:
            self.suppress_gui_events = True
            try:
                self.gui_seq_selector.value = seq_name
            finally:
                self.suppress_gui_events = False
        print(f"Switched sequence to: {seq_name}")

    def refresh_timeline(self, reset_timestep=False):
        actual_max_frame = max(0, self.num_frames - 1)
        safe_slider_max = max(1, actual_max_frame)
        try:
            self.gui_timestep.max = float(safe_slider_max)
        except Exception:
            pass

        if reset_timestep:
            self.gui_timestep.value = 0.0
        else:
            current_value = self.gui_timestep.value
            if current_value is None or not np.isfinite(current_value):
                current_value = 0.0
            self.gui_timestep.value = float(min(current_value, float(actual_max_frame)))

        # Single-frame sequence: keep slider in a safe range but disable manual stepping.
        single_frame = self.num_frames <= 1
        self.gui_timestep.disabled = self.is_playing or single_frame

    def clear_dynamic_nodes(self):
        for frame_node in self.frame_nodes:
            frame_node.remove()
        self.frame_nodes = []
        # /frames/{i} owns /frames/{i}/smpl children. Avoid double remove warnings.
        self.mesh_handles = []

        if self.cam1_handle is not None:
            self.cam1_handle.remove()
            self.cam1_handle = None
        if self.cam2_handle is not None:
            self.cam2_handle.remove()
            self.cam2_handle = None

    def add_scene_mesh(self):
        vertices = self.scene_data["scene_vertices"]
        faces = self.scene_data["scene_faces"]
        vertex_colors = self.scene_data["scene_colors"]

        print(f"Adding scene mesh with {len(vertices)} vertices and {len(faces)} faces")
        if vertex_colors is not None:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.visual.vertex_colors = vertex_colors
            handle = self.server.scene.add_mesh_trimesh(
                name="/scene/mesh",
                mesh=mesh,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
            )
            print("Scene mesh loaded with vertex colors")
        else:
            handle = self.server.scene.add_mesh_simple(
                name="/scene/mesh",
                vertices=vertices,
                faces=faces,
                color=(200, 200, 200),
                wireframe=False,
            )
            print("Scene mesh loaded with default color")
        self.scene_static_handles.append(handle)

    def add_grid_floor(self):
        scene_vertices = self.scene_data["scene_vertices"]
        floor_z = scene_vertices[:, 2].min()

        scene_x_min = scene_vertices[:, 0].min()
        scene_x_max = scene_vertices[:, 0].max()
        scene_y_min = scene_vertices[:, 1].min()
        scene_y_max = scene_vertices[:, 1].max()

        scene_x_range = scene_x_max - scene_x_min
        scene_y_range = scene_y_max - scene_y_min
        x_margin = scene_x_range * 0.2
        y_margin = scene_y_range * 0.2

        x_min = scene_x_min - x_margin
        x_max = scene_x_max + x_margin
        y_min = scene_y_min - y_margin
        y_max = scene_y_max + y_margin
        grid_spacing = 0.5

        x_min = np.floor(x_min / grid_spacing) * grid_spacing
        x_max = np.ceil(x_max / grid_spacing) * grid_spacing
        y_min = np.floor(y_min / grid_spacing) * grid_spacing
        y_max = np.ceil(y_max / grid_spacing) * grid_spacing
        print(f"Grid floor range: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

        x_lines = np.arange(x_min, x_max + grid_spacing, grid_spacing)
        for x in x_lines:
            start = np.array([x, y_min, floor_z])
            end = np.array([x, y_max, floor_z])
            handle = self.server.scene.add_spline_catmull_rom(
                name=f"/grid/x_{x:.2f}",
                positions=np.array([start, end]),
                color=(150, 150, 150),
                line_width=1.0,
            )
            self.scene_static_handles.append(handle)

        y_lines = np.arange(y_min, y_max + grid_spacing, grid_spacing)
        for y in y_lines:
            start = np.array([x_min, y, floor_z])
            end = np.array([x_max, y, floor_z])
            handle = self.server.scene.add_spline_catmull_rom(
                name=f"/grid/y_{y:.2f}",
                positions=np.array([start, end]),
                color=(150, 150, 150),
                line_width=1.0,
            )
            self.scene_static_handles.append(handle)
        print(f"Added grid floor at z={floor_z:.2f} (scene min z)")

    def setup_lighting(self):
        print("Setting up high-quality lighting and shadows...")
        self.server.scene.configure_default_lights(enabled=True, cast_shadow=True)

        scene_vertices = self.scene_data["scene_vertices"]
        scene_center = scene_vertices.mean(axis=0)
        scene_min = scene_vertices.min(axis=0)
        scene_max = scene_vertices.max(axis=0)
        scene_size = np.linalg.norm(scene_max - scene_min)

        main_light_dir = scene_center + np.array([scene_size * 0.3, -scene_size * 0.3, scene_size * 0.8])
        handle = self.server.scene.add_light_directional(
            name="/lighting/main",
            color=(255, 250, 240),
            intensity=2.0,
            cast_shadow=True,
            wxyz=self.compute_light_direction(main_light_dir, scene_center),
        )
        self.scene_static_handles.append(handle)

        fill_light_dir = scene_center + np.array([-scene_size * 0.5, scene_size * 0.3, scene_size * 0.5])
        handle = self.server.scene.add_light_directional(
            name="/lighting/fill",
            color=(200, 220, 255),
            intensity=0.8,
            cast_shadow=False,
            wxyz=self.compute_light_direction(fill_light_dir, scene_center),
        )
        self.scene_static_handles.append(handle)

        light_height = scene_max[2] - 0.5
        handle = self.server.scene.add_light_point(
            name="/lighting/point1",
            position=(scene_center[0], scene_center[1], light_height),
            color=(255, 245, 230),
            intensity=16.0,
            cast_shadow=True,
        )
        self.scene_static_handles.append(handle)

        handle = self.server.scene.add_light_point(
            name="/lighting/point2",
            position=(scene_min[0] + scene_size * 0.3, scene_min[1] + scene_size * 0.3, light_height),
            color=(255, 240, 220),
            intensity=12.0,
            cast_shadow=True,
        )
        self.scene_static_handles.append(handle)

        handle = self.server.scene.add_light_point(
            name="/lighting/point3",
            position=(scene_max[0] - scene_size * 0.3, scene_max[1] - scene_size * 0.3, light_height),
            color=(255, 240, 220),
            intensity=12.0,
            cast_shadow=True,
        )
        self.scene_static_handles.append(handle)
        print("High-quality lighting configured: 2 directional lights + 3 point lights with shadows")

    def compute_light_direction(self, light_pos, target_pos):
        direction = target_pos - light_pos
        direction = direction / np.linalg.norm(direction)
        default_dir = np.array([0, 0, -1])
        v = np.cross(default_dir, direction)
        s = np.linalg.norm(v)
        c = np.dot(default_dir, direction)

        if s < 1e-6:
            if c > 0:
                return (1.0, 0.0, 0.0, 0.0)
            return (0.0, 1.0, 0.0, 0.0)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        return tf.SO3.from_matrix(R).wxyz

    def apply_visibility(self):
        show_smpl = self.gui_show_smpl.value if self.gui_show_smpl is not None else True
        show_cameras = self.gui_show_cameras.value if self.gui_show_cameras is not None else True

        for mesh_handle in self.mesh_handles:
            mesh_handle.visible = show_smpl
        if self.cam1_handle is not None:
            self.cam1_handle.visible = show_cameras
        if self.cam2_handle is not None:
            self.cam2_handle.visible = show_cameras

    def setup_dataset_selectors(self):
        with self.server.gui.add_folder("Dataset"):
            self.gui_scene_selector = self.server.gui.add_dropdown(
                "Scene",
                options=self.scene_keys,
                initial_value=self.scene_keys[0],
            )
            self.gui_current_scene = self.server.gui.add_text(
                "Current Scene",
                initial_value=self.manifest[self.scene_keys[0]]["scene_label"],
                disabled=True,
            )
            self.gui_seq_selector = self.server.gui.add_dropdown(
                "Sequence",
                options=["-"],
                initial_value="-",
            )
            self.gui_current_seq = self.server.gui.add_text(
                "Current Seq",
                initial_value="-",
                disabled=True,
            )

        @self.gui_scene_selector.on_update
        def _(_) -> None:
            if self.suppress_gui_events:
                return
            if self.gui_scene_selector.value == self.current_scene_key:
                return
            self.queue_scene_switch(self.gui_scene_selector.value)

        @self.gui_seq_selector.on_update
        def _(_) -> None:
            if self.suppress_gui_events:
                return
            if self.gui_seq_selector.value == self.current_seq:
                return
            self.queue_sequence_switch(self.gui_seq_selector.value)

    def setup_gui(self):
        self.gui_show_smpl = self.server.gui.add_checkbox("Show SMPL", True)

        @self.gui_show_smpl.on_update
        def _(_) -> None:
            self.apply_visibility()

        self.gui_show_cameras = self.server.gui.add_checkbox("Show Cameras", True)

        @self.gui_show_cameras.on_update
        def _(_) -> None:
            self.apply_visibility()

        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Frame",
                min=0.0,
                max=1.0,
                step=1.0,
                initial_value=0.0,
                disabled=False,
            )
            self.gui_play_pause = self.server.gui.add_button("Play", disabled=False)
            self.gui_framerate = self.server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=30.0)

        def set_play_state(playing):
            self.is_playing = playing
            single_frame = self.num_frames <= 1
            self.gui_timestep.disabled = playing or single_frame
            self.gui_play_pause.name = "Pause" if playing else "Play"

        @self.gui_play_pause.on_click
        def _(_) -> None:
            set_play_state(not self.is_playing)

        @self.gui_timestep.on_update
        def _(_) -> None:
            if self.gui_timestep.value is None or not np.isfinite(self.gui_timestep.value):
                return
            frame_value = float(self.gui_timestep.value)
            if self.enforce_zero_frame:
                # Drop stale non-zero UI events after sequence switching.
                still_locking = time.time() < self.enforce_zero_frame_until
                if still_locking and int(round(frame_value)) != 0:
                    self.gui_timestep.value = 0.0
                    return
                if (not still_locking) or int(round(frame_value)) == 0:
                    self.enforce_zero_frame = False
            self.update_frame(frame_value)

    def create_frames(self):
        for i in range(self.num_frames):
            frame_node = self.server.scene.add_frame(f"/frames/{i}", show_axes=False)
            frame_node.visible = (i == 0)
            self.frame_nodes.append(frame_node)

            mesh_handle = self.server.scene.add_mesh_simple(
                name=f"/frames/{i}/smpl",
                vertices=self.smpl_vertices[i],
                faces=self.smpl_faces,
                color=(100, 150, 200),
                opacity=1.0,
                wireframe=False,
            )
            self.mesh_handles.append(mesh_handle)

        frame_idx = self.frame_indices[0]
        R1 = self.data["R1"][frame_idx]
        T1 = self.data["T1"][frame_idx]
        K1 = self.data["K1"]

        c2w1 = np.eye(4)
        c2w1[:3, :3] = R1
        c2w1[:3, 3] = T1
        camera_center1 = c2w1[:3, 3]
        camera_rotation1 = c2w1[:3, :3]

        q1 = tf.SO3.from_matrix(camera_rotation1).wxyz
        fov1 = 2 * np.arctan(K1[1, 2] / K1[1, 1])
        aspect1 = K1[0, 0] / K1[1, 1]

        self.cam1_handle = self.server.scene.add_camera_frustum(
            name="/camera1",
            fov=fov1,
            aspect=aspect1,
            wxyz=q1,
            position=camera_center1,
            scale=0.1,
            color=(255, 0, 0),
        )

        R2 = self.data["R2"][frame_idx]
        T2 = self.data["T2"][frame_idx]
        K2 = self.data["K2"]

        c2w2 = np.eye(4)
        c2w2[:3, :3] = R2
        c2w2[:3, 3] = T2
        camera_center2 = c2w2[:3, 3]
        camera_rotation2 = c2w2[:3, :3]

        q2 = tf.SO3.from_matrix(camera_rotation2).wxyz
        fov2 = 2 * np.arctan(K2[1, 2] / K2[1, 1])
        aspect2 = K2[0, 0] / K2[1, 1]

        self.cam2_handle = self.server.scene.add_camera_frustum(
            name="/camera2",
            fov=fov2,
            aspect=aspect2,
            wxyz=q2,
            position=camera_center2,
            scale=0.1,
            color=(0, 255, 0),
        )
        self.apply_visibility()

    def update_frame(self, display_frame_idx):
        if self.num_frames == 0:
            return

        if display_frame_idx is None or not np.isfinite(display_frame_idx):
            return
        display_frame_idx = int(np.clip(display_frame_idx, 0, self.num_frames - 1))
        frame_idx = self.frame_indices[display_frame_idx]

        for i, frame_node in enumerate(self.frame_nodes):
            frame_node.visible = (i == display_frame_idx)

        R1 = self.data["R1"][frame_idx]
        T1 = self.data["T1"][frame_idx]
        c2w1 = np.eye(4)
        c2w1[:3, :3] = R1
        c2w1[:3, 3] = T1
        self.cam1_handle.wxyz = tf.SO3.from_matrix(c2w1[:3, :3]).wxyz
        self.cam1_handle.position = c2w1[:3, 3]

        R2 = self.data["R2"][frame_idx]
        T2 = self.data["T2"][frame_idx]
        c2w2 = np.eye(4)
        c2w2[:3, :3] = R2
        c2w2[:3, 3] = T2
        self.cam2_handle.wxyz = tf.SO3.from_matrix(c2w2[:3, :3]).wxyz
        self.cam2_handle.position = c2w2[:3, 3]

    def run(self):
        print(f"Viser server running at http://localhost:{self.port}")
        print(f"Scenes loaded: {len(self.scene_keys)}")
        print(f"Current scene/seq: {self.scene_name}/{self.current_seq}")

        while True:
            if self.pending_scene_key is not None and not self.is_switching:
                target_scene = self.pending_scene_key
                self.pending_scene_key = None
                if target_scene != self.current_scene_key:
                    self.is_switching = True
                    self.gui_scene_selector.disabled = True
                    self.gui_seq_selector.disabled = True
                    self.gui_timestep.disabled = True
                    try:
                        print(f"Switching scene -> {target_scene}")
                        self.load_scene(target_scene, reset_timestep=True)
                    finally:
                        self.gui_scene_selector.disabled = False
                        self.gui_seq_selector.disabled = False
                        self.gui_timestep.disabled = False
                        self.is_switching = False

            if self.pending_seq_name is not None and not self.is_switching:
                target_seq = self.pending_seq_name
                self.pending_seq_name = None
                if target_seq != self.current_seq:
                    self.is_switching = True
                    self.is_playing = False
                    self.gui_play_pause.name = "Play"
                    self.gui_seq_selector.disabled = True
                    self.gui_timestep.disabled = True
                    try:
                        print(f"Switching sequence -> {target_seq}")
                        self.load_sequence(target_seq, reset_timestep=True)
                    finally:
                        self.gui_seq_selector.disabled = False
                        self.gui_timestep.disabled = False
                        self.is_switching = False

            if self.is_playing and self.num_frames > 0:
                cur = self.gui_timestep.value
                cur_i = 0 if cur is None or not np.isfinite(cur) else int(cur)
                self.gui_timestep.value = float((cur_i + 1) % self.num_frames)

            fps = self.gui_framerate.value
            if fps is None or not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            time.sleep(1.0 / float(fps))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optimized SMPL motion with scene/sequence switch, from scene_path or xlsx.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        default=None,
        help="Single scene folder path (legacy mode, contains seq* subfolders).",
    )
    parser.add_argument("--xlsx", type=str, default=None, help="xlsx manifest path (multi-scene mode, skips FAILED rows).")
    parser.add_argument("--data_root", type=str, default=None, help="Optional root prefixed to xlsx scene_folder.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum frames to load per sequence; -1 means all.")
    parser.add_argument("--stride", type=int, default=1, help="Frame sampling stride.")
    parser.add_argument("--hq", action="store_true", help="Enable high-quality rendering with multiple lights and shadows.")
    parser.add_argument(
        "--scene_mesh",
        type=str,
        default="simple",
        choices=["raw", "simple", "no"],
        help="Scene mesh mode: simple=prefer mesh_simplified.ply and fallback to mesh_raw.ply; raw=use mesh_raw.ply only; no=disable scene mesh.",
    )
    parser.add_argument(
        "--mesh_level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="SMPL mesh downsampling level: 0=full, 1=downsample once (~1723 verts), 2=coarser.",
    )

    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        parser.print_help()
        return

    args = parser.parse_args()
    if args.xlsx is None and args.scene_path is None:
        print("Error: provide either --scene_path or --xlsx")
        return

    if args.xlsx is not None:
        manifest = build_manifest_from_xlsx(args.xlsx, data_root=args.data_root)
        if not manifest:
            print("Error: no valid scene/seq found from xlsx (FAILED rows are skipped).")
            return
    else:
        scene_path = Path(args.scene_path)
        if not scene_path.exists() or not scene_path.is_dir():
            print(f"Error: invalid scene path: {scene_path}")
            return
        manifest = build_manifest_from_scene(scene_path)
        if not manifest:
            print(f"Error: no valid seq* with optim_params.npz found in {scene_path}")
            return

    total_scenes = len(manifest)
    total_seqs = sum(len(v["seq_items"]) for v in manifest.values())
    print(f"Dataset summary: scenes={total_scenes}, seqs={total_seqs}")
    first_scene_key = sorted(manifest.keys())[0]
    first_scene = manifest[first_scene_key]
    print(f"Default scene: {first_scene['scene_label']}")
    print(f"Default sequence: {first_scene['seq_items'][0][0]}")

    print("Loading SMPL model...")
    body_model = load_smpl_model()

    viewer = MotionSceneViewer(
        manifest=manifest,
        body_model=body_model,
        port=args.port,
        max_frames=args.max_frames,
        stride=args.stride,
        high_quality=args.hq,
        scene_mesh=args.scene_mesh,
        mesh_level=args.mesh_level,
    )
    viewer.run()


if __name__ == '__main__':
    main()
