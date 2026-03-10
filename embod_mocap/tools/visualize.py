import os
import cv2
import numpy as np
import argparse
import torch
import shutil
import math
import pandas as pd
from natsort import natsorted
from embod_mocap.human.utils.kp_utils import draw_kps, get_coco_joint_names, get_coco_skeleton
from embod_mocap.human.utils.mesh_utils import vis_smpl_cam, vis_smpl
from embod_mocap.human.configs import BMODEL
from embod_mocap.human.smpl import SMPL
from t3drender.cameras import PerspectiveCameras
from tqdm import tqdm
from tqdm import trange


def get_bool_from_excel(row, column, default=False):
    value = row.get(column, default)
    if value is None or pd.isna(value):
        return default
    return str(value).upper() in ['TRUE', '1.0']


def build_seq_list(seq_path=None, xlsx_path=None, data_root=None, force_all=False):
    if seq_path is not None:
        return [seq_path]

    if xlsx_path is None:
        raise ValueError("Provide either --seq_path or --xlsx")

    xlsx_path = os.path.abspath(xlsx_path)
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"xlsx not found: {xlsx_path}")

    xl = pd.ExcelFile(xlsx_path)
    if len(xl.sheet_names) > 1:
        dfs = [pd.read_excel(xlsx_path, sheet_name=sheet_name) for sheet_name in xl.sheet_names]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_excel(xlsx_path)

    seq_paths = []
    for _, row in df.iterrows():
        if get_bool_from_excel(row, 'FAILED') and not force_all:
            continue

        scene_folder_rel = str(row['scene_folder'])
        seq_name = str(row['seq_name'])
        scene_folder = os.path.join(data_root, scene_folder_rel) if data_root else scene_folder_rel
        curr_seq_path = os.path.join(scene_folder, seq_name)
        if os.path.isdir(curr_seq_path):
            seq_paths.append(curr_seq_path)
        else:
            print(f"Skip missing sequence: {curr_seq_path}")
    return seq_paths


def load_resized_frames(frames_dir, prefix, downscale):
    frames = []
    for i in range(len(os.listdir(frames_dir))):
        img_path = os.path.join(frames_dir, f"{prefix}_{i:04d}.jpg")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img.shape[1]//downscale, img.shape[0]//downscale))
        frames.append(img)
    return frames


def visualize_sequence(seq_path, args):
    downscale = args.downscale
    print(f"\n{'='*60}")
    print(f"Visualizing sequence: {seq_path}")
    print(f"{'='*60}")

    if args.input:
        output_path = os.path.join(seq_path, "concat_input.mp4")
        if args.mode == "overwrite" or not os.path.exists(output_path):
            print("visualizing input video")
            v1_frames_dir = os.path.join(seq_path, "v1", "images")
            v2_frames_dir = os.path.join(seq_path, "v2", "images")
            concat_images_to_video(v1_frames_dir, v2_frames_dir, output_path, down_scale=downscale)
        else:
            print(f"Skip visualizing input video, {seq_path} input video already exists")

    if args.processed:
        print("visualizing processed video")
        body_model = SMPL(
            model_path=BMODEL.FLDR,
            gender='neutral',
            extra_joints_regressor=BMODEL.JOINTS_REGRESSOR_EXTRA,
            create_transl=False).to(args.device)
        concated_video_path = os.path.join(seq_path, "concat_processed.mp4")

        v1_frames_dir = os.path.join(seq_path, "v1", "images")
        v2_frames_dir = os.path.join(seq_path, "v2", "images")

        v1_frames_processed_cache = os.path.join(seq_path, "v1", "frames_processed")
        v2_frames_processed_cache = os.path.join(seq_path, "v2", "frames_processed")
        os.makedirs(v1_frames_processed_cache, exist_ok=True)
        os.makedirs(v2_frames_processed_cache, exist_ok=True)
        tracking_results1 = np.load(os.path.join(seq_path, "v1", "smpl_params.npz"), allow_pickle=True)
        tracking_results2 = np.load(os.path.join(seq_path, "v2", "smpl_params.npz"), allow_pickle=True)

        v1_frames = load_resized_frames(v1_frames_dir, "v1", downscale)
        v2_frames = load_resized_frames(v2_frames_dir, "v2", downscale)

        vis_processed(v1_frames, tracking_results1, body_model, v1_frames_processed_cache, args.device, chunk_size=args.vis_chunk, down_scale=downscale)
        vis_processed(v2_frames, tracking_results2, body_model, v2_frames_processed_cache, args.device, chunk_size=args.vis_chunk, down_scale=downscale)
        concat_images_to_video(v1_frames_processed_cache, v2_frames_processed_cache, concated_video_path, down_scale=1)
        shutil.rmtree(v1_frames_processed_cache)
        shutil.rmtree(v2_frames_processed_cache)

    if args.optim_cam:
        print("visualizing optimized camera view")
        body_model = SMPL(
            model_path=BMODEL.FLDR,
            gender='neutral',
            extra_joints_regressor=BMODEL.JOINTS_REGRESSOR_EXTRA,
            create_transl=False).to(args.device)

        smpl_params_optim = np.load(os.path.join(seq_path, "optim_params.npz"))
        cameras1 = np.load(os.path.join(seq_path, "v1", "cameras.npz"))
        cameras2 = np.load(os.path.join(seq_path, "v2", "cameras.npz"))
        smpl_params_optim = {k: torch.from_numpy(v).to(args.device) for k, v in smpl_params_optim.items()}

        K1 = cameras1['K'][:3, :3]
        K2 = cameras2['K'][:3, :3]
        R1, T1 = cameras1['R'], cameras1['T']
        R2, T2 = cameras2['R'], cameras2['T']

        R1 = torch.from_numpy(R1).to(args.device)
        T1 = torch.from_numpy(T1).to(args.device)
        R2 = torch.from_numpy(R2).to(args.device)
        T2 = torch.from_numpy(T2).to(args.device)
        h, w = 960, 720

        v1_frames_dir = os.path.join(seq_path, "v1", "images")
        v2_frames_dir = os.path.join(seq_path, "v2", "images")
        v1_frames = load_resized_frames(v1_frames_dir, "v1", downscale)
        v2_frames = load_resized_frames(v2_frames_dir, "v2", downscale)

        v1_frames_processed_cache = os.path.join(seq_path, "v1", "frames_optimized")
        v2_frames_processed_cache = os.path.join(seq_path, "v2", "frames_optimized")

        K1_render = torch.from_numpy(K1).to(args.device)[None]
        K1_render[:2, :] /= downscale
        cameras_v1 = PerspectiveCameras(
            K=K1_render,
            R=R1,
            T=T1,
            device=args.device,
            in_ndc=False,
            convention='opencv',
            image_size=(h//downscale, w//downscale),
        )

        K2_render = torch.from_numpy(K2).to(args.device)[None]
        K2_render[:2, :] /= downscale
        cameras_v2 = PerspectiveCameras(
            K=K2_render,
            R=R2,
            T=T2,
            device=args.device,
            in_ndc=False,
            convention='opencv',
            image_size=(h//downscale, w//downscale),
        )
        concated_video_path = os.path.join(seq_path, "concat_optimized.mp4")

        vis_optimized(v1_frames, smpl_params_optim, cameras_v1, body_model, v1_frames_processed_cache, args.device, chunk_size=args.vis_chunk)
        vis_optimized(v2_frames, smpl_params_optim, cameras_v2, body_model, v2_frames_processed_cache, args.device, chunk_size=args.vis_chunk)
        concat_images_to_video(v1_frames_processed_cache, v2_frames_processed_cache, concated_video_path, down_scale=1)
        shutil.rmtree(v1_frames_processed_cache)
        shutil.rmtree(v2_frames_processed_cache)


def write_mp4_from_numpy(array, output_path, fps=30):
    assert array.ndim == 4, "Input array must be 4D (B, H, W, 3)"
    assert array.shape[3] == 3, "Last dimension must have size 3 (RGB channels)"
    
    B, H, W, C = array.shape
    assert C == 3, "The last dimension of the array must be 3 (RGB channels)"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(B):
        frame = array[i]
        if array.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Video saved to {output_path}")


def cache_folder_to_video(cache_folder, output_video_path, fps=30):
    images = natsorted([os.path.join(cache_folder, f) for f in os.listdir(cache_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    img = cv2.imread(images[0])
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in tqdm(images):
        img = cv2.imread(img_path)
        cv2.putText(img, f"{os.path.basename(img_path)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(img)
    out.release()   
    print(f"video saved to {output_video_path}")


def concat_images_to_video(folder1, folder2, output_video_path, rotate=False, down_scale=4, fps=30):
    images1 = natsorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))])
    images2 = natsorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(images1) != len(images2):
        raise ValueError("the number of images in the two folders must be the same!")
    img1 = cv2.imread(images1[0])
    img2 = cv2.imread(images2[0])
    height, width, _ = img1.shape
    if rotate:
        total_width = 2 * height
    else:
        total_width = 2 * width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    if rotate:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (total_width//down_scale, width//down_scale))
    else:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (total_width//down_scale, height//down_scale))

    for img_path1, img_path2 in tqdm(zip(images1, images2)):
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        img1 = cv2.resize(img1, (width//down_scale, height//down_scale))
        img2 = cv2.resize(img2, (width//down_scale, height//down_scale))
        if rotate:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
        if img1 is None or img2 is None:
            print(f"Error reading images: {img_path1} or {img_path2}")

        concatenated = cv2.hconcat([img1, img2])
        cv2.putText(concatenated, f"{os.path.basename(img_path1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(concatenated)

    out.release()   
    print(f"video saved to {output_video_path}")


def vis_optimized(imgs, smpl_params, cameras, body_model, cache_dir, device, chunk_size=60):
    # vis smpl
    body_pose = smpl_params['body_pose'] # n, 23, 3, 3
    global_orient = smpl_params['global_orient'] # n, 1, 3, 3
    betas = smpl_params['betas'].mean(0)[None]
    transl = smpl_params['transl']
    os.makedirs(cache_dir, exist_ok=True)

    body_pose = torch.Tensor(body_pose).to(device)
    global_orient = torch.Tensor(global_orient).to(device)
    betas = torch.Tensor(betas).to(device)
    transl = torch.Tensor(transl).to(device)
    image_paths = [os.path.join(cache_dir, f"vis_{i:04d}.png") for i in range(len(imgs))]
    with torch.amp.autocast(device_type='cuda', enabled=False): 
        verts = body_model(body_pose=body_pose, global_orient=global_orient, betas=betas, transl=transl, pose2rot=False).vertices
        for i in trange(math.ceil(len(image_paths) // chunk_size)):
            curr_indices = list(range(i * chunk_size, min((i + 1) * chunk_size, len(image_paths))))
            curr_image_paths = [image_paths[i] for i in curr_indices]
            curr_bg_images = [imgs[i] for i in curr_indices]
            curr_cameras = cameras[curr_indices]
            vis_smpl(verts[curr_indices], body_model, device, curr_cameras, batch_size=len(curr_indices), resolution=imgs[0].shape[:2], verbose=False, image_paths=curr_image_paths, bg_images=curr_bg_images)
            torch.cuda.empty_cache()
            

def vis_processed(imgs, tracking_results, body_model, cache_dir, device, chunk_size=60, down_scale=4):
    # vis smpl
    body_pose = tracking_results['body_pose'] # n, 23, 3, 3
    global_orient = tracking_results['global_orient'] # n, 1, 3, 3
    betas = tracking_results['betas'].mean(0)[None]
    full_cam = tracking_results['pred_cam'] # n, 3
    body_pose = torch.Tensor(body_pose).to(device)
    global_orient = torch.Tensor(global_orient).to(device)
    betas = torch.Tensor(betas).to(device)
    full_cam = torch.Tensor(full_cam).to(device)

    c = tracking_results['bbox'][:, :2] / down_scale
    s = tracking_results['bbox'][:, 2:3] * 200 / down_scale

    with torch.amp.autocast(device_type='cuda', enabled=False): 
        verts = body_model(body_pose=body_pose, global_orient=global_orient, betas=betas, pose2rot=False).vertices
        image_paths = [os.path.join(cache_dir, f"vis_{i:04d}.png") for i in range(len(imgs))]
        for i in trange(math.ceil(len(image_paths) // chunk_size)):
            curr_indices = list(range(i * chunk_size, min((i + 1) * chunk_size, len(image_paths))))
            curr_image_paths = [image_paths[i] for i in curr_indices]
            curr_bg_images = [imgs[i] for i in curr_indices]
            vis_smpl_cam(verts[curr_indices], body_model, full_cam[curr_indices], device, batch_size=len(curr_indices), resolution=imgs[0].shape[:2], verbose=False
            , image_paths=curr_image_paths, bg_images=curr_bg_images)
            for j, image_path in enumerate(curr_image_paths):
                im = cv2.imread(image_path)
                im = draw_kps(im, tracking_results['keypoints'][curr_indices[j], :, :2] / down_scale, get_coco_joint_names(), get_coco_skeleton(), point_radius=int(4/down_scale), line_thickness=int(2/down_scale))
                cv2.imwrite(image_path, im)

                color=(0, 0, 255)
                thickness = int(2 / down_scale)
                # Extract scalar values to avoid NumPy deprecation warning
                cx_val = c[i, 0].item() if hasattr(c[i, 0], 'item') else c[i, 0]
                cy_val = c[i, 1].item() if hasattr(c[i, 1], 'item') else c[i, 1]
                s_val = s[i].item() if hasattr(s[i], 'item') else s[i]
                w = h = s_val

                # Calculate top-left and bottom-right corners of the bbox
                x_min = int(cx_val - w / 2)
                y_min = int(cy_val - h / 2)
                x_max = int(cx_val + w / 2)
                y_max = int(cy_val + h / 2)

                # # Draw the bounding box
                im = im.copy()
                cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color, thickness)
                cv2.imwrite(image_path, im)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize processed frames and concatenate videos."
    )
    parser.add_argument("--seq_path", type=str, default=None, help="Single sequence folder path.")
    parser.add_argument("--xlsx", type=str, default=None, help="xlsx manifest path for batch mode.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional root prefixed to xlsx scene_folder.")
    parser.add_argument("--force_all", action="store_true", help="Include rows marked as FAILED in xlsx batch mode.")
    parser.add_argument(
        "--input",
        action="store_true",
        help="Visualize input frames.",
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        help="Visualize processed frames.",
    )
    parser.add_argument(
        "--optim_cam",
        action="store_true",
        help="Visualize optimized frames.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for processing.",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=2,
        help="Downscale factor for visualization.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overwrite",
        choices=["overwrite", "skip"],
        help="overwrite or skip",
    )
    parser.add_argument(
        "--vis_chunk",
        type=int,
        default=60,
        help="SMPL visualization chunk size.",
    )

    args = parser.parse_args()

    if not (args.input or args.processed or args.optim_cam):
        parser.error("At least one of --input, --processed, or --optim_cam is required.")

    seq_path = args.seq_path
    if seq_path is not None and args.xlsx is not None:
        parser.error("Choose either --seq_path or --xlsx, not both.")
    if seq_path is None and args.xlsx is None:
        parser.error("Provide either --seq_path or --xlsx.")
    if args.xlsx is not None and args.data_root is None:
        parser.error("--xlsx mode requires --data_root.")

    seq_paths = build_seq_list(
        seq_path=seq_path,
        xlsx_path=args.xlsx,
        data_root=args.data_root,
        force_all=args.force_all,
    )
    if not seq_paths:
        print("No valid sequences found for visualization.")
        raise SystemExit(1)

    for curr_seq_path in seq_paths:
        visualize_sequence(curr_seq_path, args)
