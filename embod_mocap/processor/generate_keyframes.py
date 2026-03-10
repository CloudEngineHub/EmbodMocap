import argparse
import json
import os
import re
import numpy as np


def count_tracks_per_frame(points2d_path, prefix="v1"):
    data = np.load(points2d_path, allow_pickle=True)
    frame_tracks = {}
    pattern = re.compile(rf"{prefix}_(\d+)\.jpg")
    for key in data.files:
        m = pattern.match(key)
        if m:
            frame_id = int(m.group(1))
            frame_tracks[frame_id] = len(data[key])
    return frame_tracks


def select_keyframes_single_view(frame_tracks, num_frames, min_tracks=10, max_keyframes=50, min_keyframes=20):
    candidates = set()
    for fid in range(num_frames):
        if frame_tracks.get(fid, 0) >= min_tracks:
            candidates.add(fid)

    threshold = min_tracks
    while len(candidates) < min_keyframes and threshold > 1:
        threshold = threshold // 2
        candidates = set()
        for fid in range(num_frames):
            if frame_tracks.get(fid, 0) >= threshold:
                candidates.add(fid)
        if len(candidates) >= min_keyframes:
            print(f"Relaxed min_tracks from {min_tracks} to {threshold}, candidates: {len(candidates)}")

    if len(candidates) <= max_keyframes:
        return sorted(candidates)

    selected = set()
    segment_size = max(1, num_frames / max_keyframes)
    for seg_idx in range(max_keyframes):
        seg_start = int(seg_idx * segment_size)
        seg_end = min(int((seg_idx + 1) * segment_size), num_frames)
        best_fid = None
        best_score = -1
        for fid in range(seg_start, seg_end):
            if fid not in candidates:
                continue
            score = frame_tracks.get(fid, 0)
            if score > best_score:
                best_score = score
                best_fid = fid
        if best_fid is not None:
            selected.add(best_fid)
    return sorted(selected)


def select_keyframes_stride(num_frames, stride):
    return sorted(range(0, num_frames, stride))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate keyframes.json with per-purpose keyframe lists."
    )
    parser.add_argument("folder", type=str, help="Path to the sequence folder.")
    parser.add_argument("--min_tracks", type=int, default=10,
                        help="Minimum track count threshold per frame.")
    parser.add_argument("--num_keyframes", type=int, default=50,
                        help="Max number of keyframes per view for p2p.")
    parser.add_argument("--min_keyframes", type=int, default=20,
                        help="Min number of keyframes per view for p2p, will relax min_tracks if needed.")
    parser.add_argument("--stride_vggt", type=int, default=100,
                        help="Stride for vggt_track keyframes.")
    parser.add_argument("--stride_unproj", type=int, default=30,
                        help="Stride for unproj_human keyframes.")
    args = parser.parse_args()

    v1_points2d = os.path.join(args.folder, "v1", "points2D.npz")
    v2_points2d = os.path.join(args.folder, "v2", "points2D.npz")

    if not os.path.exists(v1_points2d) or not os.path.exists(v2_points2d):
        print("Warning: points2D.npz not found, skip keyframe generation")
        exit(0)

    frame_tracks_v1 = count_tracks_per_frame(v1_points2d, prefix="v1")
    frame_tracks_v2 = count_tracks_per_frame(v2_points2d, prefix="v2")

    v1_images_dir = os.path.join(args.folder, "v1", "images")
    num_frames = len([f for f in os.listdir(v1_images_dir)
                      if f.startswith("v1_") and f.endswith(".jpg")])

    keyframes = {
        "vggt": select_keyframes_stride(num_frames, args.stride_vggt),
        "unproj": select_keyframes_stride(num_frames, args.stride_unproj),
        "p2p_v1": select_keyframes_single_view(
            frame_tracks_v1, num_frames, args.min_tracks, args.num_keyframes, args.min_keyframes),
        "p2p_v2": select_keyframes_single_view(
            frame_tracks_v2, num_frames, args.min_tracks, args.num_keyframes, args.min_keyframes),
    }

    out_path = os.path.join(args.folder, "keyframes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(keyframes, f, indent=2)

    print(f"Colmap registered frames: v1={len(frame_tracks_v1)}, v2={len(frame_tracks_v2)}")
    print(f"Generated keyframes.json: "
          f"vggt={len(keyframes['vggt'])}, unproj={len(keyframes['unproj'])}, "
          f"p2p_v1={len(keyframes['p2p_v1'])}, p2p_v2={len(keyframes['p2p_v2'])}")
    print(f"Saved to {out_path}")
