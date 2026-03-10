import os
from pathlib import Path
from easydict import EasyDict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROOT = os.fspath(_REPO_ROOT)

PATHS = EasyDict({
    "bbox_model_ckpt": f"{_ROOT}/checkpoints/yolov8x.pt",
    "pose_model_ckpt": f"{_ROOT}/checkpoints/vitpose-h-multi-coco.pth",
    "vit_cfg": f"{_ROOT}/embod_mocap/thirdparty/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py",
    "vimo_ckpt": f"{_ROOT}/checkpoints/vimo_checkpoint.pth.tar",
    "sam2_ckpt": f"{_ROOT}/checkpoints/sam2.1_hiera_large.pt",
    "sam2_config": f"{_ROOT}/configs/sam2.1/sam2.1_hiera_l.yaml",
    "lingbotdepth_ckpt": f"{_ROOT}/checkpoints/lingbot_depth_vitl14.pt",
    "vggt_ckpt": f"{_ROOT}/checkpoints/vggt.pt",
    "colmap_vocab_tree_path": f"{_ROOT}/checkpoints/vocab_tree_flickr100K_words32K.bin",
    "lang_sam_sam_type": "sam2.1_hiera_small",
    "lang_sam_sam_ckpt": f"{_ROOT}/checkpoints/sam2.1_hiera_small.pt",
})
