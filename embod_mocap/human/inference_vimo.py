import cv2
import os
import numpy as np
import torch
import math
from yacs.config import CfgNode as CN
from tqdm import tqdm
from .detector import DetectionModel
from embod_mocap.config_paths import PATHS
from human.utils.kp_utils import draw_kps, get_coco_joint_names, get_coco_skeleton
from human.utils.mesh_utils import vis_smpl_cam
from human.utils.camera_utils import pred_cam_to_full_cam
from human.utils.imutils import write_video_from_array
from PIL import Image
from human.utils.lang_sam_utils import lang_sam_forward, find_best_matching_channel
from human.utils.transforms import interpolate_smpl_rotmat_camera
from human.backbone.vimo import HMR_VIMO


def get_default_config():
    cfg_file = os.path.join(
        os.path.dirname(__file__),
        'config_vimo.yaml'
        )

    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    return cfg

def get_hmr_vimo(checkpoint=None, device='cuda'):
    cfg = get_default_config()
    cfg.device = device
    model = HMR_VIMO(cfg)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location='cpu')
        _ = model.load_state_dict(ckpt['model'], strict=False)

    model = model.to(device)
    _ = model.eval()

    return model

def inference_human(imgs, fps, device, need_seg=False, filelist=None, scene_ids=None, body_model=None, save_folder=None, single_person=True, flip_eval=False, vis=False):
    detector = DetectionModel(bbox_model_ckpt=PATHS.bbox_model_ckpt, pose_model_ckpt=PATHS.pose_model_ckpt, vit_cfg=PATHS.vit_cfg, device=device)
    pose_estimator = get_hmr_vimo(checkpoint=PATHS.vimo_ckpt, device=device)
    with torch.no_grad():
        print(f'>> Inference detection model on {len(imgs)} images')
        for img in tqdm(imgs):
            # if single_person:
            #     detector.detect(img, fps)
            # else:
            detector.track(img, fps)
            torch.cuda.empty_cache()
    tracking_results = detector.process(fps)
    max_bbox_len = 0
    largest_id = -1
    for _id in tracking_results:
        if len(tracking_results[_id]['bbox']) > max_bbox_len:
            max_bbox_len = len(tracking_results[_id]['bbox'])
            largest_id = _id
    if single_person:
        tracking_results = {0: tracking_results[largest_id]}
    # vis kps

    # im_kp2d = draw_kps(imgs[i], tracking_results[0]['keypoints'][i], get_coco_joint_names(), get_coco_skeleton())
    # cv2.imwrite('kps.jpg', im_kp2d)
    h, w = imgs[0].shape[:2]
    img_focal = math.sqrt(h**2 + w**2)
    img_center = np.array([w/2., h/2.])
    for _id, val in tracking_results.items():
        frame = val['frame_id']
        valid = np.ones(len(frame), dtype=bool)
        c = val['bbox'][:, :2]
        s = val['bbox'][:, 2:3] * 200
        # boxes: nx5 = x1, y1, x2, y2, score
        boxes = np.concatenate([c - s / 2, c + s / 2], axis=1)
        boxes = np.clip(boxes, 0, None)
        boxes = np.concatenate([boxes, np.ones((len(boxes), 1))], axis=1)
        results = pose_estimator.inference(imgs, boxes, valid=valid, frame=frame,
                                img_focal=img_focal, img_center=img_center)

        tracking_results[_id]['global_orient'] = results['pred_rotmat'][:, :1]
        tracking_results[_id]['body_pose'] = results['pred_rotmat'][:, 1:]
        tracking_results[_id]['betas'] = results['pred_shape']
        tracking_results[_id]['pred_cam'] = results['pred_cam']

    for _id in tracking_results:
        pred_cam = tracking_results[_id]['pred_cam']
        smpl_frame = results['frame'].tolist()
        track_frame = tracking_results[_id]['frame_id']
        mapping = [track_frame.tolist().index(i) for i in smpl_frame]
        bbox = tracking_results[_id]['bbox'][mapping]
        c = torch.from_numpy(bbox[:, :2]).float()
        s = torch.from_numpy(bbox[:, 2:3]).float() * 200
        full_cam = pred_cam_to_full_cam(pred_cam.cpu(), c, s, torch.Tensor([h, w])[None]).to(device)
        tracking_results[_id]['pred_cam'] = full_cam
        tracking_results[_id]['betas'] = tracking_results[_id]['betas'].mean(0)[None]
        smpl_rotmats = torch.cat([tracking_results[_id]['global_orient'], tracking_results[_id]['body_pose']], 1)
        smpl_rotmats, full_cam = interpolate_smpl_rotmat_camera(smpl_rotmats, full_cam, smpl_frame, list(range(len(imgs))))
        tracking_results[_id]['global_orient'] = smpl_rotmats[:, :1]
        tracking_results[_id]['body_pose'] = smpl_rotmats[:, 1:]
        tracking_results[_id]['pred_cam'] = full_cam
        assert len(tracking_results[_id]['global_orient']) == len(imgs)
        assert len(tracking_results[_id]['body_pose']) == len(imgs)
        assert len(tracking_results[_id]['pred_cam']) == len(imgs)
        
    # vis smpl
    if vis:
        body_pose = tracking_results[0]['body_pose'] # n, 23, 3, 3
        global_orient = tracking_results[0]['global_orient'] # n, 1, 3, 3
        betas = tracking_results[0]['betas'].mean(0)[None]

        with torch.cuda.amp.autocast(enabled=False): 
            verts = body_model(body_pose=body_pose, global_orient=global_orient, betas=betas, pose2rot=False).vertices
            ims = vis_smpl_cam(verts, body_model, full_cam, device, imgs[0].shape[:2])
        
        video = []
        for i in range(len(imgs)):
            im_smpl = ims[i]
            mask = (im_smpl[..., 3] != 0).astype(int)
            alpha = 0.7 # transparency
            im_smpl = im_smpl[..., :3] * mask[..., None] * alpha + imgs[i] * (1 - alpha) * mask[..., None] + imgs[i] * (1 - mask[..., None])
            im = im_smpl * mask[..., None] + imgs[i] * (1 - mask[..., None])
            im = im.astype(np.uint8)
            im = draw_kps(im, tracking_results[0]['keypoints'][i], get_coco_joint_names(), get_coco_skeleton())
            
            color=(0, 0, 255)
            thickness=2
            cx, cy = c[i]  # Unpack center coordinates
            w = h = s[i]   # Since s = max(w, h), we assume the bbox is a square

            # Calculate top-left and bottom-right corners of the bbox
            x_min = int(cx - w / 2)
            y_min = int(cy - h / 2)
            x_max = int(cx + w / 2)
            y_max = int(cy + h / 2)

            # Draw the bounding box
            im = im.copy()
            cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color, thickness)

            # Draw the center point
            cv2.circle(im, (int(cx), int(cy)), radius=3, color=color, thickness=-1)
            video.append(im)
                
        if save_folder:
            write_video_from_array(video, f'{save_folder}/{os.path.basename(save_folder)}.mp4', fps)

    if need_seg:
        c = tracking_results[0]['bbox'][:, :2]
        s = tracking_results[0]['bbox'][:, 2:3] * 200
        bbox_xyxy_ = np.concatenate([c - s / 2, c + s / 2], axis=1)
        bbox_xyxy = np.zeros((len(imgs), 4))
        bbox_xyxy[tracking_results[0]['frame_id']] = bbox_xyxy_
        os.makedirs(f'{save_folder}/masks', exist_ok=True)
        if scene_ids is None:
            scene_ids = list(range(len(imgs)))
        image_pils = [Image.fromarray(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)) for i in scene_ids]
        text_prompt = "person."

        results = lang_sam_forward(image_pils, text_prompt, chunk_size=10)
        
        masks = []
        for i in range(len(results)):
            mask = results[i]['masks']
            if i not in tracking_results[0]['frame_id']:
                mask = np.zeros((imgs[i].shape[0], imgs[i].shape[1], 3))
            elif len(mask)>1:
                main_target_mask = mask[find_best_matching_channel(bbox_xyxy[i], mask)]
                union_mask = np.any(mask, axis=0)
                other_human = np.logical_and(union_mask, np.logical_not(main_target_mask))
                mask = np.stack([main_target_mask, union_mask, other_human], axis=-1)
            elif len(mask) == 1:
                main_target_mask = mask[0]
                union_mask = mask[0]
                mask = np.stack([main_target_mask, union_mask, np.zeros_like(union_mask)], axis=-1)
            elif len(mask) == 0:
                mask = np.zeros((imgs[i].shape[0], imgs[i].shape[1], 3))

            masks.append(mask)
            cv2.imwrite(f'{save_folder}/masks/{os.path.basename(filelist[i])}', mask * 255)

    del detector, pose_estimator
    torch.cuda.empty_cache()
    return tracking_results, masks
