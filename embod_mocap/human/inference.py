import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from .detector import DetectionModel
from .pose_estimator import PoseEstimator
from embod_mocap.config_paths import PATHS
from human.utils.kp_utils import draw_kps, get_coco_joint_names, get_coco_skeleton
from human.utils.mesh_utils import vis_smpl_cam
from human.utils.camera_utils import pred_cam_to_full_cam
from human.utils.imutils import write_video_from_array
from PIL import Image
from human.utils.lang_sam_utils import lang_sam_forward


def inference_human(imgs, fps, device, need_seg=False, filelist=None, scene_ids=None, body_model=None, save_folder=None, single_person=True, flip_eval=False):
    detector = DetectionModel(bbox_model_ckpt=PATHS.bbox_model_ckpt, pose_model_ckpt=PATHS.pose_model_ckpt, vit_cfg=PATHS.vit_cfg, device=device)
    pose_estimator = PoseEstimator(hmr2_ckpt=PATHS.hmr2_ckpt, device=device, flip_eval=flip_eval)
    with torch.no_grad():
        print(f'>> Inference detection model on {len(imgs)} images')
        for img in tqdm(imgs):
            detector.track(img, fps)
            torch.cuda.empty_cache()
    tracking_results = detector.process(fps)
    if single_person:
        tracking_results = {0: tracking_results[0]}
    
    # vis kps

    # im_kp2d = draw_kps(imgs[i], tracking_results[0]['keypoints'][i], get_coco_joint_names(), get_coco_skeleton())
    # cv2.imwrite('kps.jpg', im_kp2d)

    tracking_results = pose_estimator.run(imgs, tracking_results)

    for _id in tracking_results:
        pred_cam = tracking_results[_id]['pred_cam']
        c = torch.from_numpy(tracking_results[_id]['bbox'][:, :2]).float()
        s = torch.from_numpy(tracking_results[_id]['bbox'][:, 2:3]).float() * 200
        full_cam = pred_cam_to_full_cam(pred_cam.cpu(), c, s, torch.Tensor(imgs[0].shape[:2])[None]).to(device)
        tracking_results[_id]['pred_cam'] = full_cam
        tracking_results[_id]['betas'] = tracking_results[_id]['betas'].mean(0)[None]
    # # vis smpl
    # from IPython import embed; embed()
    # i = 0
    # video = []
    # for i in range(len(imgs)):
    #     body_pose = tracking_results[0]['body_pose'] # n, 23, 3, 3
    #     global_orient = tracking_results[0]['global_orient'] # n, 1, 3, 3
    #     betas = tracking_results[0]['betas'].mean(0)[None]
        
    #     verts = body_model(body_pose=body_pose, global_orient=global_orient, betas=betas, pose2rot=False).vertices
    #     im = vis_smpl_cam(verts[i:i+1], body_model, full_cam[i:i+1], device, imgs[0].shape[:2])[0]
    #     mask = (im[..., 3] != 0).astype(int)
    #     alpha = 0.7 # transparency
    #     im_smpl = im[..., :3] * mask[..., None] * alpha + imgs[i] * (1 - alpha) * mask[..., None] + imgs[i] * (1 - mask[..., None])
    #     im = im_smpl * mask[..., None] + imgs[i] * (1 - mask[..., None])
    #     im = im.astype(np.uint8)
    #     video.append(im)
    # if save_folder:
    #     write_video_from_array(video, f'{save_folder}/motion.mp4', fps)

    if need_seg:
        os.makedirs(f'{save_folder}/masks', exist_ok=True)
        if scene_ids is None:
            scene_ids = list(range(len(imgs)))
        image_pils = [Image.fromarray(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)) for i in scene_ids]
        text_prompt = "person."
        results = lang_sam_forward(image_pils, text_prompt, chunk_size=10)
        masks = []
        for i in range(len(results)):
            mask = results[i]['masks']
            if mask.shape[0] >1:
                mask = np.any(mask, axis=0)
            else:
                mask = mask[0]
            masks.append(mask)
            cv2.imwrite(f'{save_folder}/masks/{os.path.basename(filelist[i])}', mask * 255)

    del detector, pose_estimator
    torch.cuda.empty_cache()
    return tracking_results, masks
