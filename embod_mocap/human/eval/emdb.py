import torch
import pickle as pkl

from human.eval.utils import (
    compute_global_metrics,
    as_np_array,
)
from einops import einsum
from human.configs import BMODEL
from human.smpl import SMPL
from human.utils.tensor_utils import dict2tensor, slice_dict
from human.utils.post_process import pp_static_joint, is_contact


class MetricMocap():
    def __init__(self, device):
        """
        Args:
            emdb_split: 1 to evaluate incam, 2 to evaluate global
        """
        super().__init__()
        self.device = device
        self.metric_aggregator = {
            "wa2_mpjpe": {},
            "waa_mpjpe": {},
            "rte": {},
            "jitter": {},
            "fs": {},
        }
        self.body_model = {}
        for gender in ['neutral', 'male', 'female']:
            self.body_model[gender] = SMPL(
                model_path=BMODEL.FLDR,
                gender=gender,
                create_transl=False).to(device)

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, pred_smpl_params_global, gt_file, seq_name):
        """The behaviour is the same for val/test/predict"""
        with open(gt_file, "rb") as f:
            batch = pkl.load(f)
        target_w_params = batch['smpl']
        target_w_params = dict2tensor(target_w_params, self.device)
        target_w_params['global_orient'] = target_w_params.pop('poses_root')
        target_w_params['body_pose'] = target_w_params.pop('poses_body')
        target_w_params['transl'] = target_w_params.pop('trans')
        target_w_params['betas'] = target_w_params.pop('betas')[None]
        gender = batch["gender"]
        mask = torch.Tensor(batch["good_frames_mask"]).bool().to(self.device)
        target_w_output = self.body_model[gender](**target_w_params)
        target_w_verts = target_w_output.vertices
        target_w_j3d = target_w_output.joints
        
        # betas = pred_smpl_params_global['betas'].clone()
        # pred_smpl_params_global = target_w_params
        # pred_smpl_params_global['betas'] = betas
        
        # smpl_out = self.body_model['neutral'](**pred_smpl_params_global)
        
        # contact_label = is_contact(target_w_j3d, fps=30, velocity_threshold=0.1)[:, [7, 10, 8, 11, 20, 21]]
        # pred_smpl_params_global['transl'] = pp_static_joint(contact_label[None], transl=pred_smpl_params_global['transl'][None], pred_w_j3d=smpl_out.joints[None])[0]

        smpl_out = self.body_model['neutral'](**pred_smpl_params_global)
        
        pred_ay_verts = smpl_out.vertices
        pred_ay_j3d = smpl_out.joints
        del smpl_out  # Prevent OOM

        batch_eval = {
            "pred_j3d_glob": pred_ay_j3d,
            "target_j3d_glob": target_w_j3d,
            "pred_verts_glob": pred_ay_verts,
            "target_verts_glob": target_w_verts,
        }

        global_metrics = compute_global_metrics(batch_eval, mask=mask)
        for k, v in global_metrics.items():
            global_metrics[k] = v
        for k in global_metrics:
            self.metric_aggregator[k][seq_name] = as_np_array(global_metrics[k])