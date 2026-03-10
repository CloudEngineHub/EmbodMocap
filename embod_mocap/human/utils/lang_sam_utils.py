import torch
from tqdm import trange
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from embod_mocap.config_paths import PATHS
import numpy as np
from PIL import Image

from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM
from lang_sam.models.utils import DEVICE


class LangSAM:
    def __init__(self, sam_type="sam2.1_hiera_small", sam_ckpt_path: str | None = None, gdino_model_ckpt_path: str | None = None, gdino_processor_ckpt_path: str | None = None, device=DEVICE):
        self.sam_type = sam_type

        self.sam = SAM()
        self.sam.build_model(sam_type, sam_ckpt_path, device=device)
        self.gdino = GDINO()
        self.gdino.build_model(model_ckpt_path=gdino_model_ckpt_path, processor_ckpt_path=gdino_processor_ckpt_path, device=device)

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """Predicts masks for given images and text prompts using GDINO and SAM models.

        Parameters:
            images_pil (list[Image.Image]): List of input images.
            texts_prompt (list[str]): List of text prompts corresponding to the images.
            box_threshold (float): Threshold for box predictions.
            text_threshold (float): Threshold for text predictions.

        Returns:
            list[dict]: List of results containing masks and other outputs for each image.
            Output format:
            [{
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "masks": np.ndarray,
                "mask_scores": np.ndarray,
            }, ...]
        """

        gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        for idx, result in enumerate(gdino_results):
            result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            if result["labels"]:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)
        if sam_images:
            # print(f"Predicting {len(sam_boxes)} masks")
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
            # print(f"Predicted {len(all_results)} masks")
        return all_results


def lang_sam_forward(
    image_pils,
    text_prompt,
    chunk_size=1,
    verbose=True,
    sam_type="sam2.1_hiera_small",
    sam_ckpt_path=None,
    gdino_model_ckpt_path=None,
    gdino_processor_ckpt_path=None,
):
    """Forward pass for LangSAM model.

    Parameters:
        image_pils (list[Image.Image]): List of input images.
        text_prompt (str): Text prompt for the images.
        model (LangSAM): LangSAM model.
        chunk_size (int): Chunk size for processing images.

    Returns:
        None
    """
    model = LangSAM(
        sam_type=sam_type,
        sam_ckpt_path=sam_ckpt_path,
        gdino_model_ckpt_path=gdino_model_ckpt_path,
        gdino_processor_ckpt_path=gdino_processor_ckpt_path,
    )
    results_list = []   
    if verbose:
        range_tool = trange
        print("Segmenting images...")
    else:
        range_tool = range
    for i in range_tool(0, len(image_pils), chunk_size):
        with torch.no_grad():
            num_images = len(image_pils[i:i+chunk_size])
            results = model.predict(image_pils[i:i+chunk_size], [text_prompt]*num_images)
        torch.cuda.empty_cache()
        results_list.extend(results)
    del model
    return results_list


def sam2_video(box, video_dir, device):
    predictor = build_sam2_video_predictor(PATHS.sam2_config, PATHS.sam2_ckpt, device=device)
    
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

    box = box.reshape(4)
    
    inference_state = predictor.init_state(video_path=video_dir)
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments


def mask_to_bbox(mask):
    """
    Convert a binary mask to a bounding box, ensuring the bbox is a square.
    
    Parameters:
        mask (numpy.ndarray): A binary mask of shape (H, W).
    
    Returns:
        bbox (tuple): The bbox in the format (x_min, y_min, x_max, y_max).
    """
    # Find the non-zero pixels in the mask
    y_coords, x_coords = np.where(mask > 0)

    if len(x_coords) == 0 or len(y_coords) == 0:  # No non-zero pixels in the mask
        return None

    # Determine the bounding box extents
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Ensure the bbox is square by expanding the smaller dimension
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    size = max(width, height)

    # Adjust the bbox to make it square
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_min = max(0, x_center - size // 2)
    x_max = x_min + size - 1
    y_min = max(0, y_center - size // 2)
    y_max = y_min + size - 1

    return (x_min, y_min, x_max, y_max)


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
        bbox1 (tuple): The first bbox (x_min, y_min, x_max, y_max).
        bbox2 (tuple): The second bbox (x_min, y_min, x_max, y_max).
    
    Returns:
        iou (float): The IoU value.
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate the intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calculate the area of intersection
    inter_width = max(0, inter_x_max - inter_x_min + 1)
    inter_height = max(0, inter_y_max - inter_y_min + 1)
    inter_area = inter_width * inter_height

    # Calculate the area of both bboxes
    area1 = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    area2 = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    # Calculate the union area
    union_area = area1 + area2 - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    # Compute IoU
    iou = inter_area / union_area
    return iou

def find_best_matching_channel(bbox, masks):
    """
    Find the mask channel that best matches the given bbox, based on maximum IoU.
    
    Parameters:
        bbox (tuple): The coordinates of the bbox in the format (x_min, y_min, x_max, y_max).
        masks (numpy.ndarray): A mask array of shape (N, H, W), where N is the number of masks.
    
    Returns:
        best_channel (int): The index of the mask channel that best matches the bbox.
                            Returns -1 if no match is found.
    """
    best_channel = -1
    max_iou = 0

    # Iterate over each mask channel
    for channel in range(masks.shape[0]):
        # Convert mask to bbox (ensuring square)
        mask_bbox = mask_to_bbox(masks[channel])
        if mask_bbox is None:  # Skip empty masks
            continue

        # Calculate IoU with the given bbox
        iou = calculate_iou(mask_bbox, bbox)
        if iou > max_iou:
            max_iou = iou
            best_channel = channel

    return best_channel
