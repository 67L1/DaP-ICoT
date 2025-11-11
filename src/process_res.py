import math

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

import PIL.Image
import numpy as np
import torch

def map_sam2_masks_to_qwen_llm_tokens(
        original_image: str,
        sam2_results: list,
        qwen_image_processor,
        max_tokens_per_mask: int = 64
) -> list:
    original_image = PIL.Image.open(original_image)

    """
    Maps SAM2 segmentation masks to the LLM visual token indices of Qwen2-VL.

    Args:
        original_image (PIL.Image.Image): The original PIL Image (in RGB format).
        sam2_results (list[dict]): A list of dictionaries from the SAM2 output, each containing
                                   'segmentation', 'bbox', etc.
        qwen_image_processor (Qwen2VLImageProcessor): An initialized instance of Qwen2VLImageProcessor.
        max_tokens_per_mask (int): The maximum number of LLM tokens a single mask is allowed to
                                   cover. Masks exceeding this limit are ignored.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - "mask_id" (any): The identifier for the SAM mask.
            - "qwen_llm_token_indices" (torch.LongTensor): A 1D tensor of the flattened indices
              of Qwen LLM visual tokens covered by this mask.
            - "token_count" (int): The number of unique LLM tokens covered.
            - "original_bbox_xywh" (list): The original bounding box (XYWH) from the SAM2 output.
            - ... (other information from SAM2, such as predicted_iou, stability_score)
    """

    if not isinstance(sam2_results, list):
        print("Error: sam2_results must be a list.")
        return []
    if not original_image:
        print("Error: A valid original_image must be provided.")
        return []
    if not hasattr(qwen_image_processor, 'patch_size') or \
            not hasattr(qwen_image_processor, 'merge_size'):
        print("Error: qwen_image_processor does not appear to be a valid Qwen2VLImageProcessor instance.")
        return []

    orig_w, orig_h = original_image.size

    patch_size = qwen_image_processor.patch_size
    merge_size = qwen_image_processor.merge_size
    min_pixels = qwen_image_processor.min_pixels
    max_pixels = qwen_image_processor.max_pixels

    factor_for_smart_resize = patch_size * merge_size
    try:
        resized_h, resized_w = smart_resize(
            orig_h,
            orig_w,
            factor=factor_for_smart_resize,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    except ValueError as e:
        print(f"Failed: {e}. Skip this one.")
        return []
    except NotImplementedError as e:
        print(f"Error: {e}")
        return []

    scale_w = resized_w / orig_w if orig_w > 0 else 1.0
    scale_h = resized_h / orig_h if orig_h > 0 else 1.0

    grid_t_vit_patches = 1
    grid_h_vit_patches = resized_h // patch_size
    grid_w_vit_patches = resized_w // patch_size

    if grid_h_vit_patches == 0 or grid_w_vit_patches == 0:
        return []

    num_llm_tokens_t = grid_t_vit_patches  # 时间维度上的 LLM token 数量
    num_llm_tokens_h = grid_h_vit_patches // merge_size
    num_llm_tokens_w = grid_w_vit_patches // merge_size

    if num_llm_tokens_h == 0 or num_llm_tokens_w == 0:
        return []

    llm_token_pixel_height_on_resized = patch_size * merge_size
    llm_token_pixel_width_on_resized = patch_size * merge_size

    processed_masks_info = []
    for i, mask_data in enumerate(sam2_results):
        segmentation_mask_np = mask_data.get('segmentation')
        original_bbox_xywh = mask_data.get('bbox')

        if segmentation_mask_np is None or not isinstance(segmentation_mask_np, np.ndarray):
            continue
        if segmentation_mask_np.shape[0] != orig_h or segmentation_mask_np.shape[1] != orig_w:
            continue
        if segmentation_mask_np.dtype != bool:
            segmentation_mask_np = segmentation_mask_np.astype(bool)

        y_coords_orig, x_coords_orig = np.where(segmentation_mask_np)

        if len(x_coords_orig) == 0:
            continue

        x_coords_resized = x_coords_orig * scale_w
        y_coords_resized = y_coords_orig * scale_h

        valid_resized_indices = (x_coords_resized >= 0) & (x_coords_resized < resized_w) & \
                                (y_coords_resized >= 0) & (y_coords_resized < resized_h)

        x_coords_valid_resized = x_coords_resized[valid_resized_indices]
        y_coords_valid_resized = y_coords_resized[valid_resized_indices]

        if len(x_coords_valid_resized) == 0:
            continue

        llm_token_col_indices = np.floor(x_coords_valid_resized / llm_token_pixel_width_on_resized).astype(int)
        llm_token_row_indices = np.floor(y_coords_valid_resized / llm_token_pixel_height_on_resized).astype(int)

        llm_token_col_indices = np.clip(llm_token_col_indices, 0, num_llm_tokens_w - 1)
        llm_token_row_indices = np.clip(llm_token_row_indices, 0, num_llm_tokens_h - 1)

        llm_token_time_idx_for_pixels = np.zeros_like(llm_token_row_indices, dtype=int)

        indices_1d = llm_token_time_idx_for_pixels * (num_llm_tokens_h * num_llm_tokens_w) + \
                     llm_token_row_indices * num_llm_tokens_w + \
                     llm_token_col_indices

        unique_llm_token_indices = np.unique(indices_1d)

        current_token_count = len(unique_llm_token_indices)

        if 0 < current_token_count <= max_tokens_per_mask:
            mask_info = {
                "mask_id": mask_data.get("mask_id", f"sam_mask_{i}"),
                "vq_token_indices": torch.tensor(unique_llm_token_indices, dtype=torch.long),
                "token_count": current_token_count,
                "original_bbox_xywh": list(original_bbox_xywh) if original_bbox_xywh is not None else None,
                "predicted_iou": mask_data.get("predicted_iou"),
                "stability_score": mask_data.get("stability_score"),
                "area_pixels": mask_data.get("area"),
                "segmentation": mask_data.get("segmentation")
            }
            processed_masks_info.append(mask_info)

    return processed_masks_info