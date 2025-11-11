import numpy as np
import torch
from PIL import Image
from process_res import *
import os
import gc
import pickle
import yaml
from sam2_detect import *
import json
from tqdm import tqdm
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
except ImportError:
    print("Warning: Could not import smart_resize from transformers. "
          "Ensure it's available for the script to run correctly.")


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





if __name__ == '__main__':
    for DATA_NAME in ['m3cot']:
        config_path = "../config/config.yaml"
        model_name = 'qwen' # the model you want to use
        output_path = "../data_all/"
        image_pool = {}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, yaml.FullLoader)

        detect_model = sam2Model(config_path)

        output_path = output_path + DATA_NAME + '/' + DATA_NAME + "_sam2_image_pool_qwen.pkl"

        dataset = open(config['dataset'][DATA_NAME]['EVAL_FILE']).readlines()
        dataset = [json.loads(d) for d in dataset]
        dataset = [x for x in dataset if x['image'] is not None]
        print(len(dataset))
        IMG_FOLDER = config['dataset'][DATA_NAME]['IMG_FOLDER']

        from transformers import AutoProcessor

        try:
            qwen_processor = AutoProcessor.from_pretrained(config['model'][model_name]['model_path'], trust_remote_code=True, device=config['sam2']['device'])
            qwen_image_proc = qwen_processor.image_processor
        except Exception as e:
            print(f"Failed to load Qwen Processor: {e}")
            exit()



        import math

        _smart_resize_available = False
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

            _smart_resize_available = True
            print("Succeed to load transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize")
        except ImportError:
            print('Failed to load.')


            def smart_resize(
                    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56,
                    max_pixels: int = 14 * 14 * 4 * 1280
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


        for idx, item in tqdm(enumerate(dataset), desc="Constructing pool", total=len(dataset)):
            image_path = os.path.join(IMG_FOLDER, item['image_id'] + '.png')
            image_info, image = detect_model.execute(image_path)
            image_patch_pool = map_sam2_masks_to_qwen_llm_tokens(
                original_image=image_path,
                sam2_results=image_info,
                qwen_image_processor=qwen_image_proc,
                max_tokens_per_mask=64  #
            )
            image_pool[item['id']] = image_patch_pool

            if not image_patch_pool:
                print("No masks. Please check.")
            for info in image_patch_pool:
                print(f"Mask ID: {info['mask_id']}")
                print(f"  Original BBox (XYWH): {info['original_bbox_xywh']}")
                print(f"  Covered Qwen LLM Token Count: {info['token_count']}")
                print(f"  Covered Qwen LLM Token Indices: {info['vq_token_indices'].tolist()}")
                print("-" * 20)

            del image_patch_pool
            torch.cuda.empty_cache()
            gc.collect()

        with open(output_path, 'wb') as f:
            pickle.dump(image_pool, f)
