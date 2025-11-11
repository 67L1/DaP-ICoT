import os
# if using Apple MPS, fall back to CPU for unsupported ops
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import yaml
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class sam2Model:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, yaml.FullLoader)

        self.device = torch.device(self.config['sam2']['device'])
        self.sam2_checkpoint = self.config['sam2']['sam2_checkpoint']
        self.model_cfg = self.config['sam2']['model_cfg']
        self.sam2 = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device, apply_postprocessing=False)

    def execute(self, img_path):
        ori_image = Image.open(img_path)
        image = np.array(ori_image.convert("RGB"))
        mask_generator = SAM2AutomaticMaskGenerator(self.sam2)
        masks = mask_generator.generate(image)
        return masks, ori_image
