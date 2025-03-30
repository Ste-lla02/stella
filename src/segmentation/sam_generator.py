import os
import cv2, numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from src.segmentation.evaluator import MaskFeaturing
from src.utils.configuration import Configuration
import gc
from src.utils.utils import pil_to_cv2

class Segmenter:
    def __init__(self):
        configuration = Configuration()
        # sam model file and parameters
        model_path = configuration.get('sam_model')
        sam_platform = configuration.get('sam_platform')
        sam_kind = configuration.get('sam_kind')
        sam = sam_model_registry[sam_kind](checkpoint=model_path)
        sam = sam.to(sam_platform)
        # Getting mask quality parameter values
        points_per_side = configuration.get('points_per_side')
        min_mask_quality = configuration.get('min_mask_quality')
        min_mask_stability = configuration.get('min_mask_stability')
        layers = configuration.get('layers')
        crop_n_points_downscale_factor = configuration.get('crop_n_points_downscale_factor')
        min_mask_region_area = configuration.get('min_mask_region_area')
        # generation of the segmenter
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=min_mask_quality,
            stability_score_thresh=min_mask_stability,
            crop_n_layers=layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area
        )

    def mask_generation(self, image, name, channel):
        retval = list()
        gc.collect()
        cv2_image = pil_to_cv2(image)
        colored_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(colored_image)
        f = MaskFeaturing()
        for i, mask in enumerate(masks):
            id = {'id': i}
            properties = f.evaluation(mask)
            properties = {**properties, **id}
            retval.append({**mask, **properties})
        return retval

    @staticmethod
    def mask_voting(mask_list):
        pass