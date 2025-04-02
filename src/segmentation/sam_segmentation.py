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
        # Caricare il modello e assegnarlo alla CPU
        model_path = configuration.get('sam_model')
        sam_platform = configuration.get('sam_platform')
        sam_kind = configuration.get('sam_kind')
        sam = sam_model_registry[sam_kind](checkpoint=model_path)
        sam = sam.to(sam_platform)  # Cambia in "cpu" se usi CPU, "cuda" per GPU
        # Getting mask quality parameter values
        points_per_side = configuration.get('points_per_side')
        min_mask_quality = configuration.get('min_mask_quality')
        min_mask_stability = configuration.get('min_mask_stability')
        layers = configuration.get('layers')
        crop_n_points_downscale_factor = configuration.get('crop_n_points_downscale_factor')
        min_mask_region_area = configuration.get('min_pixels')
        # Creare il generatore di maschere
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=min_mask_quality,
            stability_score_thresh=min_mask_stability,
            crop_n_layers=layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area
        )

    @staticmethod
    def make_overall_image(masks, base_image, image_name):
        configuration = Configuration()
        mask_folder = configuration.get("maskfolder")
        if len(masks) > 0:
            h, w = masks[0]['segmentation'].shape
            overlay = np.zeros((h, w, 3), dtype=np.uint8)  # Immagine vuota per le maschere
            for mask in masks:
                mask = mask['segmentation'].astype(np.uint8)  # Converti la maschera in uint8 (0-1 -> 0-255)
                color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Colore casuale
                overlay[mask > 0] = color  # Applica colore alla maschera
            mask_filename = os.path.join(mask_folder, f"{image_name}_masks.png")
            if base_image is not None:
                base_img = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)  # Converti in BGR se l'immagine di base è RGB
                alpha = 0.35
                blended = cv2.addWeighted(base_img, 1 - alpha, overlay, alpha, 0)
            else:
                blended = overlay
            cv2.imwrite(mask_filename, blended)
            print(f"Immagine salvata: {mask_filename}")

    def mask_generation(self, image):
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