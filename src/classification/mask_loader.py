from torchvision import transforms
from src.utils.utils import cv2_to_pil
import random
from src.classification.abs_loader import AbstractLoader, MaskDataset


class Mask_Loader(AbstractLoader):
    def load_mask_dataset(self):
        self.manager.load_pickle()
        X=list()
        Y=list()
        for image_name, image in self.manager.images.items():
            masks=image['masks']
            for mask in masks:
                if('label_segmentation' in mask.keys()):
                    binary_mask=mask['segmentation']
                    mask_pillow = cv2_to_pil(binary_mask)
                    X.append(mask_pillow)
                    label_mask=mask['label_segmentation']
                    Y.append(int(label_mask)-1)
        return MaskDataset(X,Y)





