import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from src.classification.abs_loader import AbstractLoader, MaskDataset
from src.utils.utils import cv2_to_pil
import numpy as np


class Image_Loader(AbstractLoader):
    def load_mask_dataset(self):
        self.code_csv=pd.read_csv(self.configuration.get('codescsv'),sep=';')
        self.manager.load_pickle()
        X=list()
        Y=list()

        for image_name, image in self.manager.images.items():
            try:
                masks=image['masks']
                masks=image['image_to_predict']
                if('label_segmentation' in masks[0].keys()):
                    masks_filtered=[mask for mask in masks if str(mask['label_segmentation'])!='4']
                    if(len(masks_filtered)>0):
                        overlay = self.manager.make_masks_overlay(masks_filtered)
                        '''
                        fig = plt.figure()
                        plt.axis('off')
                        plt.imshow(overlay)
                        plt.savefig(self.configuration.get('maskfolder')+'\\Selected_masks\\'+str(image_name)+'.png',format='png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
                        plt.close(fig)
                        '''
                        label_mask=self.code_csv[self.code_csv['Record ID']==image_name]['VUP'].values[0]
                        mask_pillow = cv2_to_pil(overlay)
                        X.append(mask_pillow)
                        Y.append(int(label_mask))
            except Exception as e:
                print('Error image '+str(image_name)+': '+str(e))
        return MaskDataset(X,Y)







