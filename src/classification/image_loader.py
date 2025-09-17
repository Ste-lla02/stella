import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from src.classification.abs_loader import AbstractLoader, AbsDataset
from src.utils.utils import cv2_to_pil
import numpy as np


class Image_Loader(AbstractLoader):
    def load_dataset(self):
        self.code_csv=pd.read_csv(self.configuration.get('codescsv'),sep=';')
        self.manager.load_pickle()
        X=list()
        Y=list()
        codes=list()

        for image_name, image in self.manager.images.items():
            if(image_name=='ID_152'):
                print('ciao')
            try:
                masks=image['masks']
                #original=image['original']
                if('label_segmentation' in masks[0].keys()):
                    masks_filtered=[mask for mask in masks if mask['label_segmentation']!=4.0]
                    if(len(masks_filtered)>0):
                        #overlay = self.manager.make_masks_overlay(masks_filtered)
                        overlay = self.manager.make_overall_image(image_name, masks_filtered, highlits=True)
                        '''
                        fig = plt.figure()
                        plt.axis('off')
                        plt.imshow(overlay)
                        plt.savefig(self.configuration.get('maskfolder')+'/Overlay/'+str(image_name)+'.png',format='png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
                        plt.close(fig)
                        '''
                        #label_mask=self.code_csv[self.code_csv['Record ID']==image_name]['VUP'].values[0]
                        label_mask=self.code_csv[self.code_csv['Key']==image_name]['VUP'].values[0]
                        mask_pillow = cv2_to_pil(overlay)
                        X.append(mask_pillow)
                        Y.append(int(label_mask))
                        codes.append(image_name)
            except Exception as e:
                print('Error image '+str(image_name)+': '+str(e))
        return AbsDataset(X, Y, codes)







