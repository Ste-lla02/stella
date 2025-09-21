import pandas as pd
from fontTools.misc.cython import returns
from sympy.parsing.maxima import sub_dict
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
        self.dataset=self.create_dataset(self.manager.images)




    def create_dataset(self,image_dict):
        X = list()
        Y = list()
        codes = list()
        for image_name, image in image_dict.items():
            try:
                masks=image['masks']
                #original=image['original']
                if('label_segmentation' in masks[0].keys()):
                    masks_filtered=[mask for mask in masks if mask['label_segmentation']!=4.0]
                    if(len(masks_filtered)>0):
                        overlay = self.manager.make_overall_image(image_name, masks_filtered, highlits=True)
                        #self.plot_overlay(overlay,image_name)
                        label_mask=self.code_csv[self.code_csv['Key']==image_name]['VUP'].values[0]
                        mask_pillow = cv2_to_pil(overlay)
                        X.append(mask_pillow)
                        Y.append(int(label_mask))
                        codes.append(image_name)
            except Exception as e:
                print('Error image '+str(image_name)+': '+str(e))
        return AbsDataset(X, Y, codes)

    def plot_overlay(self,overlay, image_name):
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(overlay)
        plt.savefig(self.configuration.get('maskfolder') + '/Overlay/' + str(image_name) + '.png', format='png',
                    dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)


        '''
            def load_split(self):
        sub_dict = self.manager.images.copy()
        split_file = self.configuration.get('split_file')
        split_df = pd.read_csv(
            split_file,
            sep=None, engine="python",  # inferisce il separatore
            header=None, names=["Record ID", "Group"]
        )
        subset_ids = split_df[split_df['Group']=='training']['Record ID'].tolist()
        train_ds = Subset(full_ds, train_indices)
        test_ds = Subset(full_ds, test_indices)
        train_dict = {k: sub_dict[k] for k in subset_ids if k in sub_dict}
        self.train_loader = self.create_dataset(train_dict)
        subset_ids = split_df[split_df['Group'] == 'test']['Record ID'].tolist()
        test_dict = {k: sub_dict[k] for k in subset_ids if k in sub_dict}
        self.test_loader = self.create_dataset(test_dict)
        '''





