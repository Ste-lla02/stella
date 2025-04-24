import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import numpy as np
import pickle
from IPython.display import clear_output
from src.utils.configuration import Configuration
from src.core.core_model import State

class Dobby():


    def __init__(self,conf):
        self.configuration = conf
        self.manager=State(conf)
        self.df_output = self.load_csv('lablescsv')
        self.df_input = self.load_csv('codescsv')
        self.label_dict = {
            '1': 'bladder',
            '2': 'urethra',
            '3': 'bladder_and_urethra',
            '4': 'other'
        }

    def load_csv(self, folder):
        df_name = self.configuration.get(folder)
        try:
            df=pd.read_csv(df_name,sep=';')

        except FileNotFoundError:
            columns=['Record ID','VUP','label_id','label_segmentation','area','predicted_iou','point_coords','stability_score','crop_box','id']
            df=pd.DataFrame(columns=columns)
        return df

    def update_pikle(self,already_processed):
        for image_name in already_processed:
            if(image_name in self.manager.images.keys()):
                overall_image = self.manager.images[image_name]
                masks = overall_image['masks']
                if(not('label_segmentation' in masks[0].keys())):
                    subset = self.df_output[self.df_output['Record ID'] == image_name]
                    for mask in masks:
                        mask['label_segmentation'] = subset[subset['id'] == mask['id']]['label_id'].values[0]
                    self.manager.save_pickle(image_name)

    def mask_labeling(self,mask,image_name):
        print("Lables: 1-bladder, 2-urethra, 3-bladder_and_urethra, 4-other")
        mask_id = mask['id']
        overlay=self.manager.make_overall_image(image_name,[mask])
        fig = plt.figure()
        plt.imshow(overlay)
        plt.title(f"Mask ID: {mask_id}")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.07)
        plt.close(fig)
        label_id = input("Insert label: ")
        label = self.label_dict.get(label_id, 'other')
        clear_output(wait=True)
        return label_id, label


    def labeling_helper(self):
        patient_id = self.df_input['Record ID'].unique()
        already_processed=self.df_output['Record ID'].unique()
        self.manager.load_pickle()
        self.update_pikle(already_processed)
        id_index=0
        check=True
        while (id_index<len(patient_id) and (check==True)):
            image_name=patient_id[id_index]
            try:
                overall_image = self.manager.images[image_name]
                masks = overall_image['masks']
                if (image_name not in already_processed):
                    is_pathologic = self.df_input[self.df_input['Record ID']==image_name]['VUP'].values[0]
                    to_remove=list()
                    for mask in masks:
                        row = { "Record ID": image_name,"VUP": is_pathologic}
                        check=True
                        while check:
                            label_id,label=self.mask_labeling(mask,image_name)
                            if(label_id not in self.label_dict.keys()):
                                print('Value not acceptable, please retry!\n')
                            else:
                                check=False
                        row["label_id"]= label_id
                        row["label_segmentation"]= label
                        for k, v in mask.items():
                            if k not in row:
                                row[k] = v
                        mask['label_segmentation']=label_id
                        self.df_output.loc[len(self.df_output)] = row
                        to_remove.append(len(self.df_output) - 1)
                    ask = input("Do you want to delete the masks for the image "+image_name+" and refill it? (Y/N): ").strip().lower()
                    if ((ask == 'y')):
                        self.df_output.drop(index=to_remove, inplace=True)
                        self.df_output.reset_index(drop=True, inplace=True)
                        id_index-=1
                    else:
                        self.manager.save_pickle(image_name)
                        self.df_output_name = self.configuration.get('lablescsv')
                        self.df_output.to_csv(self.df_output_name, index=False, sep=';')
                        ask = input("Continue? (Y/N): ").strip().lower()
                        check = (ask == 'y')
            except KeyError as e:
                print('Image '+image_name+' pickle not found')
            id_index+=1
            if (id_index==len(patient_id)):
                print("🧦")
                print("Dobby is a free elf!")
                print("https://www.youtube.com/watch?v=8DTb-lseCdQ")