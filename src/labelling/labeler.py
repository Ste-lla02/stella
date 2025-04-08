import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import numpy as np
import pickle
from IPython.display import clear_output
from src.utils.configuration import Configuration
error_list=open('errors.txt','w')
class Dobby():

    label_dict = {
        '1': 'bladder',
        '2': 'urethra',
        '3': 'bladder_and_urethra',
        '4': 'other'
    }
    def __init__(self,conf):
        self.configuration = conf


    def load_csv(self, folder):
        df_name = self.configuration.get(folder)
        try:
            df=pd.read_csv(df_name,sep=';')

        except FileNotFoundError:
            columns=['Record ID','VUP','label_id','label_segmentation','area','predicted_iou','point_coords','stability_score','crop_box','id']
            df=pd.DataFrame(columns=columns)
        return df

    def mask_labeling(self,mask,image):
        print("Lables: 1-bladder, 2-urethra, 3-bladder_and_urethra, 4-other")
        mask_id = mask['id']
        binary_mask = mask["segmentation"]
        overlay = self.overlay_mask_on_image(image.copy(), binary_mask)
        fig = plt.figure()
        plt.imshow(overlay)
        plt.title(f"Mask ID: {mask_id}")
        plt.axis('off')
        plt.show(block=False)  # Non blocca l'esecuzione
        plt.pause(0.1)  # Mostra brevemente
        plt.close(fig)  # Chiude esplicitamente la figura

        label_id = input("Insert label: ")
        label = self.label_dict.get(label_id, 'other')
        clear_output(wait=True)
        return label_id, label


    def overlay_mask_on_image(self,image, mask, alpha=0.5):
        """Sovrappone una maschera (binary) all'immagine originale con trasparenza."""
        color_mask = np.zeros_like(image)
        color_mask[mask == 1] = [255, 0, 0]  # Rosso per maschere
        return cv2.addWeighted(image, 1, color_mask, alpha, 0)

    def labeling_helper(self):
        df_output=self.load_csv('lablescsv')
        df_input=self.load_csv('codescsv')
        patient_id = df_input['Record ID'].unique()
        already_processed=df_output['Record ID'].unique()
        id_index=0
        check=True
        while (id_index<len(patient_id) and (check==True)):
            image_name=patient_id[id_index]
            if(image_name not in already_processed):
                image_path = self.configuration.get('srcfolder')+'/'+str(image_name) +'.png' # o .jpg se serve
                image = cv2.imread(image_path)
                if((image is None)==False):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    is_pathologic = df_input[df_input['Record ID']==image_name]['VUP'].values[0]
                    pkl_path = self.configuration.get('picklefolder')+'/'+image_name+'.pickle'
                    try:
                        with open(pkl_path, 'rb') as f:
                              masks_from_pkl = pickle.load(f)
                        masks = masks_from_pkl['masks']

                        to_remove=list()
                        for mask in masks:
                            row = {
                                "Record ID": image_name,
                                "VUP": is_pathologic,
                            }
                            label_id,label=self.mask_labeling(mask,image)
                            row["label_id"]= label_id
                            row["label_segmentation"]= label

                            for k, v in mask.items():
                                if k not in row:
                                    row[k] = v
                            df_output.loc[len(df_output)] = row
                            to_remove.append(len(df_output) - 1)

                        ask = input("Do you want to delete the masks for the image .. and refill it? (Y/N): ").strip().lower()
                        if ((ask == 'y' or ask == 'Y')):
                            df_output.drop(index=to_remove, inplace=True)
                            df_output.reset_index(drop=True, inplace=True)
                            id_index-=1
                        else:
                            df_output_name = self.configuration.get('lablescsv')
                            df_output.to_csv(df_output_name, index=False, sep=';')



                        ask = input("Continue? (Y/N): ").strip().lower()
                        check = (ask == 'y' or ask == 'Y')
                    except Exception as e:
                        error_list.write('ERROR pickle ' + image_name +'\n')
                        error_list.write(str(e)+'\n')

                else:
                    error_list.write('image '+image_name+' not found\n')

            id_index+=1



        if (id_index==len(patient_id)):
            print("🧦")
            print("Dobby is a free elf!")
            print("https://www.youtube.com/watch?v=8DTb-lseCdQ")