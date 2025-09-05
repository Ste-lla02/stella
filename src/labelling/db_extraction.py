import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from src.core.core_model import State

class Dobby_db():


    def __init__(self,conf):
        self.configuration = conf
        self.manager=State(conf)
        self.label_dict = {
            '1': 'bladder',
            '2': 'urethra',
            '3': 'bladder_and_urethra',
            '4': 'other'
        }

    def load_csv(self):
        df_masks = self.configuration.get('df_masks')
        df_masks=pd.read_csv(df_masks,sep=';')
        df_pickle = self.configuration.get('df_pickle')
        df_pickle=pd.read_csv(df_pickle,sep=';')
        try:
            #self.db_csv = pd.merge(df_masks,df_pickle,  how='outer', left_on='imageset_id', right_on='id')
            self.db_csv = df_masks.merge(df_pickle, left_on='imageset_id', right_on='id', how='left')
            self.db_csv.to_csv('input/labelled_masks/db_merged.csv',index=False,sep=';')
        except FileNotFoundError:
            print('File not found')
            self.db_csv = None


    def update_pikle(self):
        self.load_csv()
        self.manager.load_pickle()
        values = self.db_csv['name'].dropna().unique()
        for image_path in values:
            try:
                image_name=image_path.split('.')[0]
                if(image_name in self.manager.images.keys()):
                    overall_image = self.manager.images[image_name]
                    masks = overall_image['masks']
                    #if(not('label_segmentation' in masks[0].keys())):
                    subset = self.db_csv[self.db_csv['name'] == image_path]
                    for mask in masks:
                        label = subset[subset['mask_index'] == mask['id']]['label_segmentation'].values[0]
                        mask['label_segmentation'] = label
                    self.manager.save_pickle(image_name)
            except Exception as e:
                print(e)

