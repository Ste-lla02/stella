import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from src.core.core_model import State

class Dobby_db():


    def __init__(self,conf):
        self.configuration = conf
        self.manager=State(conf)
        self.df_output = self.load_csv('lablescsv')
        self.df_input = self.load_csv('db_files')
        self.label_dict = {
            '1': 'bladder',
            '2': 'urethra',
            '3': 'bladder_and_urethra',
            '4': 'other'
        }

    def load_csv(self):
        df_name = self.configuration.get('df_masks')
        masks_df=pd.read_csv(df_name,sep=';')
        df_pickle = self.configuration.get('df_pickle')
        df_pickle=pd.read_pickle(df_pickle,sep=';')
        try:
            merged_df = pd.merge(df_pickle, masks_df, how='inner', left_on='id', right_on='imageset_id')
        except FileNotFoundError:
            print('File not found')
            merged_df = None
        return merged_df

    def update_pikle(self,df):
        for image_name in df['imageset_id']:
            #todo estrarre dal image_name l'id del pickle ID_0
            if(image_name in self.manager.images.keys()):
                overall_image = self.manager.images[image_name]
                masks = overall_image['masks']
                if(not('label_segmentation' in masks[0].keys())):
                    subset = self.df[self.df['imageset_id'] == image_name]
                    for mask in masks:
                        mask['label_segmentation'] = subset[subset['mask_index'] == mask['mask_index']]['label_id'].values[0]
                    self.manager.save_pickle(image_name)
