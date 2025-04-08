import numpy

from src.core.core_model import State
import pandas as pd


class Loader():
    def __init__(self,conf):
        self.configuration = conf
        self.lables_csv=pd.read_csv(self.configuration.get('lablescsv'),sep=';')
        self.manager=State(conf)


    def load_mask_dataset(self):
        ids=self.lables_csv['Record ID'].unique()
        self.manager.load_pickle()
        X=list()
        Y=list()
        for id in ids:
            subset=self.lables_csv[self.lables_csv['Record ID']==id]
            overall_image=self.manager.images[id]
            masks=overall_image['masks']
            for mask in masks:
                mask_id=mask['id']
                X.append(mask['segmentation'])
                label_mask=subset[subset['id']==mask_id]['label_id'].values[0]
                Y.append(label_mask)
        return X,Y

