import numpy
import torch
#from skimage import io
#from skimage.util import img_as_float
from PIL import Image
import torchvision.datasets as datasets
from torchvision import transforms
from numpy import random
import torch.nn as nn
from torch.utils.data import random_split
from src.core.core_model import State
import pandas as pd
from torch.utils.data import Dataset
from src.utils.utils import cv2_to_pil
class MaskDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.mean, self.std=self.compute_mean_std()
        self.transform  = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)
        y = torch.tensor([y-1], dtype=torch.long)
        return x, y

    def compute_mean_std(self):
        means=[]
        stds= []
        transform_no_norm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        for img in self.images:
            tensor = transform_no_norm(img)  # (C, H, W)
            means.append(tensor.mean(dim=(1, 2)))  # media per canale
            stds.append(tensor.std(dim=(1, 2)))    # std per canale

        mean = torch.stack(means).mean(dim=0)  # (3,)
        std = torch.stack(stds).mean(dim=0)    # (3,)
        return mean.tolist(), std.tolist()


    def get_num_classes(self):
        classes=list(set(self.labels))
        return len(classes)




class Loader():
    def __init__(self,conf):
        self.configuration = conf
        self.lables_csv=pd.read_csv(self.configuration.get('lablescsv'),sep=';')
        self.manager=State(conf)
        self.dataset=self.load_mask_dataset()
        self.train_loader=None
        self.test_loader=None
        self.dataset_sizes=None

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
                binary_mask=mask['segmentation']
                mask_pillow = cv2_to_pil(binary_mask)
                X.append(mask_pillow)
                label_mask=subset[subset['id']==mask_id]['label_id'].values[0]
                Y.append(label_mask)
        return MaskDataset(X,Y)

    def load_data(self):
        '''
        Carica il dataset, calcola media e deviazione standard, lo normalizza e lo divide in train e test set.

        Returns:
            tuple: DataLoader per train, DataLoader per test, dimensioni del dataset, nomi delle classi.
        '''
        # Leggi e carica il dataset
        test_size = int(len(self.dataset.images) * 0.2)
        train_size = len(self.dataset.images) - test_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        index_train=train_dataset.indices
        X_train=[self.dataset.images[i] for i in index_train]
        Y_train=[self.dataset.labels[i] for i in index_train]
        index_test=test_dataset.indices
        X_test = [self.dataset.images[i] for i in index_test]
        Y_test = [self.dataset.labels[i] for i in index_test]
        self.train_loader = MaskDataset(X_train, Y_train)
        self.test_loader = MaskDataset(X_test, Y_test)
        self.dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}




