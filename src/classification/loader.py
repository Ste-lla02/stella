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
        self.images = images  # immagini o maschere come array NumPy
        self.labels = labels  # etichette (opzionali)
        self.transform  = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            #transforms.Normalize(mean=[self.mean], std=[self.std]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

class Loader():
    def __init__(self,conf):
        self.configuration = conf
        self.lables_csv=pd.read_csv(self.configuration.get('lablescsv'),sep=';')
        self.manager=State(conf)
        self.train_loader=None
        self.test_loader=None
        self.dataset_sizes=None



    def compute_mean_std(self, loader):
        '''
        Calcola la media e la deviazione standard delle immagini, necessarie per la normalizzazione

        Args:
            loader (DataLoader): DataLoader del dataset.

        Returns:
            tuple: Media e deviazione standard.
        '''
        num_pixels = 0
        mean = 0.0
        std = 0.0

        for images, _ in loader:
            batch_size, num_channels, height, width = images.shape
            num_pixels += batch_size * height * width
            mean += images.mean(axis=(0, 2, 3)).sum()
            std += images.std(axis=(0, 2, 3)).sum()

        mean /= num_pixels
        std /= num_pixels
        return mean, std



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
                #overlay=self.manager.make_overall_image(id,[mask])
                mask_pillow = cv2_to_pil(binary_mask)
                #tensor_mask=torch.from_numpy(overlay)
                X.append(mask_pillow)
                label_mask=subset[subset['id']==mask_id]['label_id'].values[0]
                Y.append(label_mask)
        dataset= MaskDataset(X,Y)
        return dataset
    def load_data(self):
        '''
        Carica il dataset, calcola media e deviazione standard, lo normalizza e lo divide in train e test set.

        Returns:
            tuple: DataLoader per train, DataLoader per test, dimensioni del dataset, nomi delle classi.
        '''
        # Leggi e carica il dataset
        dataset = self.load_mask_dataset()

        # Calcola la media e la deviazione standard
        #aggiungi parametri nel config
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        test_size = int(len(loader.dataset.images) * 0.30)
        train_size = len(loader.dataset.images) - test_size
        train_dataset, test_dataset = random_split(loader.dataset, [train_size, test_size])

        # Crea DataLoader per train e test
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,
                                                        num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False,
                                                       num_workers=4)

        # Ottiene le dimensioni del dataset e i nomi delle classi
        self.dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}



