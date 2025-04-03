import os
import pandas as pd
import numpy as np
import torch
from skimage import io
from skimage.util import img_as_float
from PIL import Image
import torchvision.datasets as datasets
from torchvision import transforms
from numpy import random
import torch.nn as nn
from torch.utils.data import random_split

# Classe per la gestione del dataset di immagini.

class ImageDataset:
    def __init__(self, dataset_path, batch_size, test_split):
        '''
        Args:
            dataset_path (str): Path del dataset.
            batch_size (int): Dimensione dei batch per i DataLoader
            test_split (float): Percentuale del dataset da destinare al test set
        '''
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.test_split = test_split
        self.mean = None
        self.std = None
        self.train_loader = None
        self.test_loader = None
        self.dataset_sizes = None
        self.class_names = None

    def read_images(self):
        '''
        Legge le immagini e le trasforma

        Returns:
            Dataset trasformato.
        '''
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.Resize((n, m)),
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(self.dataset_path, transform=data_transforms)
        return dataset

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


    def load_data(self):
        '''
        Carica il dataset, calcola media e deviazione standard, lo normalizza e lo divide in train e test set.

        Returns:
            tuple: DataLoader per train, DataLoader per test, dimensioni del dataset, nomi delle classi.
        '''
        # Leggi e carica il dataset
        full_dataset = self.read_images()

        # Calcola la media e la deviazione standard
        loader = torch.utils.data.DataLoader(full_dataset, batch_size=self.batch_size, shuffle=True)
        self.mean, self.std = self.compute_mean_std(loader)

        # Trasformazioni con normalizzazione
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean], std=[self.std])
        ])

        # Aggiorna il dataset con le trasformazioni normalizzate
        full_dataset = datasets.ImageFolder(self.dataset_path, transform=data_transforms)

        # Divide il dataset in train e test set
        test_size = int(len(full_dataset) * self.test_split)
        train_size = len(full_dataset) - test_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Crea DataLoader per train e test
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                       num_workers=4)

        # Ottiene le dimensioni del dataset e i nomi delle classi
        self.dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}
        self.class_names = full_dataset.classes
        return self.train_loader, self.test_loader, self.dataset_sizes, self.class_names


'''if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Path del dataset
    dataset_path = 'dataset'
    batch_size = 32
    test_split = 0.2'''

    #data = ImageDataset(dataset_path, batch_size=batch_size, test_split=test_split)
    #train_loader, test_loader, data_sizes, class_names = data.load_data()

    #print(f"Train set size: {data_sizes['train']}")
    #print(f"Test set size: {data_sizes['test']}")
    #print(f"Classi: {class_names}")'''