import random
import torch
from torchvision import transforms
from torch.utils.data import random_split
from src.core.core_model import State
import pandas as pd
from torch.utils.data import Dataset
from src.utils.utils import cv2_to_pil
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
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

    def get_class_instances(self, class_id):
        indexes_class=[i for i, val in enumerate(self.labels) if val==class_id]
        subset=[self.images[i] for i in indexes_class]
        return subset,indexes_class

    def get_max_size(self):
        classes = list(set(self.labels))
        max=0
        for c in classes:
            dim=len(self.get_class_instances(c)[0])
            if(dim>max):
                max=dim
        return max
    def add_new_instances(self,image, label):
        self.images.append(image)
        self.labels.append(label)



class Loader():
    def __init__(self,conf):
        self.configuration = conf
        #self.lables_csv=pd.read_csv(self.configuration.get('lablescsv'),sep=';')
        self.manager=State(conf)
        self.dataset=self.load_mask_dataset()
        self.train_loader=None
        self.test_loader=None
        self.dataset_sizes=None

    def load_mask_dataset(self):
        self.manager.load_pickle()
        X=list()
        Y=list()
        for image_name, image in self.manager.images.items():
            masks=image['masks']
            for mask in masks:
                if('label_segmentation' in mask.keys()):
                    binary_mask=mask['segmentation']
                    mask_pillow = cv2_to_pil(binary_mask)
                    X.append(mask_pillow)
                    label_mask=mask['label_segmentation']
                    Y.append(int(label_mask))
        return MaskDataset(X,Y)

    def load_data(self):
        self.augmentation()
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



    def image_generator(self,image,label,n):
        rotation=self.configuration.get('rotation_range')
        p_hor=self.configuration.get('flip_hor_probability')
        p_ver=self.configuration.get('flip_ver_probability')
        new_images=list()
        transformer=transforms.Compose([
            transforms.RandomHorizontalFlip(p=p_hor),
            transforms.RandomRotation(degrees=(0, rotation)),
            transforms.RandomVerticalFlip(p=p_ver)
        ])
        for i in range(n):
            result = transformer(image)
            new_images.append(result)
            self.dataset.add_new_instances(result,label)

        return new_images
    def augmentation(self):
        classes=list(set(self.dataset.labels))
        target_size=self.dataset.get_max_size()
        for c in classes:
            df_group, indexes = self.dataset.get_class_instances(c)
            current_size = len(df_group)

            if current_size < target_size:
                q, r = divmod(target_size, current_size)
                #combinations=[(random.choice(rotation), random.choice(width), random.choice(height)) for _ in range(repeat)]
                for index in indexes:
                    new_images = self.image_generator(self.dataset.images[index],int(c),q)
                    '''
                    fig = plt.figure()
                    # im=cv2_to_pil(im)
                    plt.imshow(self.dataset.images[index])
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(0.1)
                    plt.close(fig)
                    for im in new_images:
                        fig = plt.figure()
                        #im=cv2_to_pil(im)
                        plt.imshow(im)
                        plt.axis('off')
                        plt.show(block=False)
                        plt.pause(0.1)
                        plt.close(fig)
                    '''



            '''
            image_datagen = ImageDataGenerator(
                rotation_range=45,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='reflect', cval=125)

            image_generator = image_datagen.flow_from_dataframe(
                dataframe=df_balanced,
                x_col="img_path",
                y_col="label",
                class_mode="categorical",
                batch_size=4,
                shuffle=True
            )
            '''
