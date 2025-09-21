from abc import ABC, abstractmethod
import torch
from torchvision import transforms
from src.core.core_model import State
from torch.utils.data import Dataset
from torch.utils.data import random_split
import random
import pandas as pd

class AbsDataset(Dataset):
    def __init__(self, images, labels,codes=None,batch_size=32):
        self.batch_size=batch_size
        self.codes=codes
        self.images = images
        #check 44 from wherw
        self.labels = labels
        self.mean, self.std=self.compute_mean_std()
        self.transform  = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
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
        y = torch.tensor([y], dtype=torch.long)
        return x, y

    def compute_mean_std(self):
        means=[]
        stds= []
        transform_no_norm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
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

    def get_min_size(self):
        classes = list(set(self.labels))
        min=len(self.labels)
        for c in classes:
            dim=len(self.get_class_instances(c)[0])
            if(dim<min):
                min=dim
        return min
    def add_new_instances(self,image, label,code):
        self.images.append(image)
        self.labels.append(label)
        self.codes.append(code)

class AbstractLoader(ABC):
    def __init__(self, conf,task):
        self.configuration = conf
        self.manager = State(conf)
        self.dataset = self.load_dataset()  # chiamerà quello definito nella sottoclasse
        self.train_loader = None
        self.test_loader = None
        self.dataset_sizes = None
        self.task=task
        self.functions = {
            'augmentation': self.augmentation,
            'undersampling': self.undersampling,
            '': lambda: None
        }
        self.split_functions = {
            'load_split': self.load_dataset,
            'random_split': self.random_split,
        }

    @abstractmethod
    def load_dataset(self):
        pass

    def random_split(self):
        test_size = self.configuration.get(self.task + '_test_split')
        test_size = int(len(self.dataset.images) * test_size)
        train_size = len(self.dataset.images) - test_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        index_train = train_dataset.indices
        index_test = test_dataset.indices
        X_train = [self.dataset.images[i] for i in index_train]
        Y_train = [self.dataset.labels[i] for i in index_train]
        X_test = [self.dataset.images[i] for i in index_test]
        Y_test = [self.dataset.labels[i] for i in index_test]
        self.train_loader = AbsDataset(X_train, Y_train, batch_size=self.configuration.get(self.task + '_batch_size'))
        self.test_loader = AbsDataset(X_test, Y_test, batch_size=self.configuration.get(self.task + '_batch_size'))
        self.dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}





    def data_preprocessing(self):
        self.preprocessing = self.configuration.get(self.task + '_preprocessing')
        split_opt=self.configuration.get(self.task + 'split_option')
        if split_opt!='load_split':
            self.functions.get(self.preprocessing)()
        for c in range(0, self.dataset.get_num_classes()):
            print('Class ' + str(c) + ' number samples ' + str(len(self.dataset.get_class_instances(c)[1])))
        self.split_functions.get(split_opt)()




    def image_generator(self, image, label,code, n):
        rotation = self.configuration.get('rotation_range')
        p_hor = self.configuration.get('flip_hor_probability')
        p_ver = self.configuration.get('flip_ver_probability')
        new_images = list()
        transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p_hor),
            transforms.RandomRotation(degrees=(0, rotation)),
            transforms.RandomVerticalFlip(p=p_ver)
        ])
        for i in range(n):
            result = transformer(image)
            new_images.append(result)
            self.dataset.add_new_instances(result, label,code)

        return new_images

    def augmentation(self):
        classes = list(set(self.dataset.labels))
        target_size = self.dataset.get_max_size()
        for c in classes:
            df_group, indexes = self.dataset.get_class_instances(c)
            current_size = len(df_group)

            if current_size < target_size:
                q, r = divmod(target_size, current_size)
                # combinations=[(random.choice(rotation), random.choice(width), random.choice(height)) for _ in range(repeat)]
                for index in indexes:
                    code=self.dataset.images[index]['image_name']
                    self.image_generator(self.dataset.images[index], int(c),code, q-1)
                for i in range(r):
                    chosen=random.choice(indexes)
                    code = self.dataset.images[chosen]['image_name']
                    self.image_generator(self.dataset.images[chosen], int(c),code, 1)

    def undersampling(self):
        classes = list(set(self.dataset.labels))
        target_size = self.dataset.get_min_size()
        for c in classes:
            df_group, indexes = self.dataset.get_class_instances(c)
            current_size = len(df_group)
            if current_size > target_size:
                q = current_size - target_size
                to_delete = random.sample(indexes, q)
                for idx in sorted(to_delete, reverse=True):
                    del self.dataset.images[idx]
                    del self.dataset.labels[idx]
