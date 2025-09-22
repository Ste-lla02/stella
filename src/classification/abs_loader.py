from abc import ABC, abstractmethod
import torch
from torchvision import transforms
from src.core.core_model import State
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader,Subset
import random
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AbsDataset(Dataset):
    def __init__(self, images, labels, codes=None):
        self.codes = codes
        self.images = images
        self.labels = labels

        # 2) definisco il transform
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.images = [self.transform(img) for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]                  # già tensor [1,224,224]
        y = self.labels[idx]
        y = torch.tensor(y, dtype=torch.long) # scalare, non [y]
        return x, y

    def compute_mean_std(self):
        means = []
        stds = []
        transform_no_norm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        for img in self.images:
            tensor = transform_no_norm(img)      # [1, H, W]
            means.append(tensor.mean(dim=(1, 2)))  # media per canale (qui è 1 valore)
            stds.append(tensor.std(dim=(1, 2)))    # std per canale (1 valore)

        mean = torch.stack(means).mean(dim=0)  # shape [1]
        std  = torch.stack(stds).mean(dim=0)   # shape [1]
        return mean.tolist(), std.tolist()

    def get_num_classes(self):
        classes = list(set(self.labels))
        return len(classes)

    def get_class_instances(self, class_id):
        indexes_class = [i for i, val in enumerate(self.labels) if val == class_id]
        subset = [self.images[i] for i in indexes_class]
        return subset, indexes_class

    def get_max_size(self):
        classes = list(set(self.labels))
        maxv = 0
        for c in classes:
            dim = len(self.get_class_instances(c)[0])
            if dim > maxv:
                maxv = dim
        return maxv

    def get_min_size(self):
        classes = list(set(self.labels))
        minv = len(self.labels)
        for c in classes:
            dim = len(self.get_class_instances(c)[0])
            if dim < minv:
                minv = dim
        return minv

    def add_new_instances(self, image, label, code):
        # Mantieni coerenza: trasformo anche i nuovi campioni
        self.images.append(image)  # tensor [1,H,W]
        self.labels.append(label)
        if self.codes is not None:
            self.codes.append(code)


class AbstractLoader(ABC):
    def __init__(self, conf,task):
        self.configuration = conf
        self.manager = State(conf)
        self.dataset=None
        self.task = task
        self.load_dataset()  # chiamerà quello definito nella sottoclasse
        self.train_loader = None
        self.test_loader = None
        self.dataset_sizes = None
        self.functions = {
            'augmentation': self.augmentation,
            'undersampling': self.undersampling,
            '': lambda: None
        }
        self.split_functions = {
            'load_split': self.load_split,
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
        sub_train = Subset(self.dataset, index_train)
        self.train_loader = DataLoader(sub_train, batch_size=self.configuration.get(self.task + '_batch_size'),
                                       shuffle=True)
        sub_test = Subset(self.dataset, index_test)
        self.test_loader = DataLoader(sub_test, batch_size=self.configuration.get(self.task + '_batch_size'),
                                      shuffle=True)
        self.dataset_sizes = {'train': len(index_train), 'test': len(index_test)}
        self.dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}
    def load_split(self):
        split_file = self.configuration.get('split_file')
        split_df = pd.read_csv(
            split_file,
            sep=None, engine="python",  # inferisce il separatore
            header=None, names=["Record ID", "Group"]
        )
        train_codes = split_df[split_df['Group']=='train']['Record ID'].tolist()
        test_codes = split_df[split_df['Group'] == 'test']['Record ID'].tolist()
        index_train = [i for i, c in enumerate(self.dataset.codes) if c in train_codes]
        index_test = [i for i, c in enumerate(self.dataset.codes) if c in test_codes]
        sub_train=Subset(self.dataset, index_train)
        self.train_loader = DataLoader(sub_train, batch_size=self.configuration.get(self.task + '_batch_size'),
                                       shuffle=True)
        sub_test = Subset(self.dataset, index_test)
        self.test_loader = DataLoader(sub_test, batch_size=self.configuration.get(self.task + '_batch_size'),
                                      shuffle=True)
        self.dataset_sizes = {'train': len(index_train), 'test': len(index_test)}





    def data_preprocessing(self):
        self.preprocessing = self.configuration.get(self.task + '_preprocessing')
        split_opt=self.configuration.get(self.task + '_split_option')
        for c in range(0, self.dataset.get_num_classes()):
            print('Class ' + str(c) + ' number samples ' + str(len(self.dataset.get_class_instances(c)[1])))
        self.split_functions.get(split_opt)()
        self.functions.get(self.preprocessing)()
        print('Training dataset after preprocessing\n')
        for c in range(0, self.train_loader.dataset.dataset.get_num_classes()):
            print('Class ' + str(c) + ' number samples ' + str(len(self.train_loader.dataset.dataset.get_class_instances(c)[1])))





    def image_generator(self, image, label, n):
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
            self.train_loader.dataset.dataset.add_new_instances(result, label, 'AUG')
            #self.dataset.add_new_instances(result, label,code)

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
                    self.image_generator(self.dataset.images[index], int(c), q-1)
                for i in range(r):
                    chosen=random.choice(indexes)
                    self.image_generator(self.dataset.images[chosen], int(c), 1)

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
                    del self.train_loader.dataset.dataset.images[idx]
                    del self.train_loader.datasetdataset.labels[idx]
