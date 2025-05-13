from abc import ABC, abstractmethod
import torch
from torchvision import transforms
from src.core.core_model import State
from torch.utils.data import Dataset

class MaskDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        #check 44 from wherw
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

    def get_min_size(self):
        classes = list(set(self.labels))
        min=len(self.labels)
        for c in classes:
            dim=len(self.get_class_instances(c)[0])
            if(dim<min):
                min=dim
        return min
    def add_new_instances(self,image, label):
        self.images.append(image)
        self.labels.append(label)

class AbstractLoader(ABC):
    def __init__(self, conf):
        self.configuration = conf
        self.manager = State(conf)
        self.dataset = self.load_mask_dataset()  # chiamerà quello definito nella sottoclasse
        self.train_loader = None
        self.test_loader = None
        self.dataset_sizes = None
        self.functions = {
            'augmentation': self.augmentation,
            'undersampling': self.undersampling,
            '': lambda: None
        }

    @abstractmethod
    def load_mask_dataset(self):
        pass

    @abstractmethod
    def augmentation(self):
        pass

    @abstractmethod
    def undersampling(self):
        pass
