import torch.optim as optim
import torch.nn as nn
from torchvision import models
import torch
class ResNet():

    def __init__(self,conf,loader,task):
        self.conf=conf
        self.loader=loader
        self.task=task
        self.criterion=None
        self.optimizer=None
        self.model=None
        self.path=self.conf.get(self.task+'_model_path')
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def __call__(self, *args, **kwargs):
        model = models.resnet18()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512, self.loader.dataset.get_num_classes())
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=self.conf.get(self.task+'_learning_rate'))

    def load(self) -> bool:
        retval = False
        try:
            checkpoint = torch.load(self.path, weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            #checkpoint = torch.load(self.path, map_location=self.device)
            #self.model.load_state_dict(checkpoint["model_state_dict"])
            #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            retval = True
        except Exception as e:
            print(e)
        return retval
