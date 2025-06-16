import torch
import os
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from ImageDataset import ImageDataset
from classification import Classification

if __name__ == "__main__":
    torch.manual_seed(1)
    cwd = os.getcwd()

    dataset_path = "dataset/non-segmented"
    batch_size = 4
    test_split = 0.2
    num_epochs = 1
    learning_rate = 1e-4

    data = ImageDataset(dataset_path, batch_size=batch_size, test_split=test_split)
    train_loader, test_loader, data_sizes, class_names = data.load_data()

    print(f"Train set size: {data_sizes['train']}")
    print(f"Test set size: {data_sizes['test']}")
    print(f"Classi: {class_names}")

    ### riaddestro tutta la ResNet ###
    model = models.resnet50(pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    classifier = Classification(model, criterion, optimizer, data_sizes, num_epochs)
    best_model = classifier.train(train_loader)
    epoch_acc, labels_list, preds_list = classifier.test(best_model, test_loader)

    # Valutazione delle prestazioni del modello
    classifier.evaluate(labels_list, preds_list)