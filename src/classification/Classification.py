import torch
import time
import copy
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score

class Classification:
    def __init__(self, model, criterion, optimizer, dataset_sizes, num_epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset_sizes = dataset_sizes
        self.num_epochs = num_epochs
        self.best_model_wts = copy.deepcopy(model.state_dict())  # Salva i pesi migliori
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_model_path = "best_model.pth"  # Percorso dove salvare il modello migliore

    def train(self, train_loader):
        since = time.time()
        self.model.train()

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0
            labels_list, preds_list = [], []

            for inputs, labels in train_loader:
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    labels_list.append(labels.cpu())
                    preds_list.append(preds.cpu())

                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / self.dataset_sizes['train']
            epoch_acc = running_corrects.double() / self.dataset_sizes['train']
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Salva il modello con la migliore accuracy
            if epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                torch.save(self.best_model_wts, self.best_model_path)  # Salva i pesi migliori
                print(f"Best model saved at epoch {epoch}")

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        print(f"Best training accuracy: {self.best_acc:.4f} at epoch {self.best_epoch}")

        # Restituisce il modello con i migliori pesi
        self.model.load_state_dict(self.best_model_wts)
        return self.model

    def test(self, model, test_loader):
        # Carico il modello con i pesi migliori
        model.load_state_dict(torch.load(self.best_model_path))
        model.eval()

        # Verifica se i pesi sono stati caricati correttamente
        loaded_state_dict = torch.load(self.best_model_path)
        for key in model.state_dict():
            if not torch.equal(model.state_dict()[key], loaded_state_dict[key]):
                print("Warning: modello non caricato")
                break

        running_loss = 0.0
        running_corrects = 0

        all_labels = []  # Lista per le etichette reali
        all_preds = []  # Lista per le etichette predette

        for inputs, labels in test_loader:
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        epoch_loss = running_loss / self.dataset_sizes['test']
        epoch_acc = running_corrects.double() / self.dataset_sizes['test']
        print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return epoch_acc, all_labels, all_preds

    def evaluate(self, test_labels, test_preds):
        cm = confusion_matrix(test_labels, test_preds)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(test_labels, test_preds)
        sensitivity = recall_score(test_labels, test_preds)
        specificity = tn / (tn + fp)
        roc_auc = roc_auc_score(test_labels, test_preds)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Sensitivity (Recall): {sensitivity:.4f}')
        print(f'Specificity: {specificity:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')
        print(f'Confusion Matrix:\n{cm}')

        return sensitivity, specificity, accuracy
