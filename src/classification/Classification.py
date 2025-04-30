import torch
import time
import copy
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix, accuracy_score, recall_score, roc_auc_score

class Classification:
    def __init__(self, model, criterion, optimizer, dataset_sizes, num_epochs,device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset_sizes = dataset_sizes
        self.num_epochs = num_epochs
        self.best_model_wts = copy.deepcopy(model.state_dict())  # Salva i pesi migliori
        self.best_acc = 0.0
        self.best_epoch = 0
        self.device = device
        self.best_model_path = "best_model.pth"  # Percorso dove salvare il modello migliore
        self.label_dict = {
            '0': 'bladder',
            '1': 'urethra',
            '2': 'bladder_and_urethra',
            '3': 'other'
        }
        self.report_txt = open('output/label_classification_report.txt', 'w')

    def train(self, train_loader):
        since = time.time()
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0
            labels_list, preds_list = [], []

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
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

            if epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                torch.save(self.best_model_wts, self.best_model_path)
                print(f"Best model saved at epoch {epoch}")

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f"Best training accuracy: {self.best_acc:.4f} at epoch {self.best_epoch}")

        self.model.load_state_dict(self.best_model_wts)
        return self.model

    def test(self, model, test_loader):
        model.load_state_dict(torch.load(self.best_model_path))
        model.to(self.device)
        model.eval()

        loaded_state_dict = torch.load(self.best_model_path)
        for key in model.state_dict():
            if not torch.equal(model.state_dict()[key].cpu(), loaded_state_dict[key].cpu()):
                print("Warning: modello non caricato correttamente")
                break

        running_loss = 0.0
        running_corrects = 0

        all_labels = []
        all_preds = []

        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

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
        cm = multilabel_confusion_matrix(test_labels, test_preds)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(test_labels, test_preds)
        sensitivity = recall_score(test_labels, test_preds)
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        roc_auc = roc_auc_score(test_labels, test_preds)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Sensitivity (Recall): {sensitivity:.4f}')
        print(f'Specificity: {specificity:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')
        print(f'Confusion Matrix:\n{cm}')

        return sensitivity, specificity, accuracy

    def evaluate_multilabels(self, y_true, y_pred, report_path="report.txt"):
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        metrics = dict()

        with open(report_path, 'w') as self.report_txt:
            for idx, label_mcm in enumerate(mcm):
                label_name = self.label_dict.get(idx, f"Label {idx}")
                self.report_txt.write(f"Confusion Matrix for {label_name}: \n")
                self.report_txt.write(str(label_mcm))
                self.report_txt.write('\n')

                TP = label_mcm[1, 1]
                TN = label_mcm[0, 0]
                FP = label_mcm[0, 1]
                FN = label_mcm[1, 0]

                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
                accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

                metrics[label_name] = {
                    'accuracy': accuracy,
                    'specificity': specificity,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }

                self.report_txt.write(f'Metrics for class "{label_name}":\n')
                for met, val in metrics[label_name].items():
                    self.report_txt.write(f'{met}: {val:.4f}\n')
                self.report_txt.write('\n')
                self.report_txt.close()

        print(f"Multilabel evaluation report saved to: {report_path}")


