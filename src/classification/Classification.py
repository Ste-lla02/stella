import pandas as pd
import torch
import time
import copy
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix, accuracy_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import copy

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path='best_model.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = torch.tensor(float('inf'))
        self.best_model_wts = None
        self.best_epoch = -1

        self.y_true_best = None
        self.y_pred_best = None

    def __call__(self, val_loss, model, epoch, y_true=None, y_pred=None):
        score = -val_loss  # Per minimizzare la loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, y_true, y_pred)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'No improvement in validation loss for {self.counter} epoch(s).')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, y_true, y_pred)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, y_true, y_pred):
        if self.verbose:
            self.trace_func(f'Validation loss decreased. Saving model at epoch {epoch} ...')
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_epoch = epoch
        self.val_loss_min = val_loss.clone() if isinstance(val_loss, torch.Tensor) else torch.tensor(val_loss)
        torch.save(self.best_model_wts, self.path)
        self.y_true_best = y_true
        self.y_pred_best = y_pred


class Classification:
    def __init__(self, model, criterion, optimizer,device,conf, loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loader = loader
        self.dataset_sizes = loader.dataset_sizes
        self.preprocessing=conf.get('preprocessing')
        self.num_epochs = conf.get('num_epochs')
        self.device = device
        self.earlystopping=EarlyStopping(conf)  # Percorso dove salvare il modello migliore
        self.label_dict = {
            '0': 'bladder',
            '1': 'urethra',
            '2': 'bladder_and_urethra',
            '3': 'other'
        }
        self.df_name=conf.get('reportcsv')+'report_label_'+ str(self.preprocessing[0:3])+'_'+str(self.num_epochs)+'.csv'
        self.results=dict()
        self.report=pd.DataFrame(columns=['epoch','phase','loss','overall accuracy','class','accuracy','specificity', 'precision','recall','f1_score'])
        self.train_losses=list()
        self.validation_losses = list()
        self.full_graph_path = os.path.join('output', 'graphs', self.preprocessing + '_epoch_' + str(self.num_epochs))

    def train(self):
        since = time.time()
        self.model.to(self.device)
        self.model.train()

        epoch = 0
        check = False

        # TRAINING WITH EPOCHS AND EARLY STOPPING
        while epoch < self.num_epochs and not check:
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0
            labels_list, preds_list = [], []

            for inputs, labels in self.loader.train_loader:
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
            epoch_acc = epoch_acc.item()
            self.train_losses.append(epoch_loss)
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            y_true_train = torch.cat(labels_list)
            y_pred_train = torch.cat(preds_list)
            # TRAINING EVALUATION
            self.evaluate_multilabels('training', epoch, epoch_loss, epoch_acc, y_true_train, y_pred_train)

            # VALIDATION EVALUATION
            val_loss, val_acc, y_true_test, y_pred_test = self.test(self.model, epoch=epoch)
            self.validation_losses.append(val_loss)
            self.evaluate_multilabels('validating', epoch, val_loss, val_acc, y_true_test, y_pred_test)

            # EARLY STOPPING
            self.earlystopping(torch.tensor(val_loss), self.model, epoch, y_true=y_true_test, y_pred=y_pred_test)

            check = self.earlystopping.early_stop
            epoch += 1

        if check:
            print(f"Early stopping at epoch {epoch - 1}")

        # LOAD BEST MODEL
        self.model.load_state_dict(self.earlystopping.best_model_wts)
        print(f"Best model from epoch {self.earlystopping.best_epoch} loaded.")

        # FINAL EVALUATION
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f"Best validation loss: {-self.earlystopping.best_score:.4f} at epoch {self.earlystopping.best_epoch}")
        self.plot_losses()

        # CONFUSION MATRIX
        self.confusion_matrics_graph(self.earlystopping.y_true_best, self.earlystopping.y_pred_best)

        return self.model

    def test(self, model, epoch=None):
        model.to(self.device)
        model.eval()

        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []

        for inputs, labels in self.loader.test_loader:
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
        epoch_acc=epoch_acc.item()

        if epoch is not None:
            print(f'Epoch {epoch} - Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        else:
            print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return epoch_loss, epoch_acc, all_labels, all_preds




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

    def evaluate_multilabels(self, trial,epoch,loss,overall_accuracy,y_true, y_pred):
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        metrics = dict()
        for idx, label_mcm in enumerate(mcm):
            label_name = self.label_dict[str(idx)]


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
        for key, values in metrics.items():
            row = dict()
            row['phase'] = trial
            row['epoch'] = epoch
            row['loss'] = loss
            row['overall accuracy']=overall_accuracy
            row['class'] = key
            row.update(values)
            self.report.loc[len(self.report)] = row
        self.report.to_csv(self.df_name, index=False, sep=';')
        return metrics

    def evaluation_graph(self):

        if not os.path.exists(self.full_graph_path):
            os.makedirs(self.full_graph_path)
        selected_trial = 'validating'
        classes = self.report['class'].unique()
        trial=selected_trial
        for selected_class in classes:
            class_1 = self.report[(self.report['class'] == selected_class) & (self.report['phase'] == selected_trial)]
            plt.plot(class_1['epoch'], class_1['accuracy'], label='accuracy')
            plt.plot(class_1['epoch'], class_1['specificity'], label='specificity')
            plt.plot(class_1['epoch'], class_1['precision'], label='precision')
            plt.plot(class_1['epoch'], class_1['recall'], label='recall')
            plt.plot(class_1['epoch'], class_1['f1_score'], label='f1_score')
            if (selected_trial == 'validating'):
                trial = 'Validation'
            plt.title(trial + ' phase metrics evaluation for label ' + selected_class)
            plt.xlabel('Epochs')
            plt.ylabel('Metrics')
            plt.legend()
            plt.savefig(
                os.path.join(self.full_graph_path, trial + '_Class_' + selected_class + '.pdf'),
                format='pdf',
                dpi=600,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True
            )
            plt.close()



    def confusion_matrics_graph(self,y_true, y_pred):
        if not os.path.exists(self.full_graph_path):
            os.makedirs(self.full_graph_path)

        for idx, class_name in self.label_dict.items():
            y_true_binary = (y_true == int(idx)).astype(int)
            y_pred_binary = (y_pred == int(idx)).astype(int)
            id_number=str(int(idx)+1)
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            plt.figure(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['not class ' + id_number, 'class '+id_number],
                        yticklabels=['not class ' + id_number, 'class '+id_number])
            plt.title(f'Confusion Matrix Class: {id_number}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')


            filename = os.path.join(self.full_graph_path, f"confusion_matrix_{class_name}.pdf")
            plt.savefig(filename,format='pdf',
                dpi=600,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True)

            plt.close()



    def plot_losses(self):
        if not os.path.exists(self.full_graph_path):
            os.makedirs(self.full_graph_path)
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Training Loss', color='blue', marker='-')
        plt.plot(self.validation_losses, label='Validation Loss', color='orange', marker='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        filename = os.path.join(self.full_graph_path, f"losses_graph.pdf")
        plt.savefig(filename, format='pdf',
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)

        plt.close()
