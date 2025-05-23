import pandas as pd
from src.utils.utils import *
import time
from abc import ABC, abstractmethod
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix, accuracy_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import copy


class EarlyStopping:
    def __init__(self,conf,task):
        self.task=task
        self.patience = conf.get(self.task+'_patience')
        self.verbose = conf.get(self.task+'_verbose')
        self.delta = conf.get(self.task+'_delta')
        self.path = conf.get(self.task+'_model_path')
        self.trace_func = print

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


class Model:
    def __init__(self, model, criterion, optimizer,device,conf, loader,task):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loader = loader
        self.task=task
        self.dataset_sizes = loader.dataset_sizes
        self.preprocessing=conf.get(self.task+'_preprocessing')
        self.num_epochs = conf.get(self.task+'_num_epochs')
        self.device = device
        self.earlystopping=EarlyStopping(conf,self.task)
        self.df_path=os.path.join(conf.get('reportcsv'),self.task)
        create_folder(self.df_path)
        self.df_name=self.df_path+'\\report_label_'+ str(self.preprocessing[0:3])+'_'+str(self.num_epochs)+'.csv'
        self.results=dict()
        self.report=pd.DataFrame(columns=['epoch','phase','loss','overall accuracy','class','accuracy','specificity', 'precision','recall','f1_score'])
        self.train_losses=list()
        self.train_acc=list()
        self.validation_losses = list()
        self.validation_acc=list()
        self.full_graph_path = os.path.join('output', 'graphs',self.task, self.preprocessing + '_epoch_' + str(self.num_epochs))
        create_folder(self.full_graph_path)
        self.performance_report=open(self.df_path+'\\performance_report.txt','w')

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
            self.train_acc.append(epoch_acc)
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            y_true_train = torch.cat(labels_list)
            y_pred_train = torch.cat(preds_list)
            # TRAINING EVALUATION
            self.evaluate('training', epoch, epoch_loss, epoch_acc, y_true_train, y_pred_train)

            # VALIDATION EVALUATION
            val_loss, val_acc, y_true_test, y_pred_test = self.test(self.model, epoch=epoch)
            self.validation_losses.append(val_loss)
            self.validation_acc.append(val_acc)
            self.evaluate('validating', epoch, val_loss, val_acc, y_true_test, y_pred_test)

            # EARLY STOPPING
            self.earlystopping(torch.tensor(epoch_loss), self.model, epoch, y_true=y_true_test, y_pred=y_pred_test)

            check = self.earlystopping.early_stop
            epoch += 1

        if check:
            self.performance_report.write(f"Early stopping at epoch {epoch - 1}\n")

        # LOAD BEST MODEL
        self.model.load_state_dict(self.earlystopping.best_model_wts)
        self.performance_report.write(f"Best model from epoch {self.earlystopping.best_epoch} loaded.\n")

        # FINAL EVALUATION
        time_elapsed = time.time() - since
        self.performance_report.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        self.performance_report.write(f"Best validation loss: {-self.earlystopping.best_score:.4f} at epoch {self.earlystopping.best_epoch}\n")
        self.performance_report.close()
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



    @abstractmethod
    def evaluate(self, trial,epoch,loss,overall_accuracy,y_true, y_pred):
       pass

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
        plt.plot(self.train_losses, label='Training Loss', color='blue', marker='o')
        plt.plot(self.validation_losses, label='Validation Loss', color='orange', marker='o')
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
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_acc, label='Training Accuracy', color='blue', marker='o')
        plt.plot(self.validation_acc, label='Validation Accuracy', color='orange', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.tick_params(axis='y', labelcolor='black')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        filename = os.path.join(self.full_graph_path, f"accuracy_graph.pdf")
        plt.savefig(filename, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
