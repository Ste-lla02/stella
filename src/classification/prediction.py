from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, accuracy_score, recall_score, roc_auc_score
from src.classification.model import Model
from src.utils.utils import *


class Prediction(Model):

    def evaluate(self, trial, epoch, loss, overall_accuracy, y_true, y_pred):
        cm =confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        f1 = 2 * (precision * specificity) / (precision + specificity) if (precision + specificity) != 0 else 0

        row = dict()
        row['class']='0-1'
        row['epoch'] = epoch
        row['loss'] = loss
        row['overall accuracy'] = overall_accuracy
        row['accuracy'] = accuracy
        row['specificity'] = specificity
        row['precision'] = precision
        row['recall'] = sensitivity
        row['f1_score'] = f1
        row['phase'] = trial
        self.report.loc[len(self.report)] = row
        self.report.to_csv(self.df_name, index=False, sep=';')

        return sensitivity, specificity, accuracy



