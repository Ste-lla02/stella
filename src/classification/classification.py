
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, accuracy_score, recall_score, roc_auc_score
from src.classification.model import Model

class Classification(Model):


    def evaluate(self, trial, epoch, loss, overall_accuracy, y_true, y_pred):
        self.label_dict = {
            '0': 'bladder',
            '1': 'urethra',
            '2': 'bladder_and_urethra',
            '3': 'other'
        }
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
            row['overall accuracy'] = overall_accuracy
            row['class'] = key
            row.update(values)
            self.report.loc[len(self.report)] = row
        self.report.to_csv(self.df_name, index=False, sep=';')
        return metrics




