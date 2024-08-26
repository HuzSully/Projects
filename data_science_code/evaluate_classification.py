import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

class ClassificationEvaluator:
    def __init__(self, y_true, models=None, average='binary'):
        """
        Initializes the evaluator with the true labels and multiple models' predictions.

        Parameters:
            y_true (list or array): The true labels.
            models (dict): A dictionary where keys are model names and values are dictionaries with 'y_pred' and optional 'y_prob'.
            average (str): The averaging method for multiclass classification. Default is 'binary'.
                           For multiclass classification, use 'macro', 'micro', 'weighted', etc.
        """
        self.y_true = y_true
        self.models = models if models is not None else {}
        self.average = average

    def add_model(self, model_name, y_pred, y_prob=None):
        """
        Adds a model's predictions and probabilities to the evaluator.

        Parameters:
            model_name (str): The name of the model.
            y_pred (list or array): The predicted labels.
            y_prob (list or array): The predicted probabilities (optional).
        """
        self.models[model_name] = {
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    def evaluate_model(self, model_name):
        """Calculates all evaluation metrics for a specific model and returns them as a dictionary."""
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(self.y_true, self.models[model_name]['y_pred']),
            'Precision': precision_score(self.y_true, self.models[model_name]['y_pred'], average=self.average),
            'Recall': recall_score(self.y_true, self.models[model_name]['y_pred'], average=self.average),
            'F1 Score': f1_score(self.y_true, self.models[model_name]['y_pred'], average=self.average),
            'Confusion Matrix': self.confusion_matrix_as_str(model_name)
        }
        if self.models[model_name]['y_prob'] is not None:
            metrics['ROC AUC'] = roc_auc_score(self.y_true, self.models[model_name]['y_prob'], average=self.average)
        else:
            metrics['ROC AUC'] = None
        
        return metrics

    def confusion_matrix_as_str(self, model_name):
        """Returns the confusion matrix as a string to include in the DataFrame."""
        cm = confusion_matrix(self.y_true, self.models[model_name]['y_pred'])
        return '\n'.join(['\t'.join(map(str, row)) for row in cm])

    def evaluate_all_models(self):
        """Calculates all evaluation metrics for all models and returns them as a DataFrame."""
        results = []
        for model_name in self.models:
            metrics = self.evaluate_model(model_name)
            results.append(metrics)
        return pd.DataFrame(results)

# Example usage:
# y_true = [0, 1, 0, 1]
# model1_predictions = {'y_pred': [0, 1, 0, 0], 'y_prob': [0.3, 0.7, 0.2, 0.4]}
# model2_predictions = {'y_pred': [1, 1, 0, 1], 'y_prob': [0.6, 0.8, 0.4, 0.9]}
#
# evaluator = ClassificationEvaluator(y_true)
# evaluator.add_model('Model 1', model1_predictions['y_pred'], model1_predictions['y_prob'])
# evaluator.add_model('Model 2', model2_predictions['y_pred'], model2_predictions['y_prob'])
#
# results_df = evaluator.evaluate_all_models()
# print(results_df)
