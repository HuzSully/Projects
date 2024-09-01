from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import pandas as pd

classification_models = ['Classification Models: ''GBMModel', 'SVMModel', 'NaiveBayesModel', 'LogisticRegressionModel', 'RandomForestModel'
                         ,'KNNModel', 'AdaBoostModel', 'QDAModel', 'LDAModel', 'XGBoostModel', 'NeuralNetworkModel']
for model in classification_models:
    print(model)

class ModelTrainer:
    def __init__(self, model_class, random_state=None, **kwargs):
        """
        Initializes the ModelTrainer class with a specific model class and optional parameters.

        Parameters:
        - model_class: The class of the model to initialize (e.g., LogisticRegression, RandomForestClassifier).
        - random_state: Seed for random number generation (optional, depends on the model).
        - kwargs: Additional arguments to pass to the model.
        """
        self.model = model_class(random_state=random_state, **kwargs) if 'random_state' in model_class().get_params() else model_class(**kwargs)
        self.random_state = random_state

    def train(self, X, y):
        """
        Trains the model using the provided dataset.

        Parameters:
        - X: Features (numpy array or pandas DataFrame).
        - y: Target variable (numpy array or pandas Series).
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts the target variable for the given features.

        Parameters:
        - X: Features to predict the target variable for.

        Returns:
        - Predicted target variable (numpy array).
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts the probability of the target variable for the given features.

        Parameters:
        - X: Features to predict the target variable probabilities for.

        Returns:
        - Predicted probabilities (numpy array).
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("This model does not support probability predictions.")

    def set_params(self, **params):
        """
        Sets the parameters of the model.

        Parameters:
        - params: Dictionary of parameters to set.
        """
        self.model.set_params(**params)

    def get_params(self):
        """
        Gets the parameters of the model.

        Returns:
        - params: Dictionary of the model's parameters.
        """
        return self.model.get_params()

    def tune_hyperparameters(self, X, y, param_grid, cv=5, scoring='accuracy', n_jobs=-1):
        """
        Tunes the hyperparameters of the model using GridSearchCV.

        Parameters:
        - X: Features (numpy array or pandas DataFrame).
        - y: Target variable (numpy array or pandas Series).
        - param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
        - cv: Number of cross-validation folds (default is 5).
        - scoring: Scoring metric to use for evaluating the model (default is 'accuracy').
        - n_jobs: Number of jobs to run in parallel during GridSearchCV (default is -1, which uses all processors).

        Returns:
        - best_params_: Best parameters found by GridSearchCV.
        """
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

