from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import numpy as np


regression_models = ["""Regression Models: 
                     Linear Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, AdaBoost, 
                     Support Vector Machine, Neural Network, XGBoost
                     """]
for model in regression_models:
    print(model)


class ModelTrainer:
    def __init__(self, model_type='linear_regression', random_state=None, **kwargs):
        """
        Initializes the ModelTrainer class with the specified model type.

        Parameters:
        - model_type: Type of the model to use (default is 'linear_regression').
        - random_state: Seed for random number generation.
        - kwargs: Additional arguments to pass to the model.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._initialize_model(model_type, random_state, **kwargs)

    def _initialize_model(self, model_type, random_state, **kwargs):
        """
        Initializes the model based on the specified model type.

        Parameters:
        - model_type: Type of the model.
        - random_state: Seed for random number generation.
        - kwargs: Additional arguments to pass to the model.

        Returns:
        - model: Initialized model object.
        """
        model_map = {
            'linear_regression': LinearRegression,
            'knn': KNeighborsRegressor,
            'random_forest': RandomForestRegressor,
            'gbm': GradientBoostingRegressor,
            'adaboost': AdaBoostRegressor,
            'svr': SVR,
            'neural_network': MLPRegressor,
            'xgboost': XGBRegressor
        }

        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_class = model_map[model_type]

        # If the model supports a random_state parameter, pass it.
        if 'random_state' in model_class().get_params():
            return model_class(random_state=random_state, **kwargs)
        else:
            return model_class(**kwargs)

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

    def tune_hyperparameters(self, X, y, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
        """
        Tunes the hyperparameters of the model using GridSearchCV.

        Parameters:
        - X: Features (numpy array or pandas DataFrame).
        - y: Target variable (numpy array or pandas Series).
        - param_grid: Dictionary with parameter names (`str`) as keys and lists of parameter settings to try as values.
        - cv: Number of cross-validation folds (default is 5).
        - scoring: Scoring metric to use for evaluating the model (default is 'neg_mean_squared_error').
        - n_jobs: Number of jobs to run in parallel during GridSearchCV (default is -1, which uses all processors).

        Returns:
        - best_params_: Best parameters found by GridSearchCV.
        """
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def get_feature_importance(self, X=None, y=None):
        """
        Returns the feature importance of the trained model.

        Parameters:
        - X: Features (numpy array or pandas DataFrame). Required for permutation importance.
        - y: Target variable (numpy array or pandas Series). Required for permutation importance.

        Returns:
        - feature_importances: Array of feature importances.
        """
        # Ensure X has feature names (needed for the DataFrame)
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'Feature {i}' for i in range(X.shape[1])]

        if hasattr(self.model, 'feature_importances_'):
            # For models like RandomForest, GradientBoosting, XGBoost
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importances = np.abs(self.model.coef_)
        elif X is not None and y is not None:
            # Use permutation importance for models that do not have built-in feature importance
            result = permutation_importance(self.model, X, y, n_repeats=10, random_state=self.random_state, n_jobs=-1)
            importances = result.importances_mean
        else:
            raise ValueError("Feature importance not available for this model type. Provide X and y for permutation importance.")

        # Create a DataFrame with feature names and importances
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sort the DataFrame by importance in descending order
        feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        return feature_importances_df
