from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class LogisticRegressionModel:
    def __init__(self, random_state=None, **kwargs):
        """
        Initializes the LogisticRegressionModel class with optional parameters.

        Parameters:
        - random_state: Seed for random number generation.
        - kwargs: Additional arguments to pass to the LogisticRegression model.
        """
        self.model = LogisticRegression(random_state=random_state, **kwargs)
        self.random_state = random_state

    def train(self, X, y):
        """
        Trains the logistic regression model using the provided dataset.

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

    def get_coefficients(self):
        """
        Returns the coefficients of the logistic regression model.

        Returns:
        - coefficients: Coefficients of the features.
        - intercept: Intercept of the model.
        """
        return self.model.coef_, self.model.intercept_

    def set_params(self, **params):
        """
        Sets the parameters of the logistic regression model.

        Parameters:
        - params: Dictionary of parameters to set.
        """
        self.model.set_params(**params)

    def get_params(self):
        """
        Gets the parameters of the logistic regression model.

        Returns:
        - params: Dictionary of the model's parameters.
        """
        return self.model.get_params()
    

class RandomForestModel:
    def __init__(self, random_state=None, **kwargs):
        """
        Initializes the RandomForestModel class with optional parameters.

        Parameters:
        - random_state: Seed for random number generation.
        - kwargs: Additional arguments to pass to the RandomForestClassifier model.
        """
        self.model = RandomForestClassifier(random_state=random_state, **kwargs)
        self.random_state = random_state

    def train(self, X, y):
        """
        Trains the random forest model using the provided dataset.

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

    def get_feature_importances(self):
        """
        Returns the feature importances of the random forest model.

        Returns:
        - feature_importances_: Feature importances of the model.
        """
        return self.model.feature_importances_

    def set_params(self, **params):
        """
        Sets the parameters of the random forest model.

        Parameters:
        - params: Dictionary of parameters to set.
        """
        self.model.set_params(**params)

    def get_params(self):
        """
        Gets the parameters of the random forest model.

        Returns:
        - params: Dictionary of the model's parameters.
        """
        return self.model.get_params()

    def tune_hyperparameters(self, X, y, param_grid, cv=5, scoring='accuracy', n_jobs=-1):
        """
        Tunes the hyperparameters of the random forest model using GridSearchCV.

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

# Example Usage:
# rf_model = RandomForestModel(random_state=42)
# rf_model.train(X_train, y_train)
# predictions = rf_model.predict(X_test)
# best_params = rf_model.tune_hyperparameters(X_train, y_train, param_grid={'n_estimators': [100, 200], 'max_depth': [10, 20]})

