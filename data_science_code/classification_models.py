from sklearn.linear_model import LogisticRegression

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
