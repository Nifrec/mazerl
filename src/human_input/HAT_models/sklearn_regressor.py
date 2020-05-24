import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class sklearn_regressor():
    """
    Implementation based on the Model interface.
    Simple SGDRegressor model.
    """

    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data

    def train(self):
        """train the model"""
        # setup the regression model pipeline
        self.reg = make_pipeline(StandardScaler(),
                                 SGDRegressor(max_iter=1000, tol=1e-3))

        # Fit the regression model using the training data
        self.reg.fit(X, y)

    def forward(self, state):
        """perform a forward pass of the model"""
        # Predict the labels of the test set
        y_pred = self.reg.predict(state)
