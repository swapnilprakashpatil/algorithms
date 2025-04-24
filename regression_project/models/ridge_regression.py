from sklearn.linear_model import Ridge
from .base_model import BaseModel

class RidgeRegressionModel(BaseModel):
    """
    Ridge Regression Model
    A regression model that uses L2 regularization to prevent overfitting.
    """
    
    def __init__(self):
        super().__init__('Ridge Regression')
    
    def create_model(self):
        """
        Create and return a Ridge Regression model
        """
        return Ridge()
    
    def get_param_grid(self):
        """
        Return the parameter grid for hyperparameter tuning
        """
        return {
            'alpha': [0.01, 0.1, 1, 10, 100]
        } 