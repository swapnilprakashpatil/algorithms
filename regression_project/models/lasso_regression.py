from sklearn.linear_model import Lasso
from .base_model import BaseModel

class LassoRegressionModel(BaseModel):
    """
    Lasso Regression Model
    A regression model that uses L1 regularization for feature selection.
    """
    
    def __init__(self):
        super().__init__('Lasso Regression')
    
    def create_model(self):
        """
        Create and return a Lasso Regression model
        """
        return Lasso()
    
    def get_param_grid(self):
        """
        Return the parameter grid for hyperparameter tuning
        """
        return {
            'alpha': [0.01, 0.1, 1, 10, 100]
        } 