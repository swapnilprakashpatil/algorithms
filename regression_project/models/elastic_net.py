from sklearn.linear_model import ElasticNet
from .base_model import BaseModel

class ElasticNetRegressionModel(BaseModel):
    """
    Elastic Net Regression Model
    A regression model that combines L1 and L2 regularization.
    """
    
    def __init__(self):
        super().__init__('Elastic Net Regression')
    
    def create_model(self):
        """
        Create and return an Elastic Net Regression model
        """
        return ElasticNet()
    
    def get_param_grid(self):
        """
        Return the parameter grid for hyperparameter tuning
        """
        return {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        } 