from sklearn.linear_model import LinearRegression
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    """
    Linear Regression Model
    A simple linear regression model that fits a linear equation to the data.
    """
    
    def __init__(self):
        super().__init__('Linear Regression')
    
    def create_model(self):
        """
        Create and return a Linear Regression model
        """
        return LinearRegression()
    
    def get_param_grid(self):
        """
        Return the parameter grid for hyperparameter tuning
        Linear Regression has no hyperparameters to tune
        """
        return {} 