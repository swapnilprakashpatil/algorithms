from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from .base_model import BaseModel

class PolynomialRegressionModel(BaseModel):
    """
    Polynomial Regression Model
    A regression model that fits a polynomial equation to the data.
    """
    
    def __init__(self, degree=2):
        super().__init__('Polynomial Regression')
        self.degree = degree
    
    def create_model(self):
        """
        Create and return a Polynomial Regression model
        """
        return Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree)),
            ('linear', LinearRegression())
        ])
    
    def get_param_grid(self):
        """
        Return the parameter grid for hyperparameter tuning
        """
        return {
            'poly__degree': [2, 3, 4]
        } 