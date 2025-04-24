from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__('Linear Regression')
    
    def create_model(self):
        return LinearRegression()
    
    def get_param_grid(self):
        return {}

class PolynomialRegressionModel(BaseModel):
    def __init__(self, degree=2):
        super().__init__('Polynomial Regression')
        self.degree = degree
    
    def create_model(self):
        return Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree)),
            ('linear', LinearRegression())
        ])
    
    def get_param_grid(self):
        return {
            'poly__degree': [2, 3, 4]
        }

class RidgeRegressionModel(BaseModel):
    def __init__(self):
        super().__init__('Ridge Regression')
    
    def create_model(self):
        return Ridge()
    
    def get_param_grid(self):
        return {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }

class LassoRegressionModel(BaseModel):
    def __init__(self):
        super().__init__('Lasso Regression')
    
    def create_model(self):
        return Lasso()
    
    def get_param_grid(self):
        return {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }

class ElasticNetRegressionModel(BaseModel):
    def __init__(self):
        super().__init__('Elastic Net Regression')
    
    def create_model(self):
        return ElasticNet()
    
    def get_param_grid(self):
        return {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        } 