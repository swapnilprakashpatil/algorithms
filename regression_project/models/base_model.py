from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from config import GRID_SEARCH_SETTINGS

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    This class defines the common interface that all models must implement.
    """
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.best_params = None
    
    @abstractmethod
    def create_model(self):
        """
        Create and return the model instance.
        This method must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def get_param_grid(self):
        """
        Return the parameter grid for hyperparameter tuning.
        This method must be implemented by subclasses.
        """
        pass
    
    def train(self, X_train, y_train):
        """
        Train the model with optional hyperparameter tuning.
        """
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        return self
    
    def train_with_grid_search(self, X_train, y_train):
        """
        Train the model with hyperparameter tuning using grid search.
        """
        param_grid = self.get_param_grid()
        grid_search = GridSearchCV(
            estimator=self.create_model(),
            param_grid=param_grid,
            scoring=GRID_SEARCH_SETTINGS['scoring'],
            cv=GRID_SEARCH_SETTINGS['cv'],
            n_jobs=GRID_SEARCH_SETTINGS['n_jobs']
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"\nBest parameters for {self.name}:")
        print(self.best_params)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """
        Get feature importance if the model supports it.
        """
        try:
            return self.model.coef_
        except AttributeError:
            print(f"{self.name} doesn't support feature importance.")
            return None 