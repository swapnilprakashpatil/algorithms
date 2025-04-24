"""
Configuration settings for the machine learning project.
This file contains common settings that can be reused across different algorithm implementations.
"""

# Data settings
DATA_SETTINGS = {
    'test_size': 0.2,
    'random_state': 42,
    'target_column': 'price'
}

# Preprocessing settings
PREPROCESSING_SETTINGS = {
    'missing_value_strategy': 'mean',
    'outlier_threshold': 3,
    'scaling_method': 'standard'
}

# Model evaluation settings
EVALUATION_SETTINGS = {
    'metrics': ['MSE', 'RMSE', 'MAE', 'R2'],
    'cross_validation_folds': 5
}

# Visualization settings
VISUALIZATION_SETTINGS = {
    'figure_size': (10, 6),
    'correlation_matrix_size': (12, 8),
    'distribution_plot_size': (15, 10),
    'scatter_matrix_size': (15, 15)
}

# Grid search settings
GRID_SEARCH_SETTINGS = {
    'scoring': 'neg_mean_squared_error',
    'cv': 5,
    'n_jobs': -1
} 