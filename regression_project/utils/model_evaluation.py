import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_residuals(y_true, y_pred, figsize=(10, 6)):
    """
    Plot residuals against predicted values with detailed interpretation
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Add residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    plt.text(0.05, 0.95, 
             f'Mean Residual: {mean_residual:.2f}\nStd Residual: {std_residual:.2f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Add interpretation text
    interpretation = """
    Residual Plot Interpretation:
    - A good model should have residuals randomly scattered around zero
    - The red dashed line represents zero residual (perfect prediction)
    - Look for:
      * Random scatter of points (no patterns)
      * Equal spread above and below zero
      * No obvious trends or curves
    - Warning signs:
      * Fan-shaped pattern (heteroscedasticity)
      * Curved patterns (non-linear relationships)
      * Clusters of points (potential outliers)
    """
    print(interpretation)

def plot_actual_vs_predicted(y_true, y_pred, figsize=(10, 6)):
    """
    Plot actual vs predicted values with detailed interpretation
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    
    # Add R2 score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² Score: {r2:.4f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Add interpretation text
    interpretation = """
    Actual vs Predicted Plot Interpretation:
    - The red dashed line represents perfect predictions (actual = predicted)
    - Points should cluster closely around this line
    - Look for:
      * Points forming a tight cluster around the diagonal
      * Symmetric distribution above and below the line
      * No systematic over/under-prediction
    - Warning signs:
      * Points forming a curve (non-linear relationship)
      * Systematic over/under-prediction in certain ranges
      * Outliers far from the diagonal line
    """
    print(interpretation)

def plot_feature_importance(model, feature_names, figsize=(10, 6)):
    """
    Plot feature importance with detailed interpretation
    """
    try:
        # Get feature importance or coefficients
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = model.coef_
        else:
            print("Model does not support feature importance visualization")
            return
        
        # Create DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        
        # Add value labels
        for i, v in enumerate(importance_df['Importance']):
            plt.text(v, i, f' {v:.4f}', va='center')
        
        plt.tight_layout()
        
        # Add interpretation text
        interpretation = """
        Feature Importance Plot Interpretation:
        - Shows the relative importance of each feature in making predictions
        - Look for:
          * Features with significantly higher importance
          * Features with near-zero importance (potential candidates for removal)
          * Direction of influence (positive/negative coefficients)
        - Warning signs:
          * Features with unexpectedly high importance
          * Features with zero importance that should matter
          * Unstable importance rankings across different runs
        """
        print(interpretation)
        
    except Exception as e:
        print(f"Error plotting feature importance: {str(e)}")

def plot_model_comparison(metrics_dict, figsize=(12, 8)):
    """
    Plot comparison of different models' metrics with detailed interpretation
    """
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    models = list(metrics_dict.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[model][metric] for model in models]
        ax = axes[i]
        
        # Create bar plot
        bars = ax.bar(models, values)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Add interpretation text
    interpretation = """
    Model Comparison Plot Interpretation:
    - Compares multiple models across different metrics
    - Look for:
      * Consistent performance across metrics
      * Models with lowest MSE/RMSE/MAE
      * Models with highest R²
      * Trade-offs between different metrics
    - Warning signs:
      * Large variations in performance across metrics
      * Models performing significantly worse than others
      * Overfitting (high R² but poor generalization)
    """
    print(interpretation)

def evaluate_model(model, X_test, y_test, feature_names=None, model_name=None):
    """
    Complete model evaluation pipeline
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Print metrics
    print(f"\nMetrics for {model_name if model_name else 'Model'}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Residuals
    plt.subplot(2, 2, 1)
    plot_residuals(y_test, y_pred)
    
    # Plot 2: Actual vs Predicted
    plt.subplot(2, 2, 2)
    plot_actual_vs_predicted(y_test, y_pred)
    
    # Plot 3: Feature Importance (if applicable)
    if feature_names is not None:
        plt.subplot(2, 2, 3)
        plot_feature_importance(model, feature_names)
    
    plt.tight_layout()
    
    return metrics

def predict_price(model, feature_names, input_features):
    """
    Make a prediction using the trained model
    """
    # Create input array
    X = np.array([[input_features[feature] for feature in feature_names]])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction 