import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

def generate_sample_data(
    n_samples=50000,
    n_features=5,
    noise=10.0,
    feature_noise=0.1,
    add_squared=True,
    add_cubed=True,
    add_interactions=True,
    add_categorical=True,
    n_categories=4,
    missing_percentage=1.0
):
    """
    Generate a synthetic dataset for regression analysis.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features to generate
        noise (float): Standard deviation of the noise added to the target
        feature_noise (float): Standard deviation of noise added to features
        add_squared (bool): Whether to add squared terms
        add_cubed (bool): Whether to add cubed terms
        add_interactions (bool): Whether to add interaction terms
        add_categorical (bool): Whether to add categorical features
        n_categories (int): Number of categories if add_categorical is True
        missing_percentage (float): Percentage of missing values to add
        
    Returns:
        pd.DataFrame: Generated dataset
    """
    # Generate regression dataset
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add non-linear relationships
    if add_squared:
        for i in range(min(3, n_features)):  # Add squared terms for first 3 features
            df[f'feature_{i+1}_squared'] = df[f'feature_{i+1}'] ** 2
    
    if add_cubed:
        for i in range(min(2, n_features)):  # Add cubed terms for first 2 features
            df[f'feature_{i+1}_cubed'] = df[f'feature_{i+1}'] ** 3
    
    if add_interactions:
        # Add interaction terms between consecutive features
        for i in range(n_features - 1):
            df[f'feature_{i+1}_{i+2}_interaction'] = df[f'feature_{i+1}'] * df[f'feature_{i+2}']
    
    # Add noise to features
    for col in df.columns:
        df[col] = df[col] + np.random.normal(0, feature_noise, n_samples)
    
    # Add target variable
    df['target'] = y
    
    # Add categorical features
    if add_categorical:
        categories = [chr(65 + i) for i in range(n_categories)]  # A, B, C, ...
        df['category'] = np.random.choice(categories, size=n_samples)
        
        # Add another categorical feature with different distribution
        weights = np.random.dirichlet(np.ones(n_categories))
        df['category_weighted'] = np.random.choice(categories, size=n_samples, p=weights)
    
    # Add missing values
    if missing_percentage > 0:
        mask = np.random.random(df.shape) < (missing_percentage / 100)
        df = df.mask(mask)
    
    return df

if __name__ == "__main__":
    # Generate the dataset
    df = generate_sample_data()
    
    # Save to CSV
    df.to_csv('sample_data.csv', index=False)
    print(f"Generated dataset with {len(df)} samples and {len(df.columns)} columns")
    print("Dataset saved to sample_data.csv") 