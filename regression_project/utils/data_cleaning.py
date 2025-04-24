import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy import stats

def load_data(file_path):
    """
    Load data from a CSV file
    """
    return pd.read_csv(file_path)

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset
    Parameters:
    - df: pandas DataFrame
    - strategy: imputation strategy ('mean', 'median', 'most_frequent')
    """
    imputer = SimpleImputer(strategy=strategy)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def detect_outliers(df, threshold=3):
    """
    Detect and handle outliers using Z-score method
    Parameters:
    - df: pandas DataFrame
    - threshold: Z-score threshold for outlier detection
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        df[col] = np.where(z_scores > threshold, np.nan, df[col])
    return df

def scale_features(df, method='standard'):
    """
    Scale numerical features
    Parameters:
    - df: pandas DataFrame
    - method: scaling method ('standard' or 'minmax')
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def encode_categorical(df):
    """
    Encode categorical variables using one-hot encoding
    Parameters:
    - df: pandas DataFrame
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_cols = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cols, 
                                columns=encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    return df

def preprocess_data(df, target_column):
    """
    Preprocess the data for regression analysis
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    try:
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Scale features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        return X, y
        
    except Exception as e:
        raise ValueError(f"Error in data preprocessing: {str(e)}") 