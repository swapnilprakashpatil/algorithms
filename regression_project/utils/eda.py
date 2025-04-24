import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def get_data_summary(df):
    """
    Generate comprehensive statistical summary of the dataset
    """
    st.subheader("Dataset Summary")
    
    # Basic Information
    st.write("**Basic Information:**")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    # Data Types
    st.write("\n**Data Types:**")
    st.write(df.dtypes)
    
    # Missing Values
    st.write("\n**Missing Values:**")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    st.write(missing_df)
    
    # Descriptive Statistics
    st.write("\n**Descriptive Statistics:**")
    st.write(df.describe())
    
    # Unique Values
    st.write("\n**Unique Values per Column:**")
    unique_values = df.nunique()
    st.write(unique_values)

def plot_correlation_matrix(df):
    """
    Plot correlation matrix heatmap
    """
    st.subheader("Correlation Matrix")
    
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if len(numerical_df.columns) > 0:
        # Calculate correlation matrix
        corr = numerical_df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        
        # Interpretation
        st.markdown("""
        **Correlation Matrix Interpretation:**
        - Shows the relationship between numerical features
        - Values range from -1 to 1
        - Look for:
          * Strong positive correlations (values close to 1)
          * Strong negative correlations (values close to -1)
          * Weak or no correlations (values close to 0)
        - Warning signs:
          * High correlation between features (potential multicollinearity)
          * Unexpected correlations that need investigation
        """)
    else:
        st.warning("No numerical columns found for correlation analysis")

def plot_distributions(df):
    """
    Plot distributions of numerical features
    """
    st.subheader("Feature Distributions")
    
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if len(numerical_df.columns) > 0:
        # Create subplots
        n_cols = 2
        n_rows = (len(numerical_df.columns) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_df.columns):
            ax = axes[i]
            sns.histplot(data=df, x=col, kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            
            # Add skewness and kurtosis
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            ax.text(0.05, 0.95, 
                   f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}',
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Remove empty subplots
        for i in range(len(numerical_df.columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretation
        st.markdown("""
        **Distribution Plot Interpretation:**
        - Shows the distribution of values for each numerical feature
        - Look for:
          * Normal distribution (bell curve)
          * Symmetric distributions
          * Reasonable range of values
        - Warning signs:
          * Skewed distributions (skewness far from 0)
          * Heavy tails (high kurtosis)
          * Multiple peaks (bimodal distributions)
          * Outliers
        """)
    else:
        st.warning("No numerical columns found for distribution analysis")

def plot_boxplots(df):
    """
    Plot boxplots for numerical features
    """
    st.subheader("Box Plots")
    
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if len(numerical_df.columns) > 0:
        # Create subplots
        n_cols = 2
        n_rows = (len(numerical_df.columns) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_df.columns):
            ax = axes[i]
            sns.boxplot(data=df, y=col, ax=ax)
            ax.set_title(f'Box Plot of {col}')
            
            # Add outlier count
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
            ax.text(0.05, 0.95, 
                   f'Outliers: {len(outliers)}',
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Remove empty subplots
        for i in range(len(numerical_df.columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretation
        st.markdown("""
        **Box Plot Interpretation:**
        - Shows the distribution of values and identifies outliers
        - Look for:
          * Symmetric boxes (median in the middle)
          * Reasonable whisker lengths
          * Few or no outliers
        - Warning signs:
          * Many outliers
          * Asymmetric boxes
          * Extremely long whiskers
        """)
    else:
        st.warning("No numerical columns found for box plot analysis")

def plot_scatter_matrix(df, target_column):
    """
    Plot scatter matrix for numerical features against target
    """
    st.subheader("Scatter Matrix")
    
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if len(numerical_df.columns) > 0:
        # Create scatter matrix
        fig = sns.pairplot(df, vars=numerical_df.columns, hue=target_column if target_column in df.columns else None)
        st.pyplot(fig)
        
        # Interpretation
        st.markdown("""
        **Scatter Matrix Interpretation:**
        - Shows relationships between all pairs of numerical features
        - Look for:
          * Linear relationships
          * Clear patterns or clusters
          * Potential interactions between features
        - Warning signs:
          * Non-linear relationships
          * Outliers affecting relationships
          * No clear patterns
        """)
    else:
        st.warning("No numerical columns found for scatter matrix analysis")

def perform_eda(df, target_column=None):
    """
    Complete EDA pipeline
    """
    # Get data summary
    get_data_summary(df)
    
    # Plot distributions
    plot_distributions(df)
    
    # Plot boxplots
    plot_boxplots(df)
    
    # Plot correlation matrix
    plot_correlation_matrix(df)
    
    # Plot scatter matrix if target column is provided
    if target_column:
        plot_scatter_matrix(df, target_column) 