import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.data_cleaning import preprocess_data
from utils.eda import perform_eda
from models.linear_regression import LinearRegressionModel
from models.polynomial_regression import PolynomialRegressionModel
from models.ridge_regression import RidgeRegressionModel
from models.lasso_regression import LassoRegressionModel
from models.elastic_net import ElasticNetRegressionModel
from config import DATA_SETTINGS
from data.dataset_templates import DATASET_TEMPLATES
from data.parallel_generation import parallel_generate_dataset
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import multiprocessing as mp

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = ["Linear Regression"]
if 'metrics_dict' not in st.session_state:
    st.session_state.metrics_dict = {}

# Set page config
st.set_page_config(
    page_title="Regression Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
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

def plot_residuals(y_true, y_pred):
    """Plot residuals against predicted values"""
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    
    # Add residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax.text(0.05, 0.95, 
            f'Mean Residual: {mean_residual:.2f}\nStd Residual: {std_residual:.2f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown("""
    **Residual Plot Interpretation:**
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
    """)

def plot_actual_vs_predicted(y_true, y_pred):
    """Plot actual vs predicted values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs Predicted Values')
    
    # Add R2 score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² Score: {r2:.4f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown("""
    **Actual vs Predicted Plot Interpretation:**
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
    """)

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    try:
        # Get feature importance or coefficients
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = model.coef_
        else:
            st.warning("Model does not support feature importance visualization")
            return
        
        # Create DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        ax.set_title('Feature Importance')
        
        # Add value labels
        for i, v in enumerate(importance_df['Importance']):
            ax.text(v, i, f' {v:.4f}', va='center')
        
        st.pyplot(fig)
        
        # Add interpretation
        st.markdown("""
        **Feature Importance Plot Interpretation:**
        - Shows the relative importance of each feature in making predictions
        - Look for:
          * Features with significantly higher importance
          * Features with near-zero importance (potential candidates for removal)
          * Direction of influence (positive/negative coefficients)
        - Warning signs:
          * Features with unexpectedly high importance
          * Features with zero importance that should matter
          * Unstable importance rankings across different runs
        """)
        
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")

def plot_model_comparison(metrics_dict):
    """Plot comparison of different models' metrics"""
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    models = list(metrics_dict.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
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
    
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown("""
    **Model Comparison Plot Interpretation:**
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
    """)

def plot_data_distribution(df, target_column):
    """Plot distribution of features and target using Plotly"""
    st.subheader("Feature and Target Distributions")
    
    # Create subplots
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=df.columns)
    
    for i, col in enumerate(df.columns):
        row = i // n_cols + 1
        col_num = i % n_cols + 1
        
        if df[col].dtype in ['int64', 'float64']:
            # For numerical columns
            fig.add_trace(
                go.Histogram(x=df[col], name=col),
                row=row, col=col_num
            )
        else:
            # For categorical columns
            value_counts = df[col].value_counts()
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=col),
                row=row, col=col_num
            )
    
    fig.update_layout(height=300*n_rows, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(df):
    """Plot correlation heatmap using Plotly"""
    st.subheader("Feature Correlation Heatmap")
    
    # Calculate correlation matrix
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_relationships(df, target_column):
    """Plot relationships between features and target"""
    st.subheader("Feature-Target Relationships")
    
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if target_column in numerical_cols:
        numerical_cols = numerical_cols.drop(target_column)
    
    n_cols = 2
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"{col} vs {target_column}" for col in numerical_cols])
    
    for i, col in enumerate(numerical_cols):
        row = i // n_cols + 1
        col_num = i % n_cols + 1
        
        fig.add_trace(
            go.Scatter(x=df[col], y=df[target_column], mode='markers', name=col),
            row=row, col=col_num
        )
    
    fig.update_layout(height=300*n_rows, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Regression Analysis Dashboard")
    
    # Add app description
    st.markdown("""
    This dashboard allows you to perform regression analysis on your dataset. 
    Upload a CSV file, select your target variable, and choose regression models to train.
    """)
    
    # Sidebar for file upload and model selection
    st.sidebar.header("Configuration")
    
    # Add data source options
    data_source = st.sidebar.radio(
        "Data Source",
        ["Upload CSV", "Use Template Dataset"]
    )
    
    if data_source == "Use Template Dataset":
        st.sidebar.subheader("Dataset Template")
        
        # Select dataset template
        selected_template = st.sidebar.selectbox(
            "Choose a dataset template",
            list(DATASET_TEMPLATES.keys())
        )
        
        # Get template description
        template_func = DATASET_TEMPLATES[selected_template]
        template_doc = template_func.__doc__
        st.sidebar.markdown(f"**Description:** {template_doc}")
        
        # Get number of samples
        n_samples = st.sidebar.number_input(
            "Number of Samples",
            min_value=1000,
            max_value=100000,
            value=50000,
            step=1000
        )
        
        # Add parallel processing options
        st.sidebar.subheader("Parallel Processing")
        use_parallel = st.sidebar.checkbox("Use Parallel Processing", value=True)
        if use_parallel:
            n_workers = st.sidebar.slider(
                "Number of Workers",
                min_value=1,
                max_value=mp.cpu_count(),
                value=max(1, mp.cpu_count() - 1),
                step=1
            )
        else:
            n_workers = None
        
        if st.sidebar.button("Generate Dataset"):
            with st.spinner(f"Generating {selected_template} dataset..."):
                try:
                    if use_parallel:
                        st.session_state.df = parallel_generate_dataset(
                            selected_template,
                            n_samples,
                            n_workers
                        )
                    else:
                        st.session_state.df = template_func(n_samples=n_samples)
                    st.session_state.models_trained = False
                    st.session_state.metrics_dict = {}
                    st.sidebar.success("Dataset generated successfully!")
                except Exception as e:
                    st.error(f"Error generating dataset: {str(e)}")
                    return
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=['csv'])
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.models_trained = False
                st.session_state.metrics_dict = {}
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
    
    # If no data is loaded, show info message and return
    if st.session_state.df is None:
        st.info("Please upload a CSV file or select a dataset template to get started.")
        return
    
    # Display dataset info
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of samples", len(st.session_state.df))
    with col2:
        st.metric("Number of features", len(st.session_state.df.columns) - 1)
    with col3:
        st.metric("Missing values", st.session_state.df.isnull().sum().sum())
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head())
    
    # Display available columns
    st.sidebar.write("Available columns:")
    st.sidebar.write(st.session_state.df.columns.tolist())
    
    # Get target column from user
    new_target_column = st.sidebar.selectbox(
        "Select Target Column",
        st.session_state.df.columns,
        index=st.session_state.df.columns.get_loc(st.session_state.target_column) if st.session_state.target_column in st.session_state.df.columns else 0
    )
    
    # Update target column in session state
    if new_target_column != st.session_state.target_column:
        st.session_state.target_column = new_target_column
        st.session_state.models_trained = False
        st.session_state.metrics_dict = {}
    
    # Enhanced EDA Section
    st.header("Exploratory Data Analysis")
    
    # Add tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Relationships"])
    
    with tab1:
        plot_data_distribution(st.session_state.df, st.session_state.target_column)
    
    with tab2:
        plot_correlation_heatmap(st.session_state.df)
    
    with tab3:
        plot_feature_relationships(st.session_state.df, st.session_state.target_column)
    
    # Add download button for the current dataset
    if st.session_state.df is not None:
        csv = st.session_state.df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Current Data",
            data=csv,
            file_name="current_data.csv",
            mime="text/csv"
        )
    
    # Model Selection and Training Section
    st.sidebar.header("Model Selection")
    
    # Model selection checkboxes with improved layout
    st.sidebar.markdown("**Select models to train:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.session_state.selected_models = []
        if st.checkbox("Linear Regression", value="Linear Regression" in st.session_state.selected_models):
            st.session_state.selected_models.append("Linear Regression")
        if st.checkbox("Polynomial Regression", value="Polynomial Regression" in st.session_state.selected_models):
            st.session_state.selected_models.append("Polynomial Regression")
        if st.checkbox("Ridge Regression", value="Ridge Regression" in st.session_state.selected_models):
            st.session_state.selected_models.append("Ridge Regression")
    
    with col2:
        if st.checkbox("Lasso Regression", value="Lasso Regression" in st.session_state.selected_models):
            st.session_state.selected_models.append("Lasso Regression")
        if st.checkbox("Elastic Net Regression", value="Elastic Net Regression" in st.session_state.selected_models):
            st.session_state.selected_models.append("Elastic Net Regression")
    
    # Train models button
    if st.sidebar.button("Train Models"):
        if not st.session_state.selected_models:
            st.warning("Please select at least one model to train")
            return
            
        try:
            # Preprocess data
            X, y = preprocess_data(st.session_state.df, st.session_state.target_column)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=DATA_SETTINGS['test_size'],
                random_state=DATA_SETTINGS['random_state']
            )
            
            # Initialize models
            models = []
            if "Linear Regression" in st.session_state.selected_models:
                models.append(LinearRegressionModel())
            if "Polynomial Regression" in st.session_state.selected_models:
                models.append(PolynomialRegressionModel())
            if "Ridge Regression" in st.session_state.selected_models:
                models.append(RidgeRegressionModel())
            if "Lasso Regression" in st.session_state.selected_models:
                models.append(LassoRegressionModel())
            if "Elastic Net Regression" in st.session_state.selected_models:
                models.append(ElasticNetRegressionModel())
            
            # Train and evaluate models
            st.header("Model Training and Evaluation")
            
            for model in models:
                st.subheader(f"{model.name}")
                
                # Train model
                model.train_with_grid_search(X_train, y_train)
                
                # Make predictions
                y_pred = model.model.predict(X_test)
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred)
                st.session_state.metrics_dict[model.name] = metrics
                
                # Display metrics
                st.write("Model Metrics:")
                st.write(pd.DataFrame([metrics]))
                
                # Plot visualizations
                col1, col2 = st.columns(2)
                with col1:
                    plot_residuals(y_test, y_pred)
                with col2:
                    plot_actual_vs_predicted(y_test, y_pred)
                
                plot_feature_importance(model.model, X.columns)
            
            # Plot model comparison
            st.header("Model Comparison")
            plot_model_comparison(st.session_state.metrics_dict)
            
            st.session_state.models_trained = True
            
            # Prediction Section
            st.header("Make Predictions")
            st.write("Enter feature values for prediction:")
            
            input_features = {}
            for feature in X.columns:
                input_features[feature] = st.number_input(f"Enter {feature}", value=0.0)
            
            if st.button("Predict"):
                for model in models:
                    prediction = model.model.predict(pd.DataFrame([input_features]))[0]
                    st.write(f"{model.name} Prediction: {prediction:.2f}")
                    
        except ValueError as e:
            st.error(f"Data preprocessing error: {str(e)}")
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main() 