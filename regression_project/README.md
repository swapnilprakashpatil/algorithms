# Regression Analysis Dashboard

An interactive web application for performing regression analysis on datasets. Built with Streamlit and scikit-learn.

## Features

- Interactive data exploration and visualization
- Multiple regression models:
  - Linear Regression
  - Polynomial Regression
  - Ridge Regression
  - Lasso Regression
  - Elastic Net Regression
- Model evaluation with detailed metrics and visualizations
- Real-time predictions
- Sample dataset for quick testing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/regression-analysis-dashboard.git
cd regression-analysis-dashboard
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run regression_project/app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Either upload your own dataset or use the sample data provided

4. Select your target variable and choose regression models to train

5. View the results and make predictions

## Deployment

### Deploying to Streamlit Cloud

1. Create a GitHub repository for your project

2. Push your code to GitHub

3. Go to [Streamlit Cloud](https://streamlit.io/cloud)

4. Click "New app" and connect your GitHub repository

5. Select the main branch and set the main file path to `regression_project/app.py`

6. Click "Deploy"

### Deploying to Heroku

1. Create a `Procfile` in your project root:
```
web: streamlit run regression_project/app.py --server.port $PORT
```

2. Create a `setup.sh` file:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Push your code to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

## Project Structure

```
regression_project/
├── app.py                 # Main Streamlit application
├── models/               # Regression model implementations
│   ├── base_model.py
│   ├── linear_regression.py
│   ├── polynomial_regression.py
│   ├── ridge_regression.py
│   ├── lasso_regression.py
│   └── elastic_net.py
├── utils/                # Utility functions
│   ├── data_cleaning.py
│   ├── eda.py
│   └── model_evaluation.py
├── data/                 # Sample data
│   └── sample_data.csv
├── config.py            # Configuration settings
└── requirements.txt     # Project dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 