import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_friedman1, make_friedman2, make_friedman3

def generate_housing_data(n_samples=50000):
    """Generate synthetic housing price data"""
    # Base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=8,
        noise=20.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'floors', 'waterfront', 'view', 'condition'
    ])
    
    # Scale features to realistic ranges
    df['bedrooms'] = np.clip(df['bedrooms'] * 2 + 3, 1, 8)
    df['bathrooms'] = np.clip(df['bathrooms'] * 1.5 + 2, 1, 5)
    df['sqft_living'] = np.clip(df['sqft_living'] * 1000 + 1500, 500, 5000)
    df['sqft_lot'] = np.clip(df['sqft_lot'] * 5000 + 5000, 1000, 20000)
    df['floors'] = np.clip(df['floors'] * 1.5 + 1.5, 1, 3)
    df['waterfront'] = (df['waterfront'] > 0).astype(int)
    df['view'] = np.clip(df['view'] * 2 + 2, 0, 4)
    df['condition'] = np.clip(df['condition'] * 2 + 3, 1, 5)
    
    # Add target (price in thousands)
    df['price'] = np.clip(y * 200 + 300, 100, 2000)
    
    return df

def generate_glucose_data(n_samples=50000):
    """Generate synthetic glucose prediction data"""
    # Base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=7,
        noise=15.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'age', 'bmi', 'blood_pressure', 'insulin',
        'skin_thickness', 'pregnancies', 'diabetes_pedigree'
    ])
    
    # Scale features to realistic ranges
    df['age'] = np.clip(df['age'] * 10 + 30, 20, 80)
    df['bmi'] = np.clip(df['bmi'] * 10 + 25, 18, 50)
    df['blood_pressure'] = np.clip(df['blood_pressure'] * 20 + 80, 60, 140)
    df['insulin'] = np.clip(df['insulin'] * 50 + 100, 0, 300)
    df['skin_thickness'] = np.clip(df['skin_thickness'] * 10 + 20, 0, 50)
    df['pregnancies'] = np.clip(df['pregnancies'] * 2 + 2, 0, 10)
    df['diabetes_pedigree'] = np.clip(df['diabetes_pedigree'] * 0.5 + 0.5, 0, 2)
    
    # Add target (glucose level)
    df['glucose'] = np.clip(y * 50 + 100, 50, 300)
    
    return df

def generate_stock_data(n_samples=50000):
    """Generate synthetic stock price data"""
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=6,
        noise=10.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'volume', 'pe_ratio', 'market_cap', 'debt_ratio',
        'profit_margin', 'growth_rate'
    ])
    
    # Scale features to realistic ranges
    df['volume'] = np.clip(df['volume'] * 1000000 + 1000000, 100000, 10000000)
    df['pe_ratio'] = np.clip(df['pe_ratio'] * 10 + 15, 5, 50)
    df['market_cap'] = np.clip(df['market_cap'] * 10 + 20, 1, 100)
    df['debt_ratio'] = np.clip(df['debt_ratio'] * 0.3 + 0.5, 0, 1)
    df['profit_margin'] = np.clip(df['profit_margin'] * 0.1 + 0.1, -0.2, 0.4)
    df['growth_rate'] = np.clip(df['growth_rate'] * 0.2 + 0.1, -0.3, 0.5)
    
    # Add target (stock price)
    df['price'] = np.clip(y * 50 + 100, 10, 500)
    
    return df

def generate_car_data(n_samples=50000):
    """Generate synthetic car price data"""
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=8,
        noise=15.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'year', 'mileage', 'engine_size', 'horsepower',
        'fuel_efficiency', 'seats', 'doors', 'safety_rating'
    ])
    
    # Scale features to realistic ranges
    df['year'] = np.clip(df['year'] * 5 + 2015, 2000, 2023)
    df['mileage'] = np.clip(df['mileage'] * 50000 + 50000, 0, 200000)
    df['engine_size'] = np.clip(df['engine_size'] * 2 + 2, 1, 5)
    df['horsepower'] = np.clip(df['horsepower'] * 100 + 150, 50, 500)
    df['fuel_efficiency'] = np.clip(df['fuel_efficiency'] * 10 + 25, 15, 50)
    df['seats'] = np.clip(df['seats'] * 2 + 4, 2, 8)
    df['doors'] = np.clip(df['doors'] * 1.5 + 3, 2, 5)
    df['safety_rating'] = np.clip(df['safety_rating'] * 2 + 3, 1, 5)
    
    # Add target (car price in thousands)
    df['price'] = np.clip(y * 20 + 30, 10, 100)
    
    return df

def generate_salary_data(n_samples=50000):
    """Generate synthetic salary prediction data"""
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=7,
        noise=20.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'years_experience', 'education_level', 'job_title',
        'company_size', 'industry', 'location', 'skills_score'
    ])
    
    # Scale features to realistic ranges
    df['years_experience'] = np.clip(df['years_experience'] * 10 + 5, 0, 30)
    df['education_level'] = np.clip(df['education_level'] * 2 + 3, 1, 5)
    df['job_title'] = np.clip(df['job_title'] * 3 + 3, 1, 6)
    df['company_size'] = np.clip(df['company_size'] * 3 + 3, 1, 6)
    df['industry'] = np.clip(df['industry'] * 4 + 4, 1, 8)
    df['location'] = np.clip(df['location'] * 3 + 3, 1, 6)
    df['skills_score'] = np.clip(df['skills_score'] * 2 + 3, 1, 5)
    
    # Add target (annual salary in thousands)
    df['salary'] = np.clip(y * 50 + 70, 30, 300)
    
    return df

def generate_energy_data(n_samples=50000):
    """Generate synthetic energy consumption data"""
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=6,
        noise=15.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'temperature', 'humidity', 'occupancy',
        'appliance_count', 'building_size', 'insulation_rating'
    ])
    
    # Scale features to realistic ranges
    df['temperature'] = np.clip(df['temperature'] * 20 + 20, -10, 40)
    df['humidity'] = np.clip(df['humidity'] * 30 + 50, 20, 90)
    df['occupancy'] = np.clip(df['occupancy'] * 4 + 2, 1, 10)
    df['appliance_count'] = np.clip(df['appliance_count'] * 5 + 5, 1, 15)
    df['building_size'] = np.clip(df['building_size'] * 1000 + 1500, 500, 3000)
    df['insulation_rating'] = np.clip(df['insulation_rating'] * 2 + 3, 1, 5)
    
    # Add target (energy consumption in kWh)
    df['energy_consumption'] = np.clip(y * 500 + 1000, 200, 3000)
    
    return df

def generate_insurance_data(n_samples=50000):
    """Generate synthetic insurance claim data"""
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=7,
        noise=25.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'age', 'bmi', 'children', 'smoker',
        'region', 'coverage_level', 'deductible'
    ])
    
    # Scale features to realistic ranges
    df['age'] = np.clip(df['age'] * 20 + 30, 18, 80)
    df['bmi'] = np.clip(df['bmi'] * 10 + 25, 18, 50)
    df['children'] = np.clip(df['children'] * 2 + 1, 0, 5)
    df['smoker'] = (df['smoker'] > 0).astype(int)
    df['region'] = np.clip(df['region'] * 3 + 3, 1, 4)
    df['coverage_level'] = np.clip(df['coverage_level'] * 2 + 3, 1, 5)
    df['deductible'] = np.clip(df['deductible'] * 1000 + 1000, 500, 5000)
    
    # Add target (insurance claim amount)
    df['claim_amount'] = np.clip(y * 5000 + 5000, 1000, 30000)
    
    return df

def generate_retail_data(n_samples=50000):
    """Generate synthetic retail sales data"""
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=8,
        noise=30.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'price', 'discount', 'advertising',
        'competitor_price', 'season', 'holiday',
        'store_size', 'location_score'
    ])
    
    # Scale features to realistic ranges
    df['price'] = np.clip(df['price'] * 50 + 100, 20, 500)
    df['discount'] = np.clip(df['discount'] * 0.3 + 0.1, 0, 0.5)
    df['advertising'] = np.clip(df['advertising'] * 10000 + 20000, 5000, 50000)
    df['competitor_price'] = np.clip(df['competitor_price'] * 50 + 100, 20, 500)
    df['season'] = np.clip(df['season'] * 3 + 3, 1, 4)
    df['holiday'] = (df['holiday'] > 0).astype(int)
    df['store_size'] = np.clip(df['store_size'] * 2000 + 3000, 1000, 10000)
    df['location_score'] = np.clip(df['location_score'] * 2 + 3, 1, 5)
    
    # Add target (sales volume)
    df['sales'] = np.clip(y * 1000 + 2000, 500, 5000)
    
    return df

def generate_weather_data(n_samples=50000):
    """Generate synthetic weather prediction data"""
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=6,
        noise=10.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'temperature', 'humidity', 'pressure',
        'wind_speed', 'cloud_cover', 'precipitation'
    ])
    
    # Scale features to realistic ranges
    df['temperature'] = np.clip(df['temperature'] * 20 + 20, -10, 40)
    df['humidity'] = np.clip(df['humidity'] * 30 + 50, 20, 90)
    df['pressure'] = np.clip(df['pressure'] * 20 + 1010, 980, 1040)
    df['wind_speed'] = np.clip(df['wind_speed'] * 20 + 10, 0, 50)
    df['cloud_cover'] = np.clip(df['cloud_cover'] * 50 + 50, 0, 100)
    df['precipitation'] = np.clip(df['precipitation'] * 20 + 10, 0, 100)
    
    # Add target (temperature for next day)
    df['next_day_temp'] = np.clip(y * 10 + 20, -5, 35)
    
    return df

def generate_education_data(n_samples=50000):
    """Generate synthetic student performance data"""
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=7,
        noise=15.0,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[
        'study_time', 'attendance', 'parent_education',
        'family_income', 'extracurricular', 'teacher_rating',
        'class_size'
    ])
    
    # Scale features to realistic ranges
    df['study_time'] = np.clip(df['study_time'] * 2 + 2, 0, 6)
    df['attendance'] = np.clip(df['attendance'] * 20 + 80, 50, 100)
    df['parent_education'] = np.clip(df['parent_education'] * 2 + 3, 1, 5)
    df['family_income'] = np.clip(df['family_income'] * 50000 + 75000, 25000, 200000)
    df['extracurricular'] = np.clip(df['extracurricular'] * 2 + 2, 0, 5)
    df['teacher_rating'] = np.clip(df['teacher_rating'] * 2 + 3, 1, 5)
    df['class_size'] = np.clip(df['class_size'] * 10 + 25, 15, 40)
    
    # Add target (test score)
    df['test_score'] = np.clip(y * 20 + 70, 40, 100)
    
    return df

# Dictionary of available datasets
DATASET_TEMPLATES = {
    "Housing Prices": generate_housing_data,
    "Glucose Prediction": generate_glucose_data,
    "Stock Prices": generate_stock_data,
    "Car Prices": generate_car_data,
    "Salary Prediction": generate_salary_data,
    "Energy Consumption": generate_energy_data,
    "Insurance Claims": generate_insurance_data,
    "Retail Sales": generate_retail_data,
    "Weather Prediction": generate_weather_data,
    "Education Performance": generate_education_data
} 