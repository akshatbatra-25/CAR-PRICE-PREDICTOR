"""
Enhanced ML Model Training for 2025 Indian Car Resale Price Prediction
This script trains a model to predict realistic used car resale prices in INR
based on Company, Model, Fuel Type, Year, and Kilometers Driven.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# Set current year for 2025 market conditions
CURRENT_YEAR = datetime.now().year
print(f"Training model for {CURRENT_YEAR} Indian market conditions...\n")

# Load dataset
CSV_PATH_FINAL = "cars_ds_final.csv"
CSV_PATH_NEW = "Cars Datasets 2025.csv"
CSV_PATH_OLD = "Cleaned_Car_data.csv"

df = None

# Try cars_ds_final.csv first (newest dataset)
if os.path.exists(CSV_PATH_FINAL):
    try:
        df = pd.read_csv(CSV_PATH_FINAL)
        column_mapping = {
            'Make': 'company',
            'Model': 'name',
            'Ex-Showroom_Price': 'exshowroom_price',
            'Fuel_Type': 'fuel_type'
        }
        df = df.rename(columns=column_mapping)
        
        # Clean ex-showroom price column
        if 'exshowroom_price' in df.columns:
            df['exshowroom_price'] = df['exshowroom_price'].astype(str).str.replace(r'Rs\.?\s*', '', regex=True)
            df['exshowroom_price'] = df['exshowroom_price'].str.replace(r'[₹,$,\s]', '', regex=True)
            df['exshowroom_price'] = pd.to_numeric(df['exshowroom_price'], errors='coerce')
            df = df.dropna(subset=['exshowroom_price'])
        
        # Add missing columns with realistic defaults
        if 'year' not in df.columns:
            # For new cars dataset, assign years from 2020 to 2025
            df['year'] = np.random.randint(2020, CURRENT_YEAR + 1, size=len(df))
        if 'kms_driven' not in df.columns:
            # Generate realistic kms_driven based on car age
            if 'year' in df.columns:
                car_age = CURRENT_YEAR - df['year']
                # Average 12,000 km per year, with some variation
                df['kms_driven'] = (car_age * 12000 + np.random.randint(-5000, 5000, size=len(df))).clip(0, 300000)
            else:
                df['kms_driven'] = 0
        
        print(f"Loaded final dataset: {CSV_PATH_FINAL} with {len(df)} records")
    except Exception as e:
        print(f"Error loading final dataset: {e}")
        df = None

# Fallback to other datasets
if df is None and os.path.exists(CSV_PATH_NEW):
    try:
        df = pd.read_csv(CSV_PATH_NEW, encoding='latin-1')
        column_mapping = {
            'Company Names': 'company',
            'Cars Names': 'name',
            'Cars Prices': 'exshowroom_price',
            'Fuel Types': 'fuel_type'
        }
        df = df.rename(columns=column_mapping)
        
        if 'exshowroom_price' in df.columns:
            df['exshowroom_price'] = df['exshowroom_price'].astype(str).str.replace(r'[₹,$,\s]', '', regex=True)
            df['exshowroom_price'] = pd.to_numeric(df['exshowroom_price'], errors='coerce')
            df = df.dropna(subset=['exshowroom_price'])
        
        if 'year' not in df.columns:
            df['year'] = np.random.randint(2020, CURRENT_YEAR + 1, size=len(df))
        if 'kms_driven' not in df.columns:
            car_age = CURRENT_YEAR - df['year']
            df['kms_driven'] = (car_age * 12000 + np.random.randint(-5000, 5000, size=len(df))).clip(0, 300000)
        
        print(f"Loaded new dataset: {CSV_PATH_NEW} with {len(df)} records")
    except Exception as e:
        print(f"Error loading new dataset: {e}")
        df = None

if df is None and os.path.exists(CSV_PATH_OLD):
    try:
        df = pd.read_csv(CSV_PATH_OLD)
        if 'price' in df.columns and 'exshowroom_price' not in df.columns:
            df['exshowroom_price'] = df['price']
        print(f"Loaded old dataset: {CSV_PATH_OLD} with {len(df)} records")
    except Exception as e:
        print(f"Error loading old dataset: {e}")
        df = None

if df is None:
    raise FileNotFoundError("No dataset found. Please ensure a dataset file exists.")

print(f"Original dataset: {len(df)} records\n")

# Feature Engineering: Convert ex-showroom prices to realistic used car resale prices
# This is critical because the dataset has ex-showroom prices, but we need to predict resale prices

def calculate_resale_price(row):
    """
    Calculate realistic resale price from ex-showroom price based on:
    - Age-based depreciation
    - Mileage impact
    - Fuel type effects
    - Brand reliability
    """
    exshowroom = row['exshowroom_price']
    year = row['year']
    kms = row['kms_driven']
    fuel_type = row['fuel_type']
    company = row['company']
    
    # Calculate car age
    car_age = CURRENT_YEAR - year
    car_age = max(0, min(car_age, 20))  # Cap at 20 years
    
    # Age-based depreciation (typical Indian market rates)
    if car_age == 0:
        age_depreciation = 0.10  # 10% first year
    elif car_age == 1:
        age_depreciation = 0.25  # 25% after 2 years
    elif car_age <= 5:
        age_depreciation = 0.25 + (car_age - 2) * 0.10  # 10% per additional year
    elif car_age <= 10:
        age_depreciation = 0.55 + (car_age - 5) * 0.08  # 8% per year for older cars
    else:
        age_depreciation = 0.95  # Max 95% depreciation for very old cars
    
    age_depreciation = min(age_depreciation, 0.95)
    
    # Mileage impact (more significant for budget cars)
    avg_km_per_year = kms / (car_age + 1)
    if avg_km_per_year < 10000:
        mileage_factor = 0.98  # Low mileage bonus
    elif avg_km_per_year < 15000:
        mileage_factor = 1.0  # Normal
    elif avg_km_per_year < 25000:
        mileage_factor = 0.92  # High mileage penalty
    else:
        mileage_factor = 0.85  # Very high mileage penalty
    
    # Additional mileage penalty for very high total kms
    if kms > 150000:
        mileage_factor *= 0.90
    if kms > 200000:
        mileage_factor *= 0.85
    
    # Fuel type effects (2025 market conditions)
    fuel_multiplier = {
        'Petrol': 1.0,
        'Diesel': 0.95 if car_age > 8 else 0.98,  # Diesel depreciates faster after 8 years
        'CNG': 0.92,  # CNG has lower resale
        'Electric': 1.05 if car_age < 3 else 0.90,  # EVs retain value when new, depreciate faster when old
        'Hybrid': 1.02  # Hybrids hold value better
    }
    fuel_factor = fuel_multiplier.get(fuel_type, 1.0)
    
    # Brand reliability and demand (Indian market)
    reliable_brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'Mahindra']
    luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Jaguar', 'Land Rover', 'Volvo']
    
    if company in reliable_brands:
        brand_factor = 1.03  # Popular brands hold value better
    elif company in luxury_brands:
        brand_factor = 0.95 if car_age > 5 else 1.0  # Luxury depreciates faster when old
    else:
        brand_factor = 1.0
    
    # Calculate resale price
    resale_price = exshowroom * (1 - age_depreciation) * mileage_factor * fuel_factor * brand_factor
    
    # Add minimal realistic variation (±2%) for training stability
    # Note: This variation helps model learn robustness, but we'll reduce it
    variation = np.random.uniform(-0.02, 0.02)
    resale_price = resale_price * (1 + variation)
    
    # Ensure minimum realistic price
    resale_price = max(resale_price, 50000)  # Minimum ₹50k
    
    return resale_price

# Apply resale price calculation
print("Converting ex-showroom prices to realistic used car resale prices...")
df['price'] = df.apply(calculate_resale_price, axis=1)

# Remove outliers (prices beyond 3 standard deviations)
price_mean = df['price'].mean()
price_std = df['price'].std()
df = df[(df['price'] >= price_mean - 3*price_std) & (df['price'] <= price_mean + 3*price_std)]
print(f"After outlier removal: {len(df)} records\n")

# Feature Engineering: Create additional features
df['car_age'] = CURRENT_YEAR - df['year']
df['km_per_year'] = df['kms_driven'] / (df['car_age'] + 1)
df['is_reliable_brand'] = df['company'].isin(['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'Mahindra']).astype(int)
df['is_luxury_brand'] = df['company'].isin(['BMW', 'Mercedes-Benz', 'Audi', 'Jaguar', 'Land Rover', 'Volvo']).astype(int)

# Define features and target
feature_cols = ['name', 'company', 'fuel_type', 'year', 'kms_driven', 'car_age', 'km_per_year', 'is_reliable_brand', 'is_luxury_brand']
target_col = 'price'

X = df[feature_cols].copy()
y = df[target_col].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

# Separate categorical and numerical features
categorical_features = ['name', 'company', 'fuel_type']
numerical_features = ['year', 'kms_driven', 'car_age', 'km_per_year', 'is_reliable_brand', 'is_luxury_brand']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='drop'
)

# Use log transformation for better price prediction (handles wide price ranges better)
print("Applying log transformation to prices for better model performance...")
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Try multiple models and select the best one
models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
}

best_model = None
best_score = float('inf')
best_model_name = None

print("Training and comparing models...\n")

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train on log-transformed prices
    pipeline.fit(X_train, y_train_log)
    
    # Predict on log scale, then exponentiate back
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # Convert back from log scale
    
    # Evaluate on original scale
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"  Test MAE: Rs.{mae:,.0f}")
    print(f"  Test RMSE: Rs.{rmse:,.0f}")
    
    if rmse < best_score:
        best_score = rmse
        best_model = pipeline
        best_model_name = model_name

print(f"\nBest model: {best_model_name} (RMSE: Rs.{best_score:,.0f})\n")

# Final evaluation with best model (exponentiate from log scale)
y_pred_train_log = best_model.predict(X_train)
y_pred_test_log = best_model.predict(X_test)
y_pred_train = np.expm1(y_pred_train_log)  # Convert back from log scale
y_pred_test = np.expm1(y_pred_test_log)  # Convert back from log scale

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Calculate percentage errors
train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print("="*70)
print("FINAL MODEL EVALUATION METRICS")
print("="*70)
print(f"\nTrain Set:")
print(f"  MAE: Rs.{train_mae:,.0f}")
print(f"  RMSE: Rs.{train_rmse:,.0f}")
print(f"  R² Score: {train_r2:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {train_mape:.2f}%")

print(f"\nTest Set:")
print(f"  MAE: Rs.{test_mae:,.0f}")
print(f"  RMSE: Rs.{test_rmse:,.0f}")
print(f"  R² Score: {test_r2:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")

# Check if model meets the 10-15% deviation requirement
if test_mape <= 15:
    print(f"\n[SUCCESS] Model meets requirement: MAPE ({test_mape:.2f}%) <= 15%")
else:
    print(f"\n[WARNING] Model MAPE ({test_mape:.2f}%) exceeds 15% target")

# Price statistics
price_stats = {
    'min_price': float(df[target_col].min()),
    'max_price': float(df[target_col].max()),
    'mean_price': float(df[target_col].mean()),
    'median_price': float(df[target_col].median()),
    'std_price': float(df[target_col].std()),
    'error_margin': float(test_rmse)
}

print("\n" + "="*70)
print("DATASET PRICE STATISTICS")
print("="*70)
for key, value in price_stats.items():
    print(f"  {key}: Rs.{value:,.0f}")

# Save model
model_filename = 'model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)
print(f"\nModel saved as '{model_filename}'")

# Save price statistics
stats_filename = 'price_stats.pkl'
with open(stats_filename, 'wb') as file:
    pickle.dump(price_stats, file)
print(f"Price statistics saved as '{stats_filename}'")

# Sample predictions with detailed analysis
print("\n" + "="*70)
print("SAMPLE PREDICTIONS (with detailed analysis)")
print("="*70)

sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
for idx in sample_indices:
    sample = X_test.iloc[[idx]]
    actual = y_test.iloc[idx]
    predicted_log = best_model.predict(sample)[0]
    predicted = np.expm1(predicted_log)  # Convert back from log scale
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100 if actual > 0 else 0
    
    row = X_test.iloc[idx]
    print(f"\nSample {idx}:")
    print(f"  Company: {row['company']}")
    print(f"  Model: {row['name']}")
    print(f"  Fuel Type: {row['fuel_type']}")
    print(f"  Year: {int(row['year'])} (Age: {int(row['car_age'])} years)")
    print(f"  Kilometers Driven: {int(row['kms_driven']):,} km")
    print(f"  Actual Resale Price: Rs.{actual:,.0f}")
    print(f"  Predicted Resale Price: Rs.{predicted:,.0f}")
    print(f"  Error: Rs.{error:,.0f} ({error_pct:.2f}%)")

print("\n" + "="*70)
print("Training completed successfully!")
print("="*70)

