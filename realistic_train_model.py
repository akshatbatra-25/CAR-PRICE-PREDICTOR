import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# Load dataset - try newest dataset first, then fallback to others
CSV_PATH_FINAL = "cars_ds_final.csv"
CSV_PATH_NEW = "Cars Datasets 2025.csv"
CSV_PATH_OLD = "Cleaned_Car_data.csv"

df = None

# Try cars_ds_final.csv first (newest dataset)
if os.path.exists(CSV_PATH_FINAL):
    try:
        # Load final dataset
        df = pd.read_csv(CSV_PATH_FINAL)
        # Map columns to expected format
        column_mapping = {
            'Make': 'company',
            'Model': 'name',
            'Ex-Showroom_Price': 'Price',
            'Fuel_Type': 'fuel_type'
        }
        df = df.rename(columns=column_mapping)
        
        # Clean price column (remove currency symbols, commas, "Rs." prefix, and convert to numeric)
        if 'Price' in df.columns:
            df['Price'] = df['Price'].astype(str).str.replace(r'Rs\.?\s*', '', regex=True)  # Remove "Rs." or "Rs" prefix
            df['Price'] = df['Price'].str.replace(r'[₹,$,\s]', '', regex=True)  # Remove currency symbols and commas
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            df = df.dropna(subset=['Price'])  # Remove rows with invalid prices
        
        # Add missing columns with default values if not present
        if 'year' not in df.columns:
            # Default to 2024 for new cars dataset
            df['year'] = 2024
        if 'kms_driven' not in df.columns:
            # Check if Odometer column exists and can be used, otherwise default to 0
            if 'Odometer' in df.columns:
                try:
                    df['kms_driven'] = pd.to_numeric(df['Odometer'], errors='coerce').fillna(0)
                except:
                    df['kms_driven'] = 0
            else:
                df['kms_driven'] = 0
        
        print(f"Loaded final dataset: {CSV_PATH_FINAL} with {len(df)} records")
    except Exception as e:
        print(f"Error loading final dataset: {e}")
        import traceback
        traceback.print_exc()
        df = None

# Fallback to Cars Datasets 2025.csv
if df is None and os.path.exists(CSV_PATH_NEW):
    try:
        # Load new dataset with latin-1 encoding to handle special characters
        df = pd.read_csv(CSV_PATH_NEW, encoding='latin-1')
        # Map new dataset columns to expected format
        column_mapping = {
            'Company Names': 'company',
            'Cars Names': 'name',
            'Cars Prices': 'Price',
            'Fuel Types': 'fuel_type'
        }
        df = df.rename(columns=column_mapping)
        
        # Clean price column (remove currency symbols and convert to numeric)
        if 'Price' in df.columns:
            df['Price'] = df['Price'].astype(str).str.replace(r'[₹,$,\s]', '', regex=True)
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            df = df.dropna(subset=['Price'])  # Remove rows with invalid prices
        
        # Add missing columns with default values if not present
        if 'year' not in df.columns:
            # Default to 2024 for new cars dataset
            df['year'] = 2024
        if 'kms_driven' not in df.columns:
            # Default to 0 for new cars dataset (assuming they're new cars)
            df['kms_driven'] = 0
        
        print(f"Loaded new dataset: {CSV_PATH_NEW} with {len(df)} records")
    except Exception as e:
        print(f"Error loading new dataset: {e}")
        df = None

# Fallback to old dataset
if df is None and os.path.exists(CSV_PATH_OLD):
    try:
        df = pd.read_csv(CSV_PATH_OLD)
        print(f"Loaded old dataset: {CSV_PATH_OLD} with {len(df)} records")
    except Exception as e:
        print(f"Error loading old dataset: {e}")
        raise

if df is None:
    raise FileNotFoundError("No dataset found. Please ensure either 'cars_ds_final.csv', 'Cars Datasets 2025.csv', or 'Cleaned_Car_data.csv' exists.")

print(f"Original dataset: {len(df)} records")

# Ensure required columns exist (handle case variations)
required_cols = ['name', 'company', 'year', 'kms_driven', 'fuel_type']
price_col = 'Price' if 'Price' in df.columns else 'price' if 'price' in df.columns else None

if price_col is None:
    raise ValueError("Price column not found in dataset. Available columns: " + str(df.columns.tolist()))

# Check if all required columns exist
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")

# Define features and target (using original features for compatibility)
X = df[required_cols]
y = df[price_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create column transformer for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['name', 'company', 'fuel_type'])
    ],
    remainder='passthrough'
)

# Create the pipeline with slightly improved parameters
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

# Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*60)
print("MODEL EVALUATION METRICS")
print("="*60)
print(f"\nTrain Set:")
print(f"  MAE: Rs.{train_mae:,.0f}")
print(f"  RMSE: Rs.{train_rmse:,.0f}")
print(f"  R² Score: {train_r2:.4f}")

print(f"\nTest Set:")
print(f"  MAE: Rs.{test_mae:,.0f}")
print(f"  RMSE: Rs.{test_rmse:,.0f}")
print(f"  R² Score: {test_r2:.4f}")

# Calculate price range statistics for validation
price_stats = {
    'min_price': float(df[price_col].min()),
    'max_price': float(df[price_col].max()),
    'mean_price': float(df[price_col].mean()),
    'median_price': float(df[price_col].median()),
    'std_price': float(df[price_col].std()),
    'q25_price': float(df[price_col].quantile(0.25)),
    'q75_price': float(df[price_col].quantile(0.75))
}

# Calculate confidence intervals based on RMSE
price_stats['error_margin'] = test_rmse

print("\n" + "="*60)
print("DATASET PRICE STATISTICS")
print("="*60)
for key, value in price_stats.items():
    print(f"  {key}: Rs.{value:,.0f}")

# Save trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
print("\nModel saved as 'model.pkl'")

# Save price statistics for validation
with open('price_stats.pkl', 'wb') as file:
    pickle.dump(price_stats, file)
    
print("Price statistics saved as 'price_stats.pkl'")

# Test with a few predictions
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)
sample_indices = [0, 10, 50, 100]
for idx in sample_indices:
    if idx < len(X_test):
        sample = X_test.iloc[[idx]]
        actual = y_test.iloc[idx]
        predicted = model.predict(sample)[0]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100 if actual > 0 else 0
        
        print(f"\nTest {idx + 1}:")
        print(f"  Actual Price: Rs.{actual:,.0f}")
        print(f"  Predicted Price: Rs.{predicted:,.0f}")
        print(f"  Error: Rs.{error:,.0f} ({error_pct:.1f}%)")

print("\n" + "="*60)
print("Training completed successfully!")
print("="*60)


