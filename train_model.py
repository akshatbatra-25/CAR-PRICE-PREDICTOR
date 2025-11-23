import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle
import os

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

# Ensure required columns exist (handle case variations)
required_cols = ['name', 'company', 'year', 'kms_driven', 'fuel_type']
price_col = 'Price' if 'Price' in df.columns else 'price' if 'price' in df.columns else None

if price_col is None:
    raise ValueError("Price column not found in dataset. Available columns: " + str(df.columns.tolist()))

# Check if all required columns exist
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")

# Define features and target
X = df[required_cols]
y = df[price_col]

# Create column transformer for categorical features
ohe = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'), ['name', 'company', 'fuel_type'])
    ],
    remainder='passthrough'
)

# Create the pipeline
model = Pipeline(steps=[
    ('preprocessor', ohe),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as 'model.pkl'")
