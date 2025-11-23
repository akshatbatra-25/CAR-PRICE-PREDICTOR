import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
PRICE_STATS_PATH = os.path.join(os.path.dirname(__file__), "price_stats.pkl")
model = None
price_stats = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print("Model failed to load:", e)
else:
    print("Model file not found. Please train the model first.")

# Load price statistics for validation
if os.path.exists(PRICE_STATS_PATH):
    try:
        with open(PRICE_STATS_PATH, "rb") as f:
            price_stats = pickle.load(f)
        print("Price statistics loaded successfully.")
    except Exception as e:
        print("Price statistics failed to load:", e)
else:
    print("Price statistics not found. Using default ranges.")

# Load the dataset
cars_df = None
# Try newest dataset first, then fallback to other datasets
CSV_PATH_FINAL = os.path.join(os.path.dirname(__file__), "cars_ds_final.csv")
CSV_PATH_NEW = os.path.join(os.path.dirname(__file__), "Cars Datasets 2025.csv")
CSV_PATH_OLD = os.path.join(os.path.dirname(__file__), "Cleaned_Car_data.csv")

# Try cars_ds_final.csv first (newest dataset)
if os.path.exists(CSV_PATH_FINAL):
    try:
        # Load final dataset
        cars_df = pd.read_csv(CSV_PATH_FINAL)
        # Map columns to expected format
        column_mapping = {
            'Make': 'company',
            'Model': 'name',
            'Ex-Showroom_Price': 'price',
            'Fuel_Type': 'fuel_type'
        }
        cars_df = cars_df.rename(columns=column_mapping)
        
        # Clean column names (lowercase and strip)
        cars_df.columns = [c.strip().lower() for c in cars_df.columns]
        
        # Clean price column (remove currency symbols, commas, "Rs." prefix, and convert to numeric)
        if 'price' in cars_df.columns:
            cars_df['price'] = cars_df['price'].astype(str).str.replace(r'Rs\.?\s*', '', regex=True)  # Remove "Rs." or "Rs" prefix
            cars_df['price'] = cars_df['price'].str.replace(r'[₹,$,\s]', '', regex=True)  # Remove currency symbols and commas
            cars_df['price'] = pd.to_numeric(cars_df['price'], errors='coerce')
            # Remove rows with invalid prices
            cars_df = cars_df.dropna(subset=['price'])
        
        # Add missing columns with default values if not present
        if 'year' not in cars_df.columns:
            # Default to 2024 for new cars dataset
            cars_df['year'] = 2024
        if 'kms_driven' not in cars_df.columns:
            # Check if Odometer column exists and can be used, otherwise default to 0
            if 'odometer' in cars_df.columns:
                # Try to extract numeric value from odometer if it's a string
                try:
                    cars_df['kms_driven'] = pd.to_numeric(cars_df['odometer'], errors='coerce').fillna(0)
                except:
                    cars_df['kms_driven'] = 0
            else:
                cars_df['kms_driven'] = 0
        
        # Ensure required columns exist
        required_cols = ['company', 'name', 'fuel_type']
        if not all(col in cars_df.columns for col in required_cols):
            print(f"Warning: Missing required columns. Available: {cars_df.columns.tolist()}")
        
        # Remove rows with missing critical data
        cars_df = cars_df.dropna(subset=required_cols)
        
        print(f"Final dataset (cars_ds_final.csv) loaded: {len(cars_df)} records")
    except Exception as e:
        print(f"Final dataset could not be read: {e}")
        import traceback
        traceback.print_exc()
        cars_df = None

# Fallback to Cars Datasets 2025.csv
if cars_df is None and os.path.exists(CSV_PATH_NEW):
    try:
        # Load new dataset with latin-1 encoding to handle special characters
        cars_df = pd.read_csv(CSV_PATH_NEW, encoding='latin-1')
        # Map new dataset columns to expected format
        column_mapping = {
            'Company Names': 'company',
            'Cars Names': 'name',
            'Cars Prices': 'price',
            'Fuel Types': 'fuel_type'
        }
        cars_df = cars_df.rename(columns=column_mapping)
        
        # Clean column names (lowercase and strip) - do this first
        cars_df.columns = [c.strip().lower() for c in cars_df.columns]
        
        # Clean price column (remove currency symbols, commas, "Rs." prefix, and convert to numeric)
        if 'price' in cars_df.columns:
            cars_df['price'] = cars_df['price'].astype(str).str.replace(r'Rs\.?\s*', '', regex=True)  # Remove "Rs." or "Rs" prefix
            cars_df['price'] = cars_df['price'].str.replace(r'[₹,$,\s]', '', regex=True)  # Remove currency symbols and commas
            cars_df['price'] = pd.to_numeric(cars_df['price'], errors='coerce')
            # Remove rows with invalid prices
            cars_df = cars_df.dropna(subset=['price'])
        
        # Add missing columns with default values if not present
        if 'year' not in cars_df.columns:
            # Default to 2024 for new cars dataset
            cars_df['year'] = 2024
        if 'kms_driven' not in cars_df.columns:
            # Default to 0 for new cars dataset (assuming they're new cars)
            cars_df['kms_driven'] = 0
        
        # Ensure required columns exist
        required_cols = ['company', 'name', 'fuel_type']
        if not all(col in cars_df.columns for col in required_cols):
            print(f"Warning: Missing required columns. Available: {cars_df.columns.tolist()}")
        
        print(f"New dataset (Cars Datasets 2025.csv) loaded: {len(cars_df)} records")
    except Exception as e:
        print(f"New dataset could not be read: {e}")
        cars_df = None

# Fallback to old dataset if new one not available
if cars_df is None and os.path.exists(CSV_PATH_OLD):
    try:
        cars_df = pd.read_csv(CSV_PATH_OLD)
        cars_df.columns = [c.strip().lower() for c in cars_df.columns]
        print(f"Old dataset (Cleaned_Car_data.csv) loaded: {len(cars_df)} records")
    except Exception as e:
        print(f"Old dataset could not be read: {e}")

# Fallback (broad, works for many selections)
def list_companies():
    if cars_df is not None:
        return sorted(cars_df['company'].unique().tolist())
    return []

def list_models(company):
    if cars_df is not None and company:
        return sorted(cars_df[cars_df['company'] == company]['name'].unique().tolist())
    return []

def list_fuels_by_model(model):
    if cars_df is not None and model:
        fuel_types = cars_df[cars_df['name'] == model]['fuel_type'].unique().tolist()
        # Remove LPG completely from the list
        filtered_fuels = [f for f in fuel_types if f != 'LPG']
        return sorted(filtered_fuels)
    return []

def list_years():
    from datetime import datetime
    current_year = datetime.now().year
    max_year = max(2025, current_year + 1)  # Allow up to next year for new cars
    
    if cars_df is not None and 'year' in cars_df.columns:
        # Get years from dataset
        dataset_years = [int(y) for y in cars_df['year'].unique().tolist() if pd.notna(y)]
        
        if dataset_years and len(set(dataset_years)) > 1:
            # If dataset has multiple years, use them plus extend forward
            min_dataset_year = min(dataset_years)
            max_dataset_year = max(dataset_years)
            # Extend from dataset years up to current year + 1
            all_years = sorted(set(dataset_years + list(range(max_dataset_year + 1, max_year + 1))))
            # Also include years before dataset minimum (for older used cars)
            if min_dataset_year > 2000:
                all_years = sorted(set(all_years + list(range(2000, min_dataset_year))))
            return all_years
        elif dataset_years:
            # Dataset has only one or few years (like all 2024)
            # Provide a full range for used car predictions
            min_year = min(dataset_years)
            # Return years from 2000 to current year + 1
            return list(range(2000, max_year + 1))
    
    # Fallback: return years 2000 to current year + 1 (for new cars)
    return list(range(2000, max_year + 1))

def list_fuel_types():
    if cars_df is not None:
        fuel_types = cars_df['fuel_type'].unique().tolist()
        # Remove LPG completely from the list
        filtered_fuels = [f for f in fuel_types if f != 'LPG']
        return sorted(filtered_fuels)
    return []

def generate_price_tips(company, car_model, year, fuel_type, kilo_driven, predicted_price):
    """
    Generate dynamic, personalized tips explaining why the car is priced at this level.
    Returns a list of relevant tips.
    """
    tips = []
    current_year = datetime.now().year  # Get current year dynamically
    car_age = current_year - year
    
    # Calculate metrics
    avg_km_per_year = kilo_driven / (car_age if car_age > 0 else 1)
    is_luxury = company in ['Audi', 'BMW', 'Mercedes-Benz', 'Land Rover', 'Jaguar', 'Mini', 'Volvo', 'Mitsubishi']
    is_popular_brand = company in ['Maruti', 'Hyundai', 'Honda', 'Toyota']
    is_new_car = car_age <= 3
    is_old_car = car_age >= 10
    low_mileage = avg_km_per_year < 10000
    high_mileage = avg_km_per_year > 25000
    very_high_mileage = kilo_driven > 150000
    
    # India vehicle age restrictions (important for pricing)
    petrol_age_limit = 15
    diesel_age_limit = 10
    years_left_petrol = petrol_age_limit - car_age if fuel_type == 'Petrol' else 999
    years_left_diesel = diesel_age_limit - car_age if fuel_type == 'Diesel' else 999
    near_scrap_petrol = years_left_petrol <= 2 and years_left_petrol > 0
    near_scrap_diesel = years_left_diesel <= 2 and years_left_diesel > 0
    
    # Determine price category
    if predicted_price >= 1000000:
        price_category = "premium"
    elif predicted_price >= 500000:
        price_category = "mid-range"
    else:
        price_category = "budget"
    
    # India-specific age restriction warnings (highest priority tips)
    if fuel_type == 'Diesel' and car_age >= diesel_age_limit:
        tips.append(f"🚫 CRITICAL: Diesel car over 10 years - cannot be driven in Delhi/NCR/Mumbai. Value drastically reduced!")
        tips.append(f"⚠️ This car can only be used in cities without age restrictions")
        tips.append(f"💔 Scrap value only - zero resale for city use")
        return tips[:3]  # Return immediately as this is the most important info
    elif fuel_type == 'Petrol' and car_age >= petrol_age_limit:
        tips.append(f"🚫 CRITICAL: Petrol car over 15 years - cannot be driven in Delhi/NCR/Mumbai. Value drastically reduced!")
        tips.append(f"⚠️ This car can only be used in cities without age restrictions")
        tips.append(f"💔 Scrap value only - zero resale for city use")
        return tips[:3]  # Return immediately
    elif near_scrap_diesel:
        tips.append(f"⚠️ URGENT: Only {years_left_diesel} year(s) left for diesel in Delhi/NCR/Mumbai!")
        tips.append(f"📉 Price heavily discounted due to approaching 10-year scrappage rule")
        tips.append(f"🕐 Consider selling soon before value drops to zero")
    elif near_scrap_petrol:
        tips.append(f"⚠️ URGENT: Only {years_left_petrol} year(s) left for petrol in Delhi/NCR/Mumbai!")
        tips.append(f"📉 Price heavily discounted due to approaching 15-year scrappage rule")
        tips.append(f"🕐 Consider selling soon before value drops to zero")
    
    # Generate tips based on actual characteristics
    if is_new_car and low_mileage and price_category == "premium":
        tips.append(f"✨ Premium pricing due to brand reputation ({company}) and excellent condition")
        tips.append(f"🚗 Very low usage (~{int(avg_km_per_year):,} km/year) indicates well-maintained vehicle")
        tips.append(f"💎 Recent model (just {car_age} years old) retains significant market value")
    elif is_new_car and low_mileage:
        tips.append(f"🎯 Good value for {car_age}-year-old car with minimal usage")
        tips.append(f"⛽ {fuel_type} variant offers better resale than alternatives")
        tips.append(f"📊 Low depreciation due to limited kilometers driven")
    elif is_new_car and high_mileage:
        tips.append(f"⚠️ High daily usage ({int(avg_km_per_year):,} km/year) reduces resale value significantly")
        tips.append(f"🔧 Consider service history - heavy use may require more maintenance")
        tips.append(f"💡 Price adjusted due to above-average wear and tear")
    elif is_old_car and low_mileage:
        if is_luxury:
            tips.append(f"🏆 Classic luxury car factor - {company} cars maintain value better over time")
            tips.append(f"📉 Age ({car_age} years) significantly impacts price despite low mileage")
            tips.append(f"⚙️ Lower price reflects aging technology and potential repair costs")
        else:
            tips.append(f"📉 Depreciation from age ({car_age} years) dominates over low mileage")
            tips.append(f"⚠️ Vintage cars need more maintenance - factor in potential repair costs")
            tips.append(f"💡 Low mileage is good, but age-related depreciation is substantial")
    elif is_old_car and high_mileage:
        tips.append(f"🔻 Significant depreciation from both age ({car_age} years) and high usage")
        tips.append(f"⚠️ Very high mileage ({kilo_driven:,} km) indicates heavy wear")
        tips.append(f"🛠️ Budget for potential major repairs on such a heavily used vehicle")
    elif very_high_mileage:
        tips.append(f"⚠️ Price reduced by 20% due to very high mileage ({kilo_driven:,} km)")
        tips.append(f"🔧 Engine and transmission nearing end-of-life cycle")
        tips.append(f"💡 Be prepared for higher maintenance costs")
    else:
        # Default tips for average cases
        tips.append(f"📊 {car_age}-year-old car with {fuel_type} engine priced competitively")
        tips.append(f"📈 Average mileage ({int(avg_km_per_year):,} km/year) shows normal usage pattern")
        if is_popular_brand:
            tips.append(f"🎯 Popular brand ({company}) retains better resale value")
        if fuel_type == 'Diesel':
            tips.append(f"⛽ Diesel variant typically commands 10-15% premium over petrol")
        elif fuel_type == 'Petrol':
            tips.append(f"⛽ Petrol engine offers lower maintenance costs")
    
    # Add brand-specific tips
    if is_luxury and price_category == "premium":
        tips.append(f"💎 Luxury brand premium included in pricing")
    elif is_popular_brand:
        tips.append(f"👍 {company} has strong market demand and better value retention")
    
    # Add fuel type insights
    if price_category == "premium" and fuel_type == 'Diesel':
        tips.append(f"💪 Diesel luxury cars offer excellent fuel efficiency and torque")
    
    return tips[:3]  # Return top 3 most relevant tips

def create_prediction_features(company, car_model, year, fuel_type, kilo_driven):
    """
    Create input DataFrame with all required features for model prediction.
    Handles both old models (5 features) and new enhanced models (9 features).
    """
    current_year = datetime.now().year
    car_age = max(0, current_year - year)
    km_per_year = kilo_driven / (car_age + 1) if car_age >= 0 else kilo_driven
    
    # Brand classification
    reliable_brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'Mahindra']
    luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Jaguar', 'Land Rover', 'Volvo']
    
    is_reliable_brand = 1 if company in reliable_brands else 0
    is_luxury_brand = 1 if company in luxury_brands else 0
    
    # Create DataFrame with all possible features
    # Enhanced model format (9 features) - matches train_enhanced_model_2025.py
    # Feature order: ['name', 'company', 'fuel_type', 'year', 'kms_driven', 'car_age', 'km_per_year', 'is_reliable_brand', 'is_luxury_brand']
    input_df = pd.DataFrame([[
        car_model, company, fuel_type, year, kilo_driven, car_age, km_per_year, is_reliable_brand, is_luxury_brand
    ]], columns=['name', 'company', 'fuel_type', 'year', 'kms_driven', 'car_age', 'km_per_year', 'is_reliable_brand', 'is_luxury_brand'])
    
    return input_df, 'enhanced'

def get_car_features(company, car_model, fuel_type=None):
    """
    Get car features/specifications from the dataset.
    Returns a dictionary with key features if found, None otherwise.
    """
    if cars_df is None:
        return None
    
    # Search for matching car
    mask = (cars_df['company'] == company) & (cars_df['name'] == car_model)
    if fuel_type:
        mask = mask & (cars_df['fuel_type'] == fuel_type)
    
    matching_cars = cars_df[mask]
    if len(matching_cars) > 0:
        car = matching_cars.iloc[0]
        features = {}
        
        # Extract key features
        feature_mapping = {
            'displacement': 'Displacement',
            'cylinders': 'Cylinders',
            'power': 'Power',
            'torque': 'Torque',
            'body_type': 'Body_Type',
            'seating_capacity': 'Seating_Capacity',
            'fuel_tank_capacity': 'Fuel_Tank_Capacity',
            'arai_certified_mileage': 'ARAI_Certified_Mileage',
            'gears': 'Gears',
            'type': 'Type',
            'boot_space': 'Boot_Space',
            'ground_clearance': 'Ground_Clearance',
            'wheelbase': 'Wheelbase',
            'height': 'Height',
            'length': 'Length',
            'width': 'Width'
        }
        
        for key, col in feature_mapping.items():
            col_lower = col.lower().strip()
            if col_lower in cars_df.columns:
                value = car[col_lower]
                if pd.notna(value) and str(value).strip() and str(value).strip().lower() not in ['not on offer', 'not applicable', '']:
                    features[key] = str(value).strip()
        
        # Extract additional features (yes/no fields)
        # Note: columns are already normalized to lowercase when loading CSV
        yes_no_features = {
            'abs': 'abs_(anti-lock_braking_system)',
            'airbags': 'number_of_airbags',
            'power_steering': 'power_steering',
            'power_windows': 'power_windows',
            'central_locking': 'central_locking',
            'bluetooth': 'bluetooth',
            'usb_compatibility': 'usb_compatibility',
            'android_auto': 'android_auto',
            'apple_carplay': 'apple_carplay',
            'cruise_control': 'cruise_control',
            'parking_assistance': 'parking_assistance',
            'navigation_system': 'navigation_system'
        }
        
        for key, col in yes_no_features.items():
            if col in cars_df.columns:
                value = car[col]
                if pd.notna(value) and str(value).strip().lower() in ['yes', 'y']:
                    features[key] = True
        
        return features if features else None
    return None

def get_exshowroom_price(company, car_model, fuel_type=None, purchase_year=None):
    """
    Get the ex-showroom price for a car from the dataset.
    If purchase_year is provided, calculates the historical ex-showroom price for that year.
    Returns the price if found, None otherwise.
    """
    if cars_df is None:
        return None
    
    # Search for matching car
    mask = (cars_df['company'] == company) & (cars_df['name'] == car_model)
    if fuel_type:
        mask = mask & (cars_df['fuel_type'] == fuel_type)
    
    matching_cars = cars_df[mask]
    if len(matching_cars) > 0:
        # Get current ex-showroom price from dataset
        current_exshowroom_price = matching_cars.iloc[0]['price']
        
        # If purchase_year is provided, calculate historical price for that year
        if purchase_year is not None:
            current_year = datetime.now().year
            years_ago = current_year - purchase_year
            
            if years_ago > 0:
                # Car prices typically increase 3-5% per year in India due to:
                # - Inflation
                # - Feature additions/updates
                # - BS norms updates (BS4 to BS6, etc.)
                # - Material cost increases
                
                # Use average of 4% annual increase for reverse calculation
                annual_increase_rate = 0.04
                
                # Reverse calculate: if price increased by 4% per year, divide by (1.04)^years
                historical_price = current_exshowroom_price / ((1 + annual_increase_rate) ** years_ago)
                
                # Also account for model updates/generations (typically every 5-7 years)
                # If car is older than 5 years, apply additional reduction for older generation
                if years_ago >= 7:
                    # Likely a different generation/model variant
                    historical_price = historical_price * 0.85  # 15% reduction for older gen
                elif years_ago >= 5:
                    # Mid-generation updates
                    historical_price = historical_price * 0.92  # 8% reduction
                
                return float(historical_price)
            else:
                # Purchase year is current or future year, use current price
                return float(current_exshowroom_price)
        
        # Return current ex-showroom price if no purchase year specified
        return float(current_exshowroom_price)
    return None

def validate_and_enhance_prediction(predicted_price, company, car_model, year, kilo_driven, fuel_type='Petrol'):
    """
    Validate and enhance the predicted price to make it more realistic.
    Returns a dictionary with predicted price, confidence range, and validation info.
    """
    # Convert predicted_price to native Python float to avoid JSON serialization issues
    predicted_price = float(predicted_price)
    
    result = {
        'predicted_price': float(predicted_price),
        'low_range': float(predicted_price * 0.85),  # 15% below
        'high_range': float(predicted_price * 1.15),  # 15% above
        'is_valid': True,
        'confidence': 'medium'
    }
    
    # Get ex-showroom price from dataset for the purchase year (if available)
    exshowroom_price = get_exshowroom_price(company, car_model, fuel_type, purchase_year=year)
    
    # Validate based on car age and kilometers
    current_year = datetime.now().year  # Get current year dynamically (2025, 2026, etc.)
    car_age = current_year - year
    
    # IMPORTANT: For NEW cars, price should NEVER exceed ex-showroom price
    if exshowroom_price:
        # For brand new cars (current year, very low kms), cap at ex-showroom price
        if car_age <= 1 and kilo_driven < 1000:
            # New car: price should be 90-95% of ex-showroom (5-10% discount for immediate purchase)
            max_new_price = exshowroom_price * 0.95
            if predicted_price > max_new_price:
                predicted_price = max_new_price
                result['confidence'] = 'high'  # High confidence for new cars with known ex-showroom price
        elif car_age > 1:
            # For used cars, apply depreciation based on age and mileage
            # Typical depreciation: 10-15% first year, 15-20% second year, then 10-15% per year
            if car_age == 1:
                age_depreciation = 0.10  # 10% depreciation
            elif car_age == 2:
                age_depreciation = 0.25  # 25% total after 2 years
            else:
                # After 2 years: 10% per additional year, capped at 80% total
                age_depreciation = min(0.25 + (car_age - 2) * 0.10, 0.80)
            
            # Mileage depreciation: roughly 0.5% per 10,000 km
            mileage_depreciation = min(kilo_driven / 10000 * 0.005, 0.15)  # Max 15% for mileage
            
            # Total depreciation (age has more weight)
            total_depreciation = age_depreciation + mileage_depreciation * 0.5
            total_depreciation = min(total_depreciation, 0.85)  # Max 85% depreciation
            
            # Calculate expected used car price
            expected_used_price = exshowroom_price * (1 - total_depreciation)
            
            # Adjust predicted price to be more realistic
            if predicted_price > exshowroom_price:
                # Can't be higher than new car price - use expected used price
                predicted_price = expected_used_price
            elif predicted_price > expected_used_price * 1.4:
                # If predicted is more than 40% above expected, cap it
                predicted_price = expected_used_price * 1.3  # Allow 30% variance
            elif predicted_price < expected_used_price * 0.5:
                # If predicted is less than 50% of expected, raise it
                predicted_price = expected_used_price * 0.7  # Minimum 70% of expected
    
    # Apply price statistics if available
    if price_stats:
        # Ensure price is within realistic bounds
        min_price = price_stats.get('min_price', 30000)
        max_price = price_stats.get('max_price', 5000000)
        
        # If we have ex-showroom price, use it as max for new cars
        if exshowroom_price and car_age <= 1:
            max_price = min(max_price, exshowroom_price * 1.05)  # Allow 5% above for taxes/on-road
        
        if predicted_price < min_price:
            predicted_price = min_price
            result['is_valid'] = False
            result['confidence'] = 'low'
        elif predicted_price > max_price:
            predicted_price = max_price
            result['is_valid'] = False
            result['confidence'] = 'low'
        
        # Adjust confidence range based on error margin - convert to native Python float
        error_margin = price_stats.get('error_margin', 200000)
        result['low_range'] = float(max(min_price, predicted_price - error_margin))
        result['high_range'] = float(min(max_price, predicted_price + error_margin))
    
    # Update the predicted_price in result - ensure it's a native Python float
    result['predicted_price'] = float(predicted_price)
    
    # India-specific: Check vehicle age restrictions and adjust price
    petrol_age_limit = 15
    diesel_age_limit = 10
    years_left_petrol = petrol_age_limit - car_age if fuel_type == 'Petrol' else 999
    years_left_diesel = diesel_age_limit - car_age if fuel_type == 'Diesel' else 999
    
    # Price adjustments based on India's scrappage rules (apply to already adjusted price) - convert to native Python float
    if fuel_type == 'Diesel' and car_age >= diesel_age_limit:
        # Diesel over 10 years: scrap value only
        result['predicted_price'] = float(min(50000, predicted_price * 0.05))  # Max 50k or 5% of original
        result['confidence'] = 'low'
        result['is_valid'] = False
    elif fuel_type == 'Petrol' and car_age >= petrol_age_limit:
        # Petrol over 15 years: scrap value only
        result['predicted_price'] = float(min(50000, predicted_price * 0.05))  # Max 50k or 5% of original
        result['confidence'] = 'low'
        result['is_valid'] = False
    elif fuel_type == 'Diesel' and car_age >= (diesel_age_limit - 2):
        # Diesel within 2 years of limit: heavy discount
        result['predicted_price'] = float(predicted_price * 0.4)  # 60% discount
        result['confidence'] = 'low'
    elif fuel_type == 'Petrol' and car_age >= (petrol_age_limit - 2):
        # Petrol within 2 years of limit: heavy discount
        result['predicted_price'] = float(predicted_price * 0.4)  # 60% discount
        result['confidence'] = 'low'
    elif fuel_type == 'Diesel' and car_age >= 8:
        # Diesel between 8-10 years: moderate discount
        result['predicted_price'] = float(predicted_price * 0.65)  # 35% discount
    elif fuel_type == 'Petrol' and car_age >= 13:
        # Petrol between 13-15 years: moderate discount
        result['predicted_price'] = float(predicted_price * 0.65)  # 35% discount
    
    # Regular age and mileage checks (only if not already adjusted above)
    if result['confidence'] == 'medium':
        # Older cars with low mileage might be suspicious
        if car_age > 15 and kilo_driven < 10000:
            result['confidence'] = 'low'
        # Very high mileage might also reduce confidence
        elif kilo_driven > 200000:
            result['confidence'] = 'low'
            # Adjust price downwards for very high mileage (apply on already adjusted price if any) - convert to native Python float
            result['predicted_price'] = float(result['predicted_price'] * 0.8)
        # Newer cars with reasonable mileage
        elif car_age < 5 and kilo_driven < 50000:
            result['confidence'] = 'high'
    
    # Round to nearest thousand for more realistic prices - convert to native Python float
    result['predicted_price'] = float(round(result['predicted_price'] / 1000) * 1000)
    
    # Recalculate ranges based on final adjusted price - convert to native Python float
    result['low_range'] = float(round(result['predicted_price'] * 0.85 / 1000) * 1000)
    result['high_range'] = float(round(result['predicted_price'] * 1.15 / 1000) * 1000)
    
    # Generate personalized price tips
    result['tips'] = generate_price_tips(company, car_model, year, fuel_type, kilo_driven, result['predicted_price'])
    
    # Add ex-showroom price to result if available
    if exshowroom_price:
        # Convert to native Python float to avoid JSON serialization issues
        result['exshowroom_price'] = float(exshowroom_price)
        # Calculate price decrease percentage
        price_decrease = float(exshowroom_price) - float(result['predicted_price'])
        price_decrease_percent = float((price_decrease / float(exshowroom_price)) * 100) if float(exshowroom_price) > 0 else 0.0
        result['price_decrease'] = float(price_decrease)
        result['price_decrease_percent'] = float(price_decrease_percent)
    else:
        result['exshowroom_price'] = None
        result['price_decrease'] = None
        result['price_decrease_percent'] = None
    
    return result

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/app")
def index():
    return render_template("dashboard.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

@app.route("/search-page")
def search_page():
    return render_template("search.html")

@app.route("/analytics-page")
def analytics_page():
    return render_template("analytics.html")

@app.route("/login-page")
def login_page():
    return render_template("login.html")

@app.route("/get_companies")
def get_companies():
    return jsonify(list_companies())

@app.route("/get_models_by_company")
def get_models_by_company():
    company = request.args.get('company')
    return jsonify(list_models(company))

@app.route("/get_fuel_by_model")
def get_fuel_by_model():
    company = request.args.get('company')
    model = request.args.get('model')
    return jsonify(list_fuels_by_model(model))

@app.route("/api/years")
def get_years():
    return jsonify(list_years())

@app.route("/api/fuel_types")
def get_fuel_types():
    return jsonify(list_fuel_types())

@app.route("/api/fuel_types_by_brand")
def get_fuel_types_by_brand():
    brand = request.args.get('brand', '').strip()
    if not brand:
        return jsonify(list_fuel_types())
    
    if cars_df is None:
        return jsonify([])
    
    # Get fuel types available for the specific brand
    brand_cars = cars_df[cars_df['company'] == brand]
    if len(brand_cars) == 0:
        return jsonify([])
    
    fuel_types = brand_cars['fuel_type'].unique().tolist()
    fuel_types = [str(f) for f in fuel_types if pd.notna(f)]
    fuel_types.sort()
    
    return jsonify(fuel_types)

@app.route("/api/brands")
def get_brands():
    return jsonify(list_companies())

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Handle both form data and JSON data
        if request.form:
            company = request.form.get('Company', '').strip()
            car_model = request.form.get('Model', '').strip()
            year_str = request.form.get('Year', '').strip()
            fuel_type = request.form.get('Fuel_Type_Petrol', '').strip()
            kilo_driven_str = request.form.get('Kms_Driven', '').strip()
            
            # Validate required fields
            if not company:
                return jsonify({"error": "Company is required"}), 400
            if not car_model:
                return jsonify({"error": "Model is required"}), 400
            if not year_str or not kilo_driven_str:
                return jsonify({"error": "Year and Kms_Driven are required fields"}), 400
            if not fuel_type:
                return jsonify({"error": "Fuel Type is required"}), 400
            
            try:
                year = int(year_str)
                kilo_driven = int(kilo_driven_str)
            except ValueError:
                return jsonify({"error": "Year and Kms_Driven must be valid numbers"}), 400
        else:
            data = request.get_json(force=True)
            company = data.get('company', '').strip()
            car_model = data.get('model', '').strip()
            year_str = str(data.get('year', '')).strip()
            fuel_type = data.get('fuel_type', '').strip()
            kilo_driven_str = str(data.get('kms_driven', '')).strip()
            
            # Validate required fields
            if not company:
                return jsonify({"error": "company is required"}), 400
            if not car_model:
                return jsonify({"error": "model is required"}), 400
            if not year_str or not kilo_driven_str:
                return jsonify({"error": "year and kms_driven are required fields"}), 400
            if not fuel_type:
                return jsonify({"error": "fuel_type is required"}), 400
            
            try:
                year = int(year_str)
                kilo_driven = int(kilo_driven_str)
            except ValueError:
                return jsonify({"error": "year and kms_driven must be valid numbers"}), 400

        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Please check if model files exist."}), 500

        # Create input DataFrame with all required features
        input_df, model_type = create_prediction_features(company, car_model, year, fuel_type, kilo_driven)
        
        # Make prediction
        try:
            predicted_price_log = model.predict(input_df)[0]
            predicted_price_log = float(predicted_price_log)
            
            # Enhanced model uses log transformation (log1p)
            # If prediction value is between 11-15 (typical log scale for car prices), exponentiate
            # Car prices in log scale: log1p(50000) ≈ 10.8, log1p(5000000) ≈ 15.4
            if 11 <= predicted_price_log <= 16:
                # Likely log-transformed, exponentiate back using expm1
                predicted_price = np.expm1(predicted_price_log)
            elif predicted_price_log < 11:
                # Very small log value, still exponentiate
                predicted_price = np.expm1(predicted_price_log)
            else:
                # Regular prediction (already in normal scale)
                predicted_price = predicted_price_log
                
            # Ensure predicted_price is reasonable
            if predicted_price < 10000 or predicted_price > 50000000:
                # If value seems off, try without log transformation
                predicted_price = predicted_price_log
                
        except Exception as pred_error:
            # If enhanced model fails, try basic model format as fallback
            try:
                input_df_basic = pd.DataFrame([[
                    car_model, company, year, fuel_type, kilo_driven
                ]], columns=['name', 'company', 'year', 'fuel_type', 'kms_driven'])
                predicted_price = float(model.predict(input_df_basic)[0])
            except Exception as fallback_error:
                return jsonify({"error": f"Prediction failed. Enhanced model error: {str(pred_error)}. Fallback error: {str(fallback_error)}"}), 500
        
        # Validate and enhance prediction
        enhanced = validate_and_enhance_prediction(predicted_price, company, car_model, year, kilo_driven, fuel_type)
        
        # Return JSON response
        return jsonify(enhanced)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict_json():
    try:
        # Handle both form data and JSON data
        if request.form:
            company = request.form.get('Company', '').strip()
            car_model = request.form.get('Model', '').strip()
            year_str = request.form.get('Year', '').strip()
            fuel_type = request.form.get('Fuel_Type_Petrol', '').strip()
            kilo_driven_str = request.form.get('Kms_Driven', '').strip()
            
            # Validate required fields
            if not company:
                return jsonify({"error": "Company is required"}), 400
            if not car_model:
                return jsonify({"error": "Model is required"}), 400
            if not year_str or not kilo_driven_str:
                return jsonify({"error": "Year and Kms_Driven are required fields"}), 400
            if not fuel_type:
                return jsonify({"error": "Fuel Type is required"}), 400
            
            try:
                year = int(year_str)
                kilo_driven = int(kilo_driven_str)
            except ValueError:
                return jsonify({"error": "Year and Kms_Driven must be valid numbers"}), 400
        else:
            data = request.get_json(force=True)
            company = (data.get('Company') or data.get('company') or '').strip()
            car_model = (data.get('Model') or data.get('model') or '').strip()
            year_str = str(data.get('Year') or data.get('year') or '').strip()
            fuel_type = (data.get('Fuel_Type_Petrol') or data.get('fuel_type') or '').strip()
            kilo_driven_str = str(data.get('Kms_Driven') or data.get('kms_driven') or '').strip()
            
            # Validate required fields
            if not company:
                return jsonify({"error": "company is required"}), 400
            if not car_model:
                return jsonify({"error": "model is required"}), 400
            if not year_str or not kilo_driven_str:
                return jsonify({"error": "year and kms_driven are required fields"}), 400
            if not fuel_type:
                return jsonify({"error": "fuel_type is required"}), 400
            
            try:
                year = int(year_str)
                kilo_driven = int(kilo_driven_str)
            except ValueError:
                return jsonify({"error": "year and kms_driven must be valid numbers"}), 400

        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Please check if model files exist."})

        # Create input DataFrame with all required features
        input_df, model_type = create_prediction_features(company, car_model, year, fuel_type, kilo_driven)
        
        # Make prediction
        try:
            predicted_price_log = model.predict(input_df)[0]
            predicted_price_log = float(predicted_price_log)
            
            # Enhanced model uses log transformation (log1p)
            # If prediction value is between 11-15 (typical log scale for car prices), exponentiate
            # Car prices in log scale: log1p(50000) ≈ 10.8, log1p(5000000) ≈ 15.4
            if 11 <= predicted_price_log <= 16:
                # Likely log-transformed, exponentiate back using expm1
                predicted_price = np.expm1(predicted_price_log)
            elif predicted_price_log < 11:
                # Very small log value, still exponentiate
                predicted_price = np.expm1(predicted_price_log)
            else:
                # Regular prediction (already in normal scale)
                predicted_price = predicted_price_log
                
            # Ensure predicted_price is reasonable
            if predicted_price < 10000 or predicted_price > 50000000:
                # If value seems off, try without log transformation
                predicted_price = predicted_price_log
                
        except Exception as pred_error:
            # If enhanced model fails, try basic model format as fallback
            try:
                input_df_basic = pd.DataFrame([[
                    car_model, company, year, fuel_type, kilo_driven
                ]], columns=['name', 'company', 'year', 'fuel_type', 'kms_driven'])
                predicted_price = float(model.predict(input_df_basic)[0])
            except Exception as fallback_error:
                return jsonify({"error": f"Prediction failed. Enhanced model error: {str(pred_error)}. Fallback error: {str(fallback_error)}"})
        
        # Validate and enhance prediction
        enhanced = validate_and_enhance_prediction(predicted_price, company, car_model, year, kilo_driven, fuel_type)
        
        # Format prices with commas
        formatted_price = f"₹{enhanced['predicted_price']:,}"
        formatted_low = f"₹{enhanced['low_range']:,}"
        formatted_high = f"₹{enhanced['high_range']:,}"
        
        # Format ex-showroom price if available
        formatted_exshowroom = None
        if enhanced.get('exshowroom_price'):
            formatted_exshowroom = f"₹{enhanced['exshowroom_price']:,.0f}"
        
        # Get car features
        car_features = get_car_features(company, car_model, fuel_type)
        
        # Create enhanced response - ensure all numeric values are native Python types
        response = {
            "prediction": f"Predicted Price: {formatted_price}",
            "price": float(enhanced['predicted_price']),
            "low_range": float(enhanced['low_range']),
            "high_range": float(enhanced['high_range']),
            "confidence": enhanced['confidence'],
            "is_valid": enhanced['is_valid'],
            "formatted_price": formatted_price,
            "formatted_low": formatted_low,
            "formatted_high": formatted_high,
            "tips": enhanced.get('tips', []),
            "exshowroom_price": float(enhanced.get('exshowroom_price')) if enhanced.get('exshowroom_price') is not None else None,
            "formatted_exshowroom": formatted_exshowroom,
            "price_decrease": float(enhanced.get('price_decrease')) if enhanced.get('price_decrease') is not None else None,
            "price_decrease_percent": float(enhanced.get('price_decrease_percent')) if enhanced.get('price_decrease_percent') is not None else None,
            "car_features": car_features
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/search', methods=['POST'])
def search_cars():
    try:
        data = request.get_json()
        brand = data.get('brand', '')
        fuel_type = data.get('fuel_type', '')
        
        if cars_df is None:
            return jsonify([])
        
        # Filter cars based on search criteria
        filtered_cars = cars_df.copy()
        
        if brand:
            filtered_cars = filtered_cars[filtered_cars['company'] == brand]
        
        if fuel_type:
            filtered_cars = filtered_cars[filtered_cars['fuel_type'] == fuel_type]
        
        # Group by company, name, and fuel_type to get unique model+fuel combinations
        # This ensures we don't show multiple variants of the same model with different prices
        # We'll take the first occurrence (or you could average the prices if needed)
        filtered_cars = filtered_cars.drop_duplicates(subset=['company', 'name', 'fuel_type'], keep='first')
        
        # Sort by price (or company, name) for consistent ordering
        filtered_cars = filtered_cars.sort_values(['company', 'name', 'price'])
        
        # Get top 20 results
        top_cars = filtered_cars.head(20)
        
        # Prepare results with additional information
        results = []
        seen_combinations = set()  # Track seen combinations to avoid duplicates
        for _, car in top_cars.iterrows():
            # Create a unique key for this model (company + name + fuel_type)
            model_key = (str(car['company']).strip(), str(car['name']).strip(), str(car['fuel_type']).strip())
            
            # Skip if we've already seen this exact combination
            if model_key in seen_combinations:
                continue
            seen_combinations.add(model_key)
            car_data = {
                'company': car['company'],
                'name': car['name'],
                'year': int(car['year']) if pd.notna(car['year']) else None,
                'fuel_type': car['fuel_type'],
                'price': float(car['price']) if pd.notna(car['price']) else None
            }
            
            # Get fuel types available for this company
            company_cars = cars_df[cars_df['company'] == car['company']]
            available_fuel_types = company_cars['fuel_type'].unique().tolist()
            car_data['available_fuel_types'] = [str(f) for f in available_fuel_types if pd.notna(f)]
            
            # Get launch years for this car model (group by company and name)
            model_cars = cars_df[(cars_df['company'] == car['company']) & (cars_df['name'] == car['name'])]
            all_years = sorted(model_cars['year'].unique().tolist())
            all_years = [int(y) for y in all_years if pd.notna(y)]
            
            # Filter out default 2024 year if it's the only year (meaning it's just a placeholder)
            # Only show launch years if we have actual year data (multiple years or non-2024 years)
            if len(all_years) == 1 and all_years[0] == 2024:
                # Default year - don't show launch years
                car_data['launch_years'] = []
            elif len(all_years) > 1:
                # Multiple years - show all of them (these are likely actual model years)
                car_data['launch_years'] = all_years
            elif len(all_years) == 1 and all_years[0] != 2024:
                # Single non-default year - show it
                car_data['launch_years'] = all_years
            else:
                # No valid years
                car_data['launch_years'] = []
            
            # Get car features for tooltip
            car_features = get_car_features(car['company'], car['name'], car['fuel_type'])
            car_data['features'] = car_features
            
            results.append(car_data)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)