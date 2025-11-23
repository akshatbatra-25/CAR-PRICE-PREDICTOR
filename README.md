🚗 Car Price Predictor An AI-powered Machine Learning web application that predicts the resale price of a used car based on key features such as company, model, fuel type, kilometers driven, and year of manufacture. This project demonstrates data-driven decision-making in the automobile market using modern ML algorithms and an interactive Flask web interface.

🌟 Features

📊 Predicts used car resale prices accurately ⚙️ Built with Machine Learning models (Random Forest & Gradient Boosting) 🧠 Includes data preprocessing pipeline (encoding, scaling, feature engineering) 🧾 Interactive Flask-based frontend for real-time predictions 💾 Trained on real Indian car market data 📈 Evaluated with MAE, RMSE, R² Score, and MAPE 🧩 Clean modular structure (Backend + Frontend + Model)

🏗️ Technology Stack Backend Framework: Flask (Python) ML Libraries: scikit-learn (RandomForest, GradientBoosting) NumPy, Pandas Data Processing: OneHotEncoder, StandardScaler, Pipeline Model Evaluation: MAE, RMSE, R² Score, MAPE

Frontend Languages: HTML5, CSS3, JavaScript Framework: Bootstrap (for UI styling)

🧮 Model Workflow Dataset Import → Load the car dataset (CSV) Data Cleaning → Handle missing values and outliers Feature Engineering → Encode categorical data, normalize numeric fields Model Training → Train models (Random Forest, Gradient Boosting) Evaluation → Compare models based on metrics (MAE, RMSE, R²) Deployment → Save best model using pickle and connect with Flask app
