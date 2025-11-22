#!/usr/bin/env python
# coding: utf-8

# # Student Name: Nikhil Bodapudi
# # Student FAN: boda0015
# # File: Prediction_Price.py
# # Date: 21-11-2025
# # Description: The latest information on car prices in Australia for the year 2023. Which covers different features of cars sold in the Australian market.
# # Usage: Australian Vehicle Prices.csv
# # Licence:

# # Import Libraries and Load Data

# In[1]:


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("encoded_australian_vehicle_prices.csv")


# # Feature Engineering (No Leakage)

# In[2]:


# Safe feature engineering (no leakage)
df['Car_Age'] = 2025 - df['Year']
df['Is_Recent'] = (df['Year'] >= 2021).astype(int)
df['Low_KM'] = (df['Kilometres'] <= 50000).astype(int)


# # Preprocessing Pipeline and Train-Test Split

# In[3]:


# Features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Preprocessing pipeline (median imputation)
preprocessor = ColumnTransformer([
    ('impute', SimpleImputer(strategy='median'), X.columns)
], remainder='passthrough')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# # Define Models and Train with Evaluation

# ## Train and Evaluate Linear Regression

# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

lr_model = Pipeline([
    ('prep', preprocessor),
    ('model', LinearRegression())
])

lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

print(f"Linear Regression RMSE: ${lr_rmse:,.0f}")


# ## Train and Evaluate Decision Tree

# In[5]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

dt_model = Pipeline([
    ('prep', preprocessor),
    ('model', DecisionTreeRegressor(max_depth=18, random_state=42))
])

dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_preds))

print(f"Decision Tree RMSE: ${dt_rmse:,.0f}")


# ## Train and Evaluate Random Forest

# In[6]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

rf_model = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestRegressor(n_estimators=600, max_depth=22,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1))
])

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

print(f"Random Forest RMSE: ${rf_rmse:,.0f}")


# ## Train and Evaluate XGBoost

# In[7]:


from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

xgb_model = Pipeline([
    ('prep', preprocessor),
    ('model', XGBRegressor(n_estimators=800, learning_rate=0.06, max_depth=10,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1))
])

xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))

print(f"XGBoost RMSE: ${xgb_rmse:,.0f}")


# # Reporting the Best Model

# In[8]:


import joblib

# Dictionary of models and their RMSE values from previous parts
model_rmse_dict = {
    'Linear Regression': lr_rmse,
    'Decision Tree': dt_rmse,
    'Random Forest': rf_rmse,
    'XGBoost': xgb_rmse
}

model_obj_dict = {
    'Linear Regression': lr_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

# Select model with lowest RMSE
best_name = min(model_rmse_dict, key=model_rmse_dict.get)
best_rmse = model_rmse_dict[best_name]
best_model = model_obj_dict[best_name]

print("\n" + "=" * 70)
print(f"Best model selected: {best_name}")
print(f"Test RMSE: ${best_rmse:,.0f}")
print("=" * 70)

# Save best model for future use
joblib.dump(best_model, "best_price_predictor.joblib")
print(f"Model saved as 'best_price_predictor.joblib'")


# # Predict the prices 

# In[9]:


df = pd.read_csv("encoded_australian_vehicle_prices.csv")
df['Car_Age'] = 2025 - df['Year']
df['Is_Recent'] = (df['Year'] >= 2021).astype(int)
df['Low_KM'] = (df['Kilometres'] <= 50000).astype(int)
df.to_csv("final_dataset_with_features.csv", index=False)


# In[10]:


import joblib
import pandas as pd
from pyswip import Prolog

model = joblib.load("best_price_predictor.joblib")
df = pd.read_csv("final_dataset_with_features.csv")  
prolog = Prolog()
prolog.consult("car_kb.pl")

for idx in df.sample(8, random_state=42).index:
    row = df.loc[idx]
    actual = float(row['Price'])
    input_data = row.drop('Price').to_frame().T
    pred = float(model.predict(input_data)[0])

    car = f"car{idx}"
    prolog.assertz(f"predicted_price({car}, {int(pred)})")
    prolog.assertz(f"actual_price({car}, {int(actual)})")
    prolog.assertz(f"year({car}, {int(row['Year'])})")
    prolog.assertz(f"km({car}, {int(row['Kilometres'])})")
    
    # Assert new predicates
    prolog.assertz(f"fuelcons({car}, {float(row['FuelConsumption'])})")
    prolog.assertz(f"seats({car}, {int(row['Seats'])})")
    prolog.assertz(f"transmission({car}, {int(row['Transmission'])})")
    prolog.assertz(f"fueltype({car}, {int(row['FuelType'])})")
    prolog.assertz(f"bodytype({car}, {int(row['BodyType'])})")

    result = list(prolog.query(f"recommend({car}, R)"))
    if result and 'R' in result[0] and result[0]['R']:
        reasons = result[0]['R']
        reasons = [r.decode() if isinstance(r, bytes) else str(r) for r in reasons]
        unique_reasons = list(dict.fromkeys(reasons))  # Remove duplicates
    else:
        unique_reasons = ["No specific reasons"]

    if len(unique_reasons) > 5:
        reasons_print = ', '.join(unique_reasons[:5]) + " ... (and more)"
    else:
        reasons_print = ', '.join(unique_reasons)

    deal = "BARGAIN!" if actual < pred * 0.88 else "Fair Price"

    print(f"\nCar {idx} | Year: {int(row['Year'])} | KM: {int(row['Kilometres']):,}")
    print(f"   Actual Price    : ${actual:,.0f}")
    print(f"   ML Predicted    : ${pred:,.0f}")
    print(f"   Deal Assessment : {deal}")
    print(f"   Reasons         : {reasons_print}")


# In[ ]:




