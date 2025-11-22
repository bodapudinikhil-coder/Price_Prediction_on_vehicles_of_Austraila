# Vehicle Price Prediction & Recommendation System

## Description

This project predicts used vehicle prices using supervised machine learning and enhances the predictions with a Prolog-based expert system for explainable deal assessment. It trains and compares several regression models on an Australian vehicle price dataset, selects the best-performing model, and combines model outputs with symbolic rules to classify deals (e.g., bargain vs fair) and provide human-readable reasons.

## Features

- Data loading and preprocessing for an Australian vehicle price dataset.  
- Safe feature engineering, including vehicle age, recent-year flag, and low-kilometre indicators.  
- Training and evaluation of multiple regression models: Linear Regression, Decision Tree, Random Forest, XGBoost.  
- Automatic selection and saving of the best model based on test RMSE.  
- Export of a final engineered dataset for analysis.  
- Integration with a Prolog knowledge base for rule-based recommendations and explanations.

## Project Structure

- `Prediction_Price.py` – Main Python script for data processing, model training, evaluation, selection, and Prolog integration.  
- `australian_vehicle_prices.csv` – Original dataset with vehicle listings and prices.  
- `final_dataset_with_features.csv` – Engineered dataset output from the script.  
- `best_price_predictor.joblib` – Serialized best-performing model.  
- `carkb.pl` – Prolog knowledge base file with domain rules (must be present).

## Data and Features

Input CSV with columns such as:

- Price  
- Year  
- Kilometres  
- FuelConsumption  
- Seats  
- Transmission (encoded)  
- FuelType (encoded)  
- BodyType (encoded)  

Derived features created:

- `CarAge`: 2025 - Year  
- `IsRecent`: Flag if year ≥ 2021  
- `LowKM`: Flag if kilometres < 50,000 (threshold configurable)  

Target variable is `Price`. Predictors include original and engineered features.

## Models and Methodology

- Preprocessing with median imputation using `SimpleImputer` in a `ColumnTransformer`.  
- Train/test split (80/20).  
- Models trained within pipelines:  
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - XGBRegressor (XGBoost)  
- Evaluation metric: Root Mean Squared Error (RMSE) on test data.  
- Best model selected by lowest RMSE and saved as `best_price_predictor.joblib`.

## Prolog Integration and Recommendations

- Loads `best_price_predictor.joblib`.  
- Reads engineered dataset `final_dataset_with_features.csv`.  
- Consults `carkb.pl` Prolog knowledge base.  
- Samples subset of rows for demonstration.  
- Asserts facts like predicted price, actual price, year, mileage, and features into Prolog.  
- Queries Prolog predicate `recommend_car/2` for deal classification and reasons.  
- Prints assessment such as "BARGAIN!" or "Fair Price" with explanations.

This combines quantitative ML prediction with symbolic reasoning for interpretability.

## Requirements

- Python 3.x  
- pandas, numpy, scikit-learn, xgboost, joblib, pyswip libraries  
- SWI-Prolog (or a compatible Prolog) installed and accessible  
- Prolog KB file `carkb.pl`

Install Python dependencies with:

pip install pandas numpy scikit-learn xgboost joblib pyswip

## Installation

1. Clone or download the repository.  
2. Place `australian_vehicle_prices.csv` and `carkb.pl` in the root directory.  
3. Install required Python packages and SWI-Prolog.

## Usage

Run the main script:
python Prediction_Price.py

This loads data, preprocesses it, trains models, evaluates and selects the best model, saves it, exports the engineered dataset, and runs the Prolog-based deal recommendation demo.

## Example Output

Linear Regression RMSE: 7,100

Decision Tree RMSE: 6,200

Random Forest RMSE: 4,900

XGBoost RMSE: 4,300

Best model selected: XGBoost

Test RMSE: 4,300

Model saved as best_price_predictor.joblib

Car 10523 | Year: 2019 | KM: 32,000
Actual Price: 27,500
ML Predicted: 25,800
Deal Assessment: BARGAIN!
Reasons: low mileage, recent model year, efficient fuel consumption

## Customization

- Modify feature engineering thresholds and logic in the script.  
- Tune model hyperparameters.  
- Extend Prolog rules in `carkb.pl` for more domain knowledge.  
- Adjust sample size or use full dataset for Prolog recommendations.

## License

Open-source for academic and research use; please attribute appropriately.

---

This comprehensive README covers your project’s description, setup instructions, usage, and technical details to make the repo easy to understand and use.
