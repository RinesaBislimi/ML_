
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


file_path = "dataset.csv"
data = pd.read_csv(file_path)


data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"], errors='coerce')
data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"], errors='coerce')

data["pickup_hour"] = data["tpep_pickup_datetime"].dt.hour
data["pickup_day"] = data["tpep_pickup_datetime"].dt.day
data["pickup_month"] = data["tpep_pickup_datetime"].dt.month

data["trip_duration"] = (data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]).dt.total_seconds()

columns_to_drop = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "store_and_fwd_flag"]
data.drop([col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)

data.fillna(data.median(numeric_only=True), inplace=True)

data["total_amount"] = pd.to_numeric(data["total_amount"], errors="coerce")


X = data.drop(columns=["total_amount"])
y = data["total_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42),
    "CatBoost": cb.CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}
    
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

print("Model training and evaluation completed. Results saved to results.pkl")