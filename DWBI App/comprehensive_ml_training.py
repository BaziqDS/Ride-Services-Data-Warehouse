"""
comprehensive_ml_training.py

Redesigned ML Training Pipeline with:
- Fare Amount Prediction (Regression)
- Zone Demand Prediction (Regression - rides per zone per hour)
- 4 models per task with simplified overfitting detection
- Best model selection based on R² score and R² gap
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional boosters
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


class MLTrainingPipeline:
    def __init__(self, db_config):
        self.db_config = db_config
        self.fare_models = {}
        self.demand_models = {}
        self.fare_metrics = {}
        self.demand_metrics = {}
        self.best_fare_model = None
        self.best_demand_model = None
        self.fare_scaler = None
        self.demand_scaler = None
        
    # -------------------------
    # Data Loaders
    # -------------------------
    def load_fare_data(self):
        """Load data for fare prediction"""
        query = """
            SELECT
                fr.ride_id,
                fr.customer_id,
                fr.driver_id,
                fr.vehicle_id,
                fr.payment_method_id,
                fr.pickup_zone_id,
                fr.dropoff_zone_id,
                fr.ride_date,
                fr.fare_amount,
                fr.distance_km,
                fr.duration_minutes,

                (fr.ride_date - dc.signup_date) AS customer_age_days,
                (fr.ride_date - dd.join_date) AS driver_experience_days,

                dv.year AS vehicle_year,
                dv.make AS vehicle_make,
                dv.model AS vehicle_model,

                dz_pickup.center_lat AS pickup_lat,
                dz_pickup.center_lon AS pickup_lon,
                dz_dropoff.center_lat AS dropoff_lat,
                dz_dropoff.center_lon AS dropoff_lon,

                dw.temperature_2m,
                dw.precipitation,
                dw.rain,
                dw.snowfall,
                dw.cloud_cover,
                dw.avg_wind_speed,

                EXTRACT(HOUR FROM fr.ride_date::timestamp) AS hour,
                EXTRACT(DOW FROM fr.ride_date::timestamp) AS day_of_week,
                EXTRACT(MONTH FROM fr.ride_date::timestamp) AS month,
                EXTRACT(YEAR FROM fr.ride_date::timestamp) AS year

            FROM fact_rides fr
            LEFT JOIN dim_customers dc ON fr.customer_id = dc.customer_id
            LEFT JOIN dim_drivers dd ON fr.driver_id = dd.driver_id
            LEFT JOIN dim_vehicles dv ON fr.vehicle_id = dv.vehicle_id
            LEFT JOIN dim_zones dz_pickup ON fr.pickup_zone_id = dz_pickup.zone_id
            LEFT JOIN dim_zones dz_dropoff ON fr.dropoff_zone_id = dz_dropoff.zone_id
            LEFT JOIN dim_weather dw
                ON fr.pickup_zone_id = dw.zone_id
               AND fr.ride_date = dw.date

            WHERE fr.ride_date IS NOT NULL
              AND fr.fare_amount > 0
              AND fr.distance_km > 0
              AND fr.duration_minutes > 0;
        """
        
        try:
            conn = psycopg2.connect(**self.db_config)
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading fare data: {e}")
            return None

    def load_demand_data(self):
        """Load data for zone demand prediction (rides per zone per hour)"""
        query = """
            SELECT
                fr.pickup_zone_id AS zone_id,
                EXTRACT(HOUR FROM fr.ride_date::timestamp) AS hour,
                EXTRACT(DOW FROM fr.ride_date::timestamp) AS day_of_week,
                EXTRACT(MONTH FROM fr.ride_date::timestamp) AS month,
                EXTRACT(YEAR FROM fr.ride_date::timestamp) AS year,
                DATE(fr.ride_date) AS date,
                
                dz.center_lat,
                dz.center_lon,
                
                dw.temperature_2m,
                dw.precipitation,
                dw.rain,
                dw.snowfall,
                dw.cloud_cover,
                dw.avg_wind_speed,
                
                COUNT(*) AS ride_count
                
            FROM fact_rides fr
            LEFT JOIN dim_zones dz ON fr.pickup_zone_id = dz.zone_id
            LEFT JOIN dim_weather dw
                ON fr.pickup_zone_id = dw.zone_id
               AND DATE(fr.ride_date) = dw.date
            
            WHERE fr.ride_date IS NOT NULL
            
            GROUP BY 
                fr.pickup_zone_id,
                EXTRACT(HOUR FROM fr.ride_date::timestamp),
                EXTRACT(DOW FROM fr.ride_date::timestamp),
                EXTRACT(MONTH FROM fr.ride_date::timestamp),
                EXTRACT(YEAR FROM fr.ride_date::timestamp),
                DATE(fr.ride_date),
                dz.center_lat,
                dz.center_lon,
                dw.temperature_2m,
                dw.precipitation,
                dw.rain,
                dw.snowfall,
                dw.cloud_cover,
                dw.avg_wind_speed;
        """
        
        try:
            conn = psycopg2.connect(**self.db_config)
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading demand data: {e}")
            return None

    # -------------------------
    # Feature Engineering
    # -------------------------
    def engineer_fare_features(self, df):
        """Engineer features for fare prediction"""
        df = df.copy()
        
        # Convert to numeric
        df["customer_age_days"] = pd.to_numeric(df["customer_age_days"], errors="coerce").fillna(0).astype(float)
        df["driver_experience_days"] = pd.to_numeric(df["driver_experience_days"], errors="coerce").fillna(0).astype(float)
        df["vehicle_year"] = pd.to_numeric(df.get("vehicle_year", 2020), errors="coerce").fillna(2020)
        df["year"] = pd.to_numeric(df.get("year", 2024), errors="coerce").fillna(2024)
        df["vehicle_age"] = df["year"] - df["vehicle_year"]
        
        # Trip features
        df["distance_km"] = pd.to_numeric(df.get("distance_km", 0), errors="coerce").fillna(0)
        df["duration_minutes"] = pd.to_numeric(df.get("duration_minutes", 0), errors="coerce").fillna(0)
        df["avg_speed_kmh"] = df["distance_km"] / (df["duration_minutes"] / 60 + 1e-6)
        df["zone_distance"] = (df["pickup_zone_id"] - df["dropoff_zone_id"]).abs()
        
        # Weather features
        df["temperature_2m"] = df["temperature_2m"].fillna(20)
        df["precipitation"] = df["precipitation"].fillna(0)
        df["rain"] = df["rain"].fillna(0)
        df["snowfall"] = df["snowfall"].fillna(0)
        df["cloud_cover"] = df["cloud_cover"].fillna(50)
        df["avg_wind_speed"] = df["avg_wind_speed"].fillna(5)
        df["weather_severity"] = df["precipitation"] + df["rain"] + df["snowfall"]
        
        # Time features
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)
        df["is_rush_hour"] = df["hour"].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
        df["is_night"] = df["hour"].apply(lambda x: 1 if (x <= 5 or x >= 22) else 0)
        
        # Drop invalid rows
        df = df.dropna(subset=["fare_amount", "distance_km", "duration_minutes"])
        df = df[df["fare_amount"] > 0]
        
        return df

    def engineer_demand_features(self, df):
        """Engineer features for demand prediction"""
        df = df.copy()
        
        # Location features
        df["center_lat"] = df["center_lat"].fillna(40.7128)
        df["center_lon"] = df["center_lon"].fillna(-74.0060)
        
        # Weather features
        df["temperature_2m"] = df["temperature_2m"].fillna(20)
        df["precipitation"] = df["precipitation"].fillna(0)
        df["rain"] = df["rain"].fillna(0)
        df["snowfall"] = df["snowfall"].fillna(0)
        df["cloud_cover"] = df["cloud_cover"].fillna(50)
        df["avg_wind_speed"] = df["avg_wind_speed"].fillna(5)
        df["weather_severity"] = df["precipitation"] + df["rain"] + df["snowfall"]
        
        # Time features
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)
        df["is_rush_hour"] = df["hour"].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
        df["is_night"] = df["hour"].apply(lambda x: 1 if (x <= 5 or x >= 22) else 0)
        
        return df

    # -------------------------
    # Prepare Features
    # -------------------------
    def prepare_fare_features(self, df):
        """Prepare features for fare prediction"""
        feature_cols = [
            "distance_km", "duration_minutes", "avg_speed_kmh",
            "pickup_zone_id", "dropoff_zone_id", "zone_distance",
            "customer_age_days", "driver_experience_days", "vehicle_age",
            "temperature_2m", "precipitation", "weather_severity",
            "cloud_cover", "avg_wind_speed",
            "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "is_night"
        ]
        
        X = df[feature_cols].copy()
        X = X.fillna(0).astype(float)
        y = df["fare_amount"].astype(float)
        
        self.fare_scaler = StandardScaler()
        X_scaled = self.fare_scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols

    def prepare_demand_features(self, df):
        """Prepare features for demand prediction"""
        feature_cols = [
            "zone_id", "center_lat", "center_lon",
            "temperature_2m", "precipitation", "weather_severity",
            "cloud_cover", "avg_wind_speed",
            "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "is_night"
        ]
        
        X = df[feature_cols].copy()
        X = X.fillna(0).astype(float)
        y = df["ride_count"].astype(float)
        
        self.demand_scaler = StandardScaler()
        X_scaled = self.demand_scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols

    # -------------------------
    # Model Initialization
    # -------------------------
    def initialize_models(self):
        """Initialize models with regularization to prevent overfitting"""
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGBRegressor is not None:
            models["XGBoost"] = XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
        
        # Add LightGBM if available
        if LGBMRegressor is not None:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
        
        return models

    # -------------------------
    # Model Selection Score
    # -------------------------
    def calculate_selection_score(self, test_r2, r2_gap):
        """
        Calculate model selection score based on:
        1. Test R² score (higher is better)
        2. R² gap (lower is better)
        
        Score = test_r2 - (penalty_factor * r2_gap)
        """
        # Penalty factor for R² gap (adjust this to control how much we penalize overfitting)
        penalty_factor = 2.0
        
        selection_score = test_r2 - (penalty_factor * r2_gap)
        
        return selection_score

    # -------------------------
    # Training and Evaluation
    # -------------------------
    def train_models(self, X, y, task_name="fare"):
        """Train all models with simplified overfitting detection"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = self.initialize_models()
        metrics = {}
        
        for name, model in models.items():
            print(f"Training {name} for {task_name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Train metrics
            train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            
            # Test metrics
            test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(
                    model, X, y,
                    scoring="neg_mean_squared_error",
                    cv=5,
                    n_jobs=-1
                )
                cv_rmse = np.sqrt(-cv_scores)
                cv_rmse_mean = float(np.mean(cv_rmse))
                cv_rmse_std = float(np.std(cv_rmse))
            except Exception:
                cv_rmse_mean = float("nan")
                cv_rmse_std = float("nan")
            
            # Calculate R² gap and selection score
            r2_gap = train_r2 - test_r2
            selection_score = self.calculate_selection_score(test_r2, r2_gap)
            
            metrics[name] = {
                "train_rmse": float(train_rmse),
                "test_rmse": float(test_rmse),
                "train_mae": float(train_mae),
                "test_mae": float(test_mae),
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "cv_rmse_mean": cv_rmse_mean,
                "cv_rmse_std": cv_rmse_std,
                "r2_gap": float(abs(r2_gap)),
                "selection_score": float(selection_score)
            }
        
        # Select best model based on selection score
        best_name = max(metrics.items(), key=lambda x: x[1]["selection_score"])[0]
        best_model = models[best_name]
        
        print(f"\n✅ Best {task_name} model: {best_name}")
        print(f"   Test R²: {metrics[best_name]['test_r2']:.4f}")
        print(f"   R² Gap: {metrics[best_name]['r2_gap']:.4f}")
        print(f"   Selection Score: {metrics[best_name]['selection_score']:.4f}")
        
        return models, metrics, best_name, best_model

    # -------------------------
    # Save Results
    # -------------------------
    def save_results(self, fare_models, demand_models, fare_metrics, demand_metrics, 
                     best_fare_name, best_demand_name):
        """Save models and metrics"""
        os.makedirs("ml_models", exist_ok=True)
        
        # Save models
        joblib.dump(fare_models[best_fare_name], "ml_models/best_fare_model.pkl")
        joblib.dump(demand_models[best_demand_name], "ml_models/best_demand_model.pkl")
        joblib.dump(self.fare_scaler, "ml_models/fare_scaler.pkl")
        joblib.dump(self.demand_scaler, "ml_models/demand_scaler.pkl")
        
        # Save metrics
        all_metrics = {
            "fare_prediction": fare_metrics,
            "demand_prediction": demand_metrics
        }
        
        with open("ml_models/metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4)
        
        # Save CSV for easier viewing
        fare_rows = []
        for model_name, m in fare_metrics.items():
            row = {"task": "fare", "model": model_name}
            row.update(m)
            fare_rows.append(row)
        
        demand_rows = []
        for model_name, m in demand_metrics.items():
            row = {"task": "demand", "model": model_name}
            row.update(m)
            demand_rows.append(row)
        
        df_metrics = pd.DataFrame(fare_rows + demand_rows)
        df_metrics.to_csv("ml_models/metrics.csv", index=False)
        
        # Save best model info
        best_info = {
            "timestamp": str(datetime.now()),
            "fare_model": {
                "name": best_fare_name,
                "metrics": fare_metrics[best_fare_name]
            },
            "demand_model": {
                "name": best_demand_name,
                "metrics": demand_metrics[best_demand_name]
            }
        }
        
        with open("ml_models/best_models_info.json", "w") as f:
            json.dump(best_info, f, indent=4)
        
        print(f"\n✅ Models saved:")
        print(f"   - Best Fare Model: {best_fare_name}")
        print(f"   - Best Demand Model: {best_demand_name}")

    # -------------------------
    # Main Training Pipeline
    # -------------------------
    def run_full_pipeline(self):
        """Run complete training pipeline for both tasks"""
        print("=" * 60)
        print("FARE PREDICTION TRAINING")
        print("=" * 60)
        
        # Fare prediction
        fare_df = self.load_fare_data()
        if fare_df is None or fare_df.empty:
            print("❌ No fare data loaded")
            return False
        
        print(f"Loaded {len(fare_df):,} fare records")
        
        fare_df = self.engineer_fare_features(fare_df)
        X_fare, y_fare, fare_features = self.prepare_fare_features(fare_df)
        
        fare_models, fare_metrics, best_fare_name, best_fare_model = self.train_models(
            X_fare, y_fare, "fare"
        )
        
        print("\n" + "=" * 60)
        print("DEMAND PREDICTION TRAINING")
        print("=" * 60)
        
        # Demand prediction
        demand_df = self.load_demand_data()
        if demand_df is None or demand_df.empty:
            print("❌ No demand data loaded")
            return False
        
        print(f"Loaded {len(demand_df):,} demand records")
        
        demand_df = self.engineer_demand_features(demand_df)
        X_demand, y_demand, demand_features = self.prepare_demand_features(demand_df)
        
        demand_models, demand_metrics, best_demand_name, best_demand_model = self.train_models(
            X_demand, y_demand, "demand"
        )
        
        # Save everything
        self.save_results(
            fare_models, demand_models,
            fare_metrics, demand_metrics,
            best_fare_name, best_demand_name
        )
        
        return True


# -------------------------
# CLI Execution
# -------------------------
if __name__ == "__main__":
    db_config = {
        "host": "localhost",
        "database": "rides_warehouse",
        "user": "postgres",
        "password": "sqlbaz2003"
    }
    
    pipeline = MLTrainingPipeline(db_config)
    success = pipeline.run_full_pipeline()
    
    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed!")