"""
pages/ml_predictions.py
Simplified ML Interface - Based on R² and R² Gap
Two tabs: Training & Performance, Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from comprehensive_ml_training import MLTrainingPipeline
except Exception:
    MLTrainingPipeline = None


# -------------------------
# Helper Functions
# -------------------------
def load_models_and_metrics():
    """Load saved models, scalers, and metrics"""
    try:
        fare_model = joblib.load("ml_models/best_fare_model.pkl")
        demand_model = joblib.load("ml_models/best_demand_model.pkl")
        fare_scaler = joblib.load("ml_models/fare_scaler.pkl")
        demand_scaler = joblib.load("ml_models/demand_scaler.pkl")
        
        with open("ml_models/metrics.json", "r") as f:
            metrics = json.load(f)
        
        with open("ml_models/best_models_info.json", "r") as f:
            best_info = json.load(f)
            
        return fare_model, demand_model, fare_scaler, demand_scaler, metrics, best_info
    except Exception as e:
        return None, None, None, None, None, None


def create_performance_chart(metrics_dict, task_name):
    """Create performance comparison chart with selection score"""
    models = list(metrics_dict.keys())
    test_r2 = [m["test_r2"] for m in metrics_dict.values()]
    
    # Calculate r2_gap and selection_score if not present (backward compatibility)
    r2_gap = []
    selection_score = []
    for m in metrics_dict.values():
        gap = m.get('r2_gap', m['train_r2'] - m['test_r2'])
        r2_gap.append(gap)
        
        if 'selection_score' in m:
            sel_score = m['selection_score']
        else:
            sel_score = m['test_r2'] - (2.0 * gap)
        selection_score.append(sel_score)
    
    # Find best model (highest selection score)
    best_idx = selection_score.index(max(selection_score))
    colors = ['#2ecc71' if i == best_idx else '#3498db' for i in range(len(models))]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f'{task_name} - Test R² Score',
            f'{task_name} - R² Gap',
            f'{task_name} - Selection Score'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # Test R²
    fig.add_trace(
        go.Bar(
            x=models, y=test_r2, 
            name='Test R²', 
            marker_color=colors,
            text=[f"{v:.4f}" for v in test_r2], 
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # R² Gap
    fig.add_trace(
        go.Bar(
            x=models, y=r2_gap, 
            name='R² Gap', 
            marker_color=colors,
            text=[f"{v:.4f}" for v in r2_gap], 
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # Selection Score
    fig.add_trace(
        go.Bar(
            x=models, y=selection_score, 
            name='Selection Score', 
            marker_color=colors,
            text=[f"{v:.4f}" for v in selection_score], 
            textposition='outside'
        ),
        row=1, col=3
    )
    
    fig.update_yaxes(title_text="R² Score", row=1, col=1)
    fig.update_yaxes(title_text="Gap", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=3)
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def show():
    st.title("🤖 Machine Learning Models")
    
    # Create two main tabs
    tab1, tab2 = st.tabs(["🏋️ Training & Performance", "🔮 Predictions"])
    
    # =========================================================================
    # TAB 1: TRAINING & PERFORMANCE
    # =========================================================================
    with tab1:
        st.header("Model Training & Performance Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            This system trains **two separate models**:
            1. **Fare Prediction** - Predicts ride fare based on distance, duration, zones, weather, etc.
            2. **Demand Prediction** - Predicts number of rides per zone per hour
            
            **Model Selection Criteria:**
            - **Test R² Score**: Measures model accuracy on unseen data (higher is better)
            - **R² Gap**: Difference between training and test R² (lower is better)
            - **Selection Score**: Combined metric that balances accuracy and generalization
            
            Formula: `Selection Score = Test R² - (2.0 × R² Gap)`
            
            The model with the **highest selection score** is chosen as the best model.
            """)
        
        with col2:
            if MLTrainingPipeline is None:
                st.error("Training module not found")
            else:
                if st.button("🚀 Start Training", type="primary", use_container_width=True):
                    with st.spinner("Training models... This may take a few minutes"):
                        try:
                            db_config = {
                                "host": st.secrets.get("postgres_host", "localhost"),
                                "database": st.secrets.get("postgres_db", "rides_dwh"),
                                "user": st.secrets.get("postgres_user", "postgres"),
                                "password": st.secrets.get("postgres_password", "")
                            }
                            
                            pipeline = MLTrainingPipeline(db_config)
                            
                            # Progress updates
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Loading data...")
                            progress_bar.progress(20)
                            
                            success = pipeline.run_full_pipeline()
                            
                            progress_bar.progress(100)
                            
                            if success:
                                st.success("✅ Training completed! Models saved successfully.")
                                st.rerun()
                            else:
                                st.error("❌ Training failed. Check console for errors.")
                                
                        except Exception as e:
                            st.error(f"Training error: {str(e)}")
                            st.exception(e)
        
        st.markdown("---")
        
        # Load and display metrics
        _, _, _, _, metrics, best_info = load_models_and_metrics()
        
        if metrics is None:
            st.info("👆 No trained models found. Click 'Start Training' to begin.")
            return
        
        # Display best models info
        st.subheader("🏆 Best Models Selected")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💵 Fare Prediction")
            if best_info:
                fare_info = best_info["fare_model"]
                st.success(f"**Model:** {fare_info['name']}")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Test R²", f"{fare_info['metrics']['test_r2']:.4f}")
                with m2:
                    r2_gap = fare_info['metrics'].get('r2_gap', 
                        fare_info['metrics']['train_r2'] - fare_info['metrics']['test_r2'])
                    st.metric("R² Gap", f"{r2_gap:.4f}")
                with m3:
                    # Calculate selection score if not present
                    if 'selection_score' in fare_info['metrics']:
                        sel_score = fare_info['metrics']['selection_score']
                    else:
                        test_r2 = fare_info['metrics']['test_r2']
                        sel_score = test_r2 - (2.0 * r2_gap)
                    st.metric("Selection Score", f"{sel_score:.4f}")
                
                st.metric("Test RMSE", f"${fare_info['metrics']['test_rmse']:.2f}")
        
        with col2:
            st.markdown("### 📊 Demand Prediction")
            if best_info:
                demand_info = best_info["demand_model"]
                st.success(f"**Model:** {demand_info['name']}")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Test R²", f"{demand_info['metrics']['test_r2']:.4f}")
                with m2:
                    r2_gap = demand_info['metrics'].get('r2_gap', 
                        demand_info['metrics']['train_r2'] - demand_info['metrics']['test_r2'])
                    st.metric("R² Gap", f"{r2_gap:.4f}")
                with m3:
                    # Calculate selection score if not present
                    if 'selection_score' in demand_info['metrics']:
                        sel_score = demand_info['metrics']['selection_score']
                    else:
                        test_r2 = demand_info['metrics']['test_r2']
                        sel_score = test_r2 - (2.0 * r2_gap)
                    st.metric("Selection Score", f"{sel_score:.4f}")
                
                st.metric("Test RMSE", f"{demand_info['metrics']['test_rmse']:.2f} rides")
        
        st.markdown("---")
        
        # Detailed metrics comparison
        st.subheader("📈 All Models Performance Comparison")
        
        # Fare models comparison
        st.markdown("### 💵 Fare Prediction Models")
        fare_metrics = metrics.get("fare_prediction", {})
        
        if fare_metrics:
            # Create chart
            fig = create_performance_chart(fare_metrics, "Fare Prediction")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            with st.expander("📋 Detailed Fare Model Metrics"):
                fare_df = pd.DataFrame(fare_metrics).T
                
                # Reorder columns for better readability
                cols_order = [
                    'selection_score', 'test_r2', 'r2_gap',
                    'train_r2', 'test_rmse', 'train_rmse',
                    'test_mae', 'train_mae',
                    'cv_rmse_mean', 'cv_rmse_std'
                ]
                fare_df = fare_df[[col for col in cols_order if col in fare_df.columns]]
                
                # Highlight best model
                best_model = fare_df['selection_score'].idxmax()
                
                # Format and display
                st.dataframe(
                    fare_df.style.format({
                        'selection_score': '{:.4f}',
                        'test_r2': '{:.4f}',
                        'train_r2': '{:.4f}',
                        'r2_gap': '{:.4f}',
                        'test_rmse': '${:.2f}',
                        'train_rmse': '${:.2f}',
                        'test_mae': '${:.2f}',
                        'train_mae': '${:.2f}',
                        'cv_rmse_mean': '${:.2f}',
                        'cv_rmse_std': '${:.2f}'
                    }).highlight_max(subset=['selection_score', 'test_r2'], color='lightgreen')
                      .highlight_min(subset=['r2_gap'], color='lightgreen'),
                    use_container_width=True
                )
                
                st.info(f"✅ **Best Model:** {best_model} (highest selection score)")
        
        st.markdown("---")
        
        # Demand models comparison
        st.markdown("### 📊 Demand Prediction Models")
        demand_metrics = metrics.get("demand_prediction", {})
        
        if demand_metrics:
            # Create chart
            fig = create_performance_chart(demand_metrics, "Demand Prediction")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            with st.expander("📋 Detailed Demand Model Metrics"):
                demand_df = pd.DataFrame(demand_metrics).T
                
                # Reorder columns
                cols_order = [
                    'selection_score', 'test_r2', 'r2_gap',
                    'train_r2', 'test_rmse', 'train_rmse',
                    'test_mae', 'train_mae',
                    'cv_rmse_mean', 'cv_rmse_std'
                ]
                demand_df = demand_df[[col for col in cols_order if col in demand_df.columns]]
                
                # Highlight best model
                best_model = demand_df['selection_score'].idxmax()
                
                # Format and display
                st.dataframe(
                    demand_df.style.format({
                        'selection_score': '{:.4f}',
                        'test_r2': '{:.4f}',
                        'train_r2': '{:.4f}',
                        'r2_gap': '{:.4f}',
                        'test_rmse': '{:.2f}',
                        'train_rmse': '{:.2f}',
                        'test_mae': '{:.2f}',
                        'train_mae': '{:.2f}',
                        'cv_rmse_mean': '{:.2f}',
                        'cv_rmse_std': '{:.2f}'
                    }).highlight_max(subset=['selection_score', 'test_r2'], color='lightgreen')
                      .highlight_min(subset=['r2_gap'], color='lightgreen'),
                    use_container_width=True
                )
                
                st.info(f"✅ **Best Model:** {best_model} (highest selection score)")
        
        # Model selection explanation
        with st.expander("ℹ️ Understanding Model Selection"):
            st.markdown("""
            ### How Models Are Selected
            
            The system uses a **selection score** that balances two key metrics:
            
            1. **Test R² Score** (accuracy on unseen data)
               - Measures how well the model predicts new data
               - Range: 0 to 1 (higher is better)
               - 0.8+ is considered excellent
            
            2. **R² Gap** (overfitting indicator)
               - Difference between training and test R²
               - Lower gap = better generalization
               - Gap > 0.15 may indicate overfitting
            
            ### Selection Score Formula
            ```
            Selection Score = Test R² - (2.0 × R² Gap)
            ```
            
            This formula:
            - Rewards high test accuracy
            - Penalizes large gaps (overfitting)
            - Balances performance vs generalization
            
            ### Example
            - **Model A**: Test R² = 0.85, Gap = 0.05 → Score = 0.85 - 0.10 = **0.75**
            - **Model B**: Test R² = 0.90, Gap = 0.20 → Score = 0.90 - 0.40 = **0.50**
            
            Model A wins despite lower R² because it generalizes better!
            """)
    
    # =========================================================================
    # TAB 2: PREDICTIONS
    # =========================================================================
    with tab2:
        st.header("Make Predictions")
        
        fare_model, demand_model, fare_scaler, demand_scaler, metrics, best_info = load_models_and_metrics()
        
        if fare_model is None or demand_model is None:
            st.warning("⚠️ No trained models found. Please train models first in the 'Training & Performance' tab.")
            return
        
        # Create two prediction sections
        pred_tab1, pred_tab2 = st.tabs(["💵 Fare Prediction", "📊 Demand Prediction"])
        
        # ---------------------------------------------------------------------
        # FARE PREDICTION
        # ---------------------------------------------------------------------
        with pred_tab1:
            st.subheader("Predict Ride Fare")
            
            with st.form("fare_prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Trip Details**")
                    distance_km = st.number_input("Distance (km)", 0.1, 100.0, 5.0, 0.5)
                    duration_minutes = st.number_input("Duration (minutes)", 1, 300, 20, 1)
                    pickup_zone = st.number_input("Pickup Zone ID", 1, 9999, 100)
                    dropoff_zone = st.number_input("Dropoff Zone ID", 1, 9999, 200)
                
                with col2:
                    st.markdown("**Customer & Driver**")
                    customer_age_days = st.number_input("Customer Age (days)", 0, 3650, 365)
                    driver_exp_days = st.number_input("Driver Experience (days)", 0, 3650, 730)
                    vehicle_age = st.number_input("Vehicle Age (years)", 0, 30, 3)
                
                with col3:
                    st.markdown("**Time**")
                    hour = st.slider("Hour of Day", 0, 23, 12)
                    day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 3)
                    month = st.slider("Month", 1, 12, 6)
                
                st.markdown("**Weather Conditions**")
                w1, w2, w3, w4 = st.columns(4)
                with w1:
                    temperature = st.number_input("Temperature (°C)", -20.0, 45.0, 20.0)
                with w2:
                    precipitation = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0)
                with w3:
                    cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 50)
                with w4:
                    wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)
                
                predict_fare_btn = st.form_submit_button("🎯 Predict Fare", type="primary", use_container_width=True)
            
            if predict_fare_btn:
                try:
                    # Calculate derived features
                    avg_speed_kmh = distance_km / (duration_minutes / 60 + 1e-6)
                    zone_distance = abs(pickup_zone - dropoff_zone)
                    weather_severity = precipitation
                    is_weekend = 1 if day_of_week in [5, 6] else 0
                    is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
                    is_night = 1 if (hour <= 5 or hour >= 22) else 0
                    
                    # Create feature vector (20 features)
                    features = np.array([[
                        distance_km, duration_minutes, avg_speed_kmh,
                        pickup_zone, dropoff_zone, zone_distance,
                        customer_age_days, driver_exp_days, vehicle_age,
                        temperature, precipitation, weather_severity,
                        cloud_cover, wind_speed,
                        hour, day_of_week, month, is_weekend, is_rush_hour, is_night
                    ]])
                    
                    # Scale and predict
                    X_scaled = fare_scaler.transform(features)
                    predicted_fare = fare_model.predict(X_scaled)[0]
                    
                    # Display result
                    st.success(f"### Predicted Fare: ${predicted_fare:.2f}")
                    
                    # Show model information
                    if best_info:
                        fare_metrics = best_info["fare_model"]["metrics"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", best_info['fare_model']['name'])
                        with col2:
                            st.metric("Test R²", f"{fare_metrics['test_r2']:.4f}")
                        with col3:
                            st.metric("Typical Error", f"±${fare_metrics['test_rmse']:.2f}")
                    
                    # Show feature importance (simplified)
                    with st.expander("📊 Trip Details"):
                        factors_df = pd.DataFrame({
                            'Factor': ['Distance', 'Duration', 'Average Speed', 'Zone Distance', 'Weather', 'Time Period'],
                            'Value': [
                                f"{distance_km:.1f} km",
                                f"{duration_minutes} min",
                                f"{avg_speed_kmh:.1f} km/h",
                                f"{zone_distance}",
                                f"{weather_severity:.1f} mm precipitation",
                                f"{'Weekend' if is_weekend else 'Weekday'}, {'Rush Hour' if is_rush_hour else 'Normal'}"
                            ]
                        })
                        st.table(factors_df)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.exception(e)
        
        # ---------------------------------------------------------------------
        # DEMAND PREDICTION
        # ---------------------------------------------------------------------
        with pred_tab2:
            st.subheader("Predict Zone Demand")
            
            with st.form("demand_prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Zone Details**")
                    zone_id = st.number_input("Zone ID", 1, 9999, 100)
                    center_lat = st.number_input("Center Latitude", 0.0, 90.0, 40.7128, format="%.4f")
                    center_lon = st.number_input("Center Longitude", -180.0, 0.0, -74.0060, format="%.4f")
                
                with col2:
                    st.markdown("**Time**")
                    hour_demand = st.slider("Hour of Day", 0, 23, 12, key="demand_hour")
                    day_of_week_demand = st.slider("Day of Week (0=Mon)", 0, 6, 3, key="demand_dow")
                    month_demand = st.slider("Month", 1, 12, 6, key="demand_month")
                
                st.markdown("**Weather Conditions**")
                w1, w2, w3, w4 = st.columns(4)
                with w1:
                    temp_demand = st.number_input("Temperature (°C)", -20.0, 45.0, 20.0, key="demand_temp")
                with w2:
                    precip_demand = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0, key="demand_precip")
                with w3:
                    cloud_demand = st.slider("Cloud Cover (%)", 0, 100, 50, key="demand_cloud")
                with w4:
                    wind_demand = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0, key="demand_wind")
                
                predict_demand_btn = st.form_submit_button("🎯 Predict Demand", type="primary", use_container_width=True)
            
            if predict_demand_btn:
                try:
                    # Calculate derived features
                    weather_severity = precip_demand
                    is_weekend = 1 if day_of_week_demand in [5, 6] else 0
                    is_rush_hour = 1 if hour_demand in [7, 8, 9, 17, 18, 19] else 0
                    is_night = 1 if (hour_demand <= 5 or hour_demand >= 22) else 0
                    
                    # Create feature vector (14 features)
                    features = np.array([[
                        zone_id, center_lat, center_lon,
                        temp_demand, precip_demand, weather_severity,
                        cloud_demand, wind_demand,
                        hour_demand, day_of_week_demand, month_demand,
                        is_weekend, is_rush_hour, is_night
                    ]])
                    
                    # Scale and predict
                    X_scaled = demand_scaler.transform(features)
                    predicted_demand = demand_model.predict(X_scaled)[0]
                    
                    # Display result
                    st.success(f"### Predicted Demand: {int(round(predicted_demand))} rides/hour")
                    
                    # Show model information
                    if best_info:
                        demand_metrics = best_info["demand_model"]["metrics"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", best_info['demand_model']['name'])
                        with col2:
                            st.metric("Test R²", f"{demand_metrics['test_r2']:.4f}")
                        with col3:
                            st.metric("Typical Error", f"±{demand_metrics['test_rmse']:.1f} rides")
                    
                    # Demand category
                    if predicted_demand < 5:
                        demand_level = "🟢 Low"
                        color = "green"
                    elif predicted_demand < 15:
                        demand_level = "🟡 Medium"
                        color = "orange"
                    else:
                        demand_level = "🔴 High"
                        color = "red"
                    
                    st.markdown(f"### Demand Level: :{color}[{demand_level}]")
                    
                    # Show factors
                    with st.expander("📊 Zone & Time Details"):
                        factors_df = pd.DataFrame({
                            'Factor': ['Zone', 'Location', 'Time', 'Weather', 'Day Type'],
                            'Value': [
                                f"Zone {zone_id}",
                                f"({center_lat:.4f}, {center_lon:.4f})",
                                f"{hour_demand}:00 {'(Rush Hour)' if is_rush_hour else ''}{'(Night)' if is_night else ''}",
                                f"{temp_demand}°C, {precip_demand}mm precipitation",
                                'Weekend' if is_weekend else 'Weekday'
                            ]
                        })
                        st.table(factors_df)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.exception(e)


if __name__ == "__main__":
    show()