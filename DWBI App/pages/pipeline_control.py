"""
pages/pipeline_control.py
ETL Pipeline Control and Monitoring
"""

import streamlit as st
import requests
from datetime import datetime
def show():
    st.title("⚙️ ETL Pipeline")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("⚠️ Ensure both OLTP databases contain data before triggering the pipeline")
    with col2:
        st.link_button(
            "🚀 Open Airflow Dashboard",
            "http://localhost:8080/dags"
        )
    
    st.markdown("---")
    
    # Pipeline Architecture Diagram
    st.subheader("Pipeline Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    ETL PIPELINE FLOW                         │
    └─────────────────────────────────────────────────────────────┘
    
    1. EXTRACTION PHASE
       ├── extract_rides_task ──────────> MySQL OLTP 1
       ├── extract_review_task ─────────> MySQL OLTP 2
       └── extract_weather_task ────────> Open-Meteo API
                    │
                    ▼
    2. TRANSFORMATION PHASE
       └── transform_task
            ├── Clean & validate data
            ├── Create zones (K-Means clustering)
            ├── Enrich with weather data
            └── Calculate derived metrics
                    │
                    ▼
    3. MODELING PHASE
       └── star_schema_task
            ├── Build dimension tables
            └── Build fact tables
                    │
                    ▼
    4. LOADING PHASE
       └── load_task ────────────────────> PostgreSQL DWH
    ```
    """)
    
    # Connection Settings
    st.markdown("---")
    with st.expander("⚙️ Connection Settings"):
        st.markdown("""
        **Airflow Connection Details:**
        - URL: Configure in `.streamlit/secrets.toml`
        - Username: Configure in `.streamlit/secrets.toml`
        - Password: Configure in `.streamlit/secrets.toml`
        
        **Example secrets.toml:**
        ```toml
        airflow_url = "http://localhost:8080/api/v2"
        airflow_username = "admin"
        airflow_password = "admin"
        
        postgres_host = "localhost"
        postgres_db = "rides_dwh"
        postgres_user = "postgres"
        postgres_password = "your_password"
        
        mysql_host = "localhost"
        mysql_user = "root"
        mysql_password = "your_password"
        ```
        """)