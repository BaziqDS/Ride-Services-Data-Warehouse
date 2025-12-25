"""
Ride Sharing Data Warehouse & BI Application
Main Streamlit Entry Point
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Ride Sharing DWH & BI Platform",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🚕 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Module:",
    [
        "🏠 Home",
        "📝 Data Insertion",
        "📊 Analytics Dashboard",
        "🤖 ML Predictions",
        "⚙️ ETL Pipeline Control"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project:** Ride Sharing DWH  
**Version:** 1.0  
**Built with:** Streamlit + Apache Airflow
""")

# Main content area
if page == "🏠 Home":
    st.markdown('<p class="main-header">🚕 Ride Sharing Data Warehouse & BI Platform</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Ride Sharing Analytics Platform
    
    This integrated platform provides comprehensive data management, analytics, and machine learning capabilities 
    for ride-sharing operations.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📝 Data Insertion
        - Insert single ride records
        - Batch upload via CSV
        - Add support tickets
        - Manage customer & driver data
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Analytics Dashboard
        - Ride demand analysis
        - Revenue metrics
        - Weather impact analysis
        - Support ticket insights
        """)
    
    with col3:
        st.markdown("""
        #### 🤖 ML Predictions
        - Demand forecasting
        - Customer segmentation
        - High-risk zone identification
        - Trip duration prediction
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Business Questions Answered
    
    This platform addresses key business questions including:
    
    **Rides & Demand Analysis:**
    - Most popular pickup and dropoff zones
    - Demand patterns by time, day, and month
    - Average fare and tip by zone
    - Revenue per driver and vehicle type
    
    **Weather & Operations:**
    - Weather impact on ride demand
    - Correlation between weather and trip duration
    
    **Customer Experience:**
    - Most common issue types and their locations
    - Ticket resolution times
    - Driver complaint patterns
    
    **Predictive Analytics:**
    - Ride demand forecasting by zone and hour
    - Customer complaint behavior clustering
    - High-risk zone identification
    """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate between different modules")

elif page == "📝 Data Insertion":
    from pages import data_insertion
    data_insertion.show()

elif page == "📊 Analytics Dashboard":
    from pages import analytics_dashboard
    analytics_dashboard.show()

elif page == "🤖 ML Predictions":
    from pages import ml_predictions
    ml_predictions.show()

elif page == "⚙️ ETL Pipeline Control":
    from pages import pipeline_control
    pipeline_control.show()