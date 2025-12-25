"""
pages/analytics_dashboard.py
Analytics Dashboard answering all business questions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime

def get_postgres_connection():
    """Create PostgreSQL connection to DWH"""
    try:
        conn = psycopg2.connect(
            host=st.secrets.get("postgres_host", "localhost"),
            database=st.secrets.get("postgres_db", "rides_dwh"),
            user=st.secrets.get("postgres_user", "postgres"),
            password=st.secrets.get("postgres_password", "")
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

@st.cache_data(ttl=300)
def load_fact_rides():
    """Load fact_rides with dimension joins"""
    conn = get_postgres_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
        SELECT 
            fr.ride_id, fr.customer_id, fr.driver_id, fr.vehicle_id,
            fr.pickup_zone_id, fr.dropoff_zone_id, fr.ride_date,
            fr.fare_amount, fr.distance_km, fr.duration_minutes,
            dz1.center_lat as pickup_lat, dz1.center_lon as pickup_lon,
            dz2.center_lat as dropoff_lat, dz2.center_lon as dropoff_lon,
            dv.make, dv.model, dv.year,
            dp.method_name as payment_method
        FROM fact_rides fr
        LEFT JOIN dim_zones dz1 ON fr.pickup_zone_id = dz1.zone_id
        LEFT JOIN dim_zones dz2 ON fr.dropoff_zone_id = dz2.zone_id
        LEFT JOIN dim_vehicles dv ON fr.vehicle_id = dv.vehicle_id
        LEFT JOIN dim_payment_methods dp ON fr.payment_method_id = dp.payment_method_id
        WHERE fr.ride_date IS NOT NULL
    """
    
    try:
        df = pd.read_sql(query, conn)
        df['ride_date'] = pd.to_datetime(df['ride_date'])
        df['hour'] = df['ride_date'].dt.hour
        df['day_of_week'] = df['ride_date'].dt.day_name()
        df['month'] = df['ride_date'].dt.month_name()
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_weather_data():
    """Load weather data with zone information"""
    conn = get_postgres_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
        SELECT 
            dw.zone_id, dw.date, dw.temperature_2m, dw.precipitation,
            dw.rain, dw.snowfall, dw.avg_wind_speed,
            dz.center_lat, dz.center_lon
        FROM dim_weather dw
        JOIN dim_zones dz ON dw.zone_id = dz.zone_id
        WHERE dw.date IS NOT NULL
    """
    
    try:
        df = pd.read_sql(query, conn)
        df['date'] = pd.to_datetime(df['date'])
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading weather: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_support_tickets():
    """Load support tickets data"""
    conn = get_postgres_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
        SELECT 
            fst.ticket_id, fst.customer_id, fst.agent_id,
            fst.issue_type, fst.priority, fst.status,
            fst.opened_at, fst.closed_at, fst.resolution_time_minutes,
            da.agent_name, da.region
        FROM fact_support_tickets fst
        LEFT JOIN dim_agents da ON fst.agent_id = da.agent_id
    """
    
    try:
        df = pd.read_sql(query, conn)
        df['opened_at'] = pd.to_datetime(df['opened_at'])
        df['closed_at'] = pd.to_datetime(df['closed_at'])
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading tickets: {str(e)}")
        return pd.DataFrame()

def show():
    st.title("📊 Analytics Dashboard")
    st.markdown("Comprehensive analytics answering key business questions")
    
    # Load data
    with st.spinner("Loading data from Data Warehouse..."):
        rides_df = load_fact_rides()
        weather_df = load_weather_data()
        tickets_df = load_support_tickets()
    
    if rides_df.empty:
        st.warning("No ride data available. Please run the ETL pipeline first.")
        return
    
    # Create tabs for different analysis categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚕 Rides & Demand",
        "🌧️ Weather Impact",
        "💬 Customer Support",
        "📈 Revenue Analysis"
    ])
    
    # ===== TAB 1: Rides & Demand Analysis =====
    with tab1:
        st.header("Rides & Demand Analysis")
        
        # Question 1: Most Popular Zones
        st.subheader("1️⃣ Most Popular Pickup & Dropoff Zones")
        col1, col2 = st.columns(2)
        
        with col1:
            pickup_counts = rides_df['pickup_zone_id'].value_counts().head(10)
            fig = px.bar(
                x=pickup_counts.values,
                y=pickup_counts.index.astype(str),
                orientation='h',
                labels={'x': 'Number of Rides', 'y': 'Zone ID'},
                title='Top 10 Pickup Zones'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            dropoff_counts = rides_df['dropoff_zone_id'].value_counts().head(10)
            fig = px.bar(
                x=dropoff_counts.values,
                y=dropoff_counts.index.astype(str),
                orientation='h',
                labels={'x': 'Number of Rides', 'y': 'Zone ID'},
                title='Top 10 Dropoff Zones'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Question 2: Demand by Time
        st.subheader("2️⃣ Ride Demand by Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_demand = rides_df.groupby('day_of_week').size().reindex(day_order).reset_index(name='ride_count')
        fig = px.bar(
            daily_demand, x='day_of_week', y='ride_count',
            labels={'day_of_week': 'Day of Week', 'ride_count': 'Number of Rides'},
            title='Ride Demand by Day of Week'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Question 3: Average Fare by Zone
        st.subheader("3️⃣ Average Fare Amount by Zone")
        
        zone_fare = rides_df.groupby('pickup_zone_id').agg({
            'fare_amount': 'mean',
            'ride_id': 'count'
        }).reset_index()
        zone_fare.columns = ['zone_id', 'avg_fare', 'ride_count']
        zone_fare = zone_fare[zone_fare['ride_count'] >= 5].sort_values('avg_fare', ascending=False).head(15)
        
        fig = px.bar(
            zone_fare, x='zone_id', y='avg_fare',
            hover_data=['ride_count'],
            labels={'zone_id': 'Zone ID', 'avg_fare': 'Average Fare ($)', 'ride_count': 'Rides'},
            title='Average Fare by Pickup Zone (Min 5 rides)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Question 4: Trip Metrics
        st.subheader("4️⃣ Average Trip Distance & Duration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Distance", f"{rides_df['distance_km'].mean():.2f} km")
        with col2:
            st.metric("Avg Duration", f"{rides_df['duration_minutes'].mean():.1f} min")
        with col3:
            st.metric("Avg Fare", f"${rides_df['fare_amount'].mean():.2f}")
        
        # Scatter: Distance vs Duration
        fig = px.scatter(
            rides_df.sample(min(1000, len(rides_df))),
            x='distance_km', y='duration_minutes',
            color='fare_amount',
            labels={'distance_km': 'Distance (km)', 'duration_minutes': 'Duration (min)'},
            title='Trip Distance vs Duration (Sample)',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Question 5: Revenue by Vehicle Type
        st.subheader("5️⃣ Revenue Analysis by Vehicle Type")
        
        vehicle_revenue = rides_df.groupby(['make', 'model']).agg({
            'fare_amount': 'sum',
            'ride_id': 'count'
        }).reset_index().sort_values('fare_amount', ascending=False).head(10)
        vehicle_revenue.columns = ['Make', 'Model', 'Total Revenue', 'Rides']
        
        fig = px.bar(
            vehicle_revenue,
            x='Total Revenue',
            y=vehicle_revenue['Make'] + ' ' + vehicle_revenue['Model'],
            orientation='h',
            hover_data=['Rides'],
            title='Top 10 Vehicle Models by Revenue'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 2: Weather Impact =====
    with tab2:
        st.header("Weather Impact Analysis")
        
        if weather_df.empty:
            st.warning("No weather data available")
        else:
            # Create compatible date columns for merging
            rides_df['ride_date_only'] = rides_df['ride_date'].dt.date
            weather_df['date_only'] = weather_df['date'].dt.date
            
            # Merge rides with weather
            rides_with_weather = rides_df.merge(
                weather_df[['zone_id', 'date_only', 'temperature_2m', 'precipitation', 'rain', 'snowfall']],
                left_on=['pickup_zone_id', 'ride_date_only'],
                right_on=['zone_id', 'date_only'],
                how='left'
            )
            
            # Drop temporary columns
            rides_with_weather = rides_with_weather.drop(['ride_date_only', 'date_only'], axis=1, errors='ignore')
            
            # Question 6: Weather Conditions Impact
            st.subheader("6️⃣ How Weather Affects Ride Demand")
            
            # Create weather categories
            rides_with_weather['weather_condition'] = 'Clear'
            rides_with_weather.loc[rides_with_weather['rain'] > 0, 'weather_condition'] = 'Rainy'
            rides_with_weather.loc[rides_with_weather['snowfall'] > 0, 'weather_condition'] = 'Snowy'
            
            # Handle NaN values in weather condition
            rides_with_weather['weather_condition'] = rides_with_weather['weather_condition'].fillna('Unknown')
            
            weather_demand = rides_with_weather.groupby('weather_condition').size().reset_index(name='rides')
            
            fig = px.pie(
                weather_demand, values='rides', names='weather_condition',
                title='Ride Distribution by Weather Condition'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Temperature vs Ride Count
            st.subheader("Temperature Impact on Demand")
            
            # Filter out NaN temperatures
            temp_data = rides_with_weather.dropna(subset=['temperature_2m'])
            
            if not temp_data.empty:
                temp_demand = temp_data.groupby(
                    pd.cut(temp_data['temperature_2m'], bins=10)
                ).size().reset_index(name='rides')
                temp_demand['temp_range'] = temp_demand['temperature_2m'].astype(str)
                
                fig = px.bar(
                    temp_demand, x='temp_range', y='rides',
                    labels={'temp_range': 'Temperature Range (°C)', 'rides': 'Number of Rides'},
                    title='Ride Demand by Temperature'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Question 7: Weather Impact on Trip Duration
            st.subheader("7️⃣ Weather Impact on Trip Duration")
            
            weather_metrics = rides_with_weather.groupby('weather_condition').agg({
                'duration_minutes': 'mean',
                'distance_km': 'mean',
                'fare_amount': 'mean'
            }).reset_index()
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Avg Duration', 'Avg Distance', 'Avg Fare')
            )
            
            fig.add_trace(
                go.Bar(x=weather_metrics['weather_condition'], y=weather_metrics['duration_minutes'], name='Duration'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=weather_metrics['weather_condition'], y=weather_metrics['distance_km'], name='Distance'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=weather_metrics['weather_condition'], y=weather_metrics['fare_amount'], name='Fare'),
                row=1, col=3
            )
            
            fig.update_layout(height=400, showlegend=False, title_text="Weather Condition Impacts")
            st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 3: Customer Support =====
    with tab3:
        st.header("Customer Support Analysis")
        
        if tickets_df.empty:
            st.warning("No support ticket data available")
        else:
            # Question 8: Most Common Issue Types
            st.subheader("8️⃣ Most Common Issue Types")
            
            issue_counts = tickets_df['issue_type'].value_counts().reset_index()
            issue_counts.columns = ['Issue Type', 'Count']
            
            fig = px.bar(
                issue_counts, x='Issue Type', y='Count',
                title='Support Tickets by Issue Type',
                color='Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Question 9: Resolution Times
            st.subheader("9️⃣ Average Ticket Resolution Time")
            
            col1, col2 = st.columns(2)
            
            with col1:
                avg_resolution = tickets_df.groupby('issue_type')['resolution_time_minutes'].mean().sort_values(ascending=False)
                fig = px.bar(
                    x=avg_resolution.values / 60,  # Convert to hours
                    y=avg_resolution.index,
                    orientation='h',
                    labels={'x': 'Avg Resolution Time (hours)', 'y': 'Issue Type'},
                    title='Average Resolution Time by Issue Type'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                priority_resolution = tickets_df.groupby('priority')['resolution_time_minutes'].mean().sort_values()
                fig = px.bar(
                    x=priority_resolution.index,
                    y=priority_resolution.values / 60,
                    labels={'x': 'Priority', 'y': 'Avg Resolution Time (hours)'},
                    title='Resolution Time by Priority',
                    color=priority_resolution.values,
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Question 10: Agent Performance
            st.subheader("🔟 Agent Performance Metrics")
            
            agent_stats = tickets_df.groupby('agent_name').agg({
                'ticket_id': 'count',
                'resolution_time_minutes': 'mean'
            }).reset_index().sort_values('ticket_id', ascending=False).head(10)
            agent_stats.columns = ['Agent', 'Tickets Handled', 'Avg Resolution (min)']
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Tickets Handled', 'Avg Resolution Time')
            )
            
            fig.add_trace(
                go.Bar(x=agent_stats['Agent'], y=agent_stats['Tickets Handled'], name='Tickets'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=agent_stats['Agent'], y=agent_stats['Avg Resolution (min)'], name='Resolution Time'),
                row=1, col=2
            )
            
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400, showlegend=False, title_text="Top 10 Agents Performance")
            st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 4: Revenue Analysis =====
    with tab4:
        st.header("Revenue Analysis")
        
        # Overall Revenue Metrics
        st.subheader("Overall Revenue Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", f"${rides_df['fare_amount'].sum():,.2f}")
        with col2:
            st.metric("Total Rides", f"{len(rides_df):,}")
        with col3:
            st.metric("Avg Revenue per Ride", f"${rides_df['fare_amount'].mean():.2f}")
        with col4:
            st.metric("Total Distance", f"{rides_df['distance_km'].sum():,.0f} km")
        
        # Revenue by Driver
        st.subheader("Revenue by Driver")
        
        driver_revenue = rides_df.groupby('driver_id').agg({
            'fare_amount': ['sum', 'count', 'mean']
        }).reset_index()
        driver_revenue.columns = ['driver_id', 'total_revenue', 'rides', 'avg_fare']
        driver_revenue = driver_revenue.sort_values('total_revenue', ascending=False).head(15)
        
        fig = px.bar(
            driver_revenue, x='driver_id', y='total_revenue',
            hover_data=['rides', 'avg_fare'],
            labels={'driver_id': 'Driver ID', 'total_revenue': 'Total Revenue ($)'},
            title='Top 15 Drivers by Revenue'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Revenue Trends Over Time
        st.subheader("Revenue Trends Over Time")
        
        daily_revenue = rides_df.groupby(rides_df['ride_date'].dt.date).agg({
            'fare_amount': 'sum',
            'ride_id': 'count'
        }).reset_index()
        daily_revenue.columns = ['date', 'revenue', 'rides']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=daily_revenue['date'], y=daily_revenue['revenue'], 
                      name='Revenue', mode='lines+markers'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=daily_revenue['date'], y=daily_revenue['rides'],
                      name='Rides', mode='lines+markers', line=dict(dash='dash')),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Rides", secondary_y=True)
        fig.update_layout(title_text="Daily Revenue and Ride Count Trends", height=500)
        
        st.plotly_chart(fig, use_container_width=True)