"""
pages/data_insertion.py
Data Insertion Module for OLTP databases
"""

import streamlit as st
import pandas as pd
import mysql.connector
from datetime import datetime, date
import io

def get_mysql_connection(db_name):
    """Create MySQL connection"""
    try:
        conn = mysql.connector.connect(
            host=st.secrets.get("mysql_host", "localhost"),
            user=st.secrets.get("mysql_user", "root"),
            password=st.secrets.get("mysql_password", ""),
            database=db_name
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

def insert_customer(conn, data):
    """Insert customer into ride_mgmt_oltp"""
    cursor = conn.cursor()
    query = """
        INSERT INTO customers (first_name, last_name, email, phone)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(query, (data['first_name'], data['last_name'], data['email'], data['phone']))
    conn.commit()
    return cursor.lastrowid

def insert_driver(conn, data):
    """Insert driver into ride_mgmt_oltp"""
    cursor = conn.cursor()
    query = """
        INSERT INTO drivers (first_name, last_name, license_number, phone, hire_date, is_active)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        data['first_name'], data['last_name'], data['license_number'],
        data['phone'], data['hire_date'], data.get('is_active', True)
    ))
    conn.commit()
    return cursor.lastrowid

def insert_vehicle(conn, data):
    """Insert vehicle into ride_mgmt_oltp"""
    cursor = conn.cursor()
    query = """
        INSERT INTO vehicles (license_plate, make, model, year, color, vehicle_type, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        data['license_plate'], data['make'], data['model'],
        data['year'], data['color'], data['vehicle_type'], data.get('is_active', True)
    ))
    conn.commit()
    return cursor.lastrowid

def insert_ride(conn, data):
    """Insert ride into ride_mgmt_oltp"""
    cursor = conn.cursor()
    query = """
        INSERT INTO rides (
            customer_id, driver_id, vehicle_id, vendor_id,
            pickup_datetime, dropoff_datetime, passenger_count,
            trip_distance, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, fare_amount,
            extra, mta_tax, tip_amount, tolls_amount,
            improvement_surcharge, total_amount, ratecode_id,
            store_and_fwd_flag, payment_method_id, ride_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        data['customer_id'], data['driver_id'], data['vehicle_id'], data.get('vendor_id', 1),
        data['pickup_datetime'], data.get('dropoff_datetime'), data.get('passenger_count', 1),
        data['trip_distance'], data['pickup_longitude'], data['pickup_latitude'],
        data['dropoff_longitude'], data['dropoff_latitude'], data['fare_amount'],
        data.get('extra', 0), data.get('mta_tax', 0), data.get('tip_amount', 0),
        data.get('tolls_amount', 0), data.get('improvement_surcharge', 0),
        data['total_amount'], data.get('ratecode_id', 1),
        data.get('store_and_fwd_flag', 'N'), data['payment_method_id'],
        data.get('ride_status', 'Completed')
    ))
    conn.commit()
    return cursor.lastrowid

def insert_support_ticket(conn, data):
    """Insert support ticket into ride_review_oltp"""
    cursor = conn.cursor()
    query = """
        INSERT INTO support_tickets (
            ride_id, customer_id, driver_id, issue_type,
            description, status, priority, assigned_agent_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        data['ride_id'], data['customer_id'], data.get('driver_id'),
        data['issue_type'], data['description'], data.get('status', 'Open'),
        data.get('priority', 'Medium'), data.get('assigned_agent_id')
    ))
    conn.commit()
    return cursor.lastrowid

def batch_insert_rides(conn, df):
    """Batch insert rides from DataFrame"""
    cursor = conn.cursor()
    query = """
        INSERT INTO rides (
            customer_id, driver_id, vehicle_id, pickup_datetime,
            dropoff_datetime, passenger_count, trip_distance,
            pickup_longitude, pickup_latitude, dropoff_longitude,
            dropoff_latitude, fare_amount, total_amount,
            payment_method_id, ride_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    records = []
    for _, row in df.iterrows():
        records.append((
            int(row['customer_id']), int(row['driver_id']), int(row['vehicle_id']),
            row['pickup_datetime'], row.get('dropoff_datetime'),
            int(row.get('passenger_count', 1)), float(row['trip_distance']),
            float(row['pickup_longitude']), float(row['pickup_latitude']),
            float(row['dropoff_longitude']), float(row['dropoff_latitude']),
            float(row['fare_amount']), float(row['total_amount']),
            int(row['payment_method_id']), row.get('ride_status', 'Completed')
        ))
    
    cursor.executemany(query, records)
    conn.commit()
    return cursor.rowcount

def show():
    st.title("📝 Data Insertion Module")
    st.markdown("Insert data into OLTP databases (single record or batch upload)")
    
    tab1, tab2, tab3 = st.tabs(["Single Insert", "Batch Upload", "Support Tickets"])
    
    # ===== TAB 1: Single Insert =====
    with tab1:
        st.subheader("Insert Single Record")
        
        insert_type = st.selectbox(
            "Select Record Type",
            ["Customer", "Driver", "Vehicle", "Ride"]
        )
        
        if insert_type == "Customer":
            with st.form("customer_form"):
                col1, col2 = st.columns(2)
                with col1:
                    first_name = st.text_input("First Name*")
                    email = st.text_input("Email*")
                with col2:
                    last_name = st.text_input("Last Name*")
                    phone = st.text_input("Phone*")
                
                submit = st.form_submit_button("Insert Customer")
                
                if submit:
                    if all([first_name, last_name, email, phone]):
                        conn = get_mysql_connection("ride_mgmt_oltp")
                        if conn:
                            try:
                                customer_id = insert_customer(conn, {
                                    'first_name': first_name,
                                    'last_name': last_name,
                                    'email': email,
                                    'phone': phone
                                })
                                st.success(f"✅ Customer inserted successfully! ID: {customer_id}")
                                conn.close()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please fill all required fields")
        
        elif insert_type == "Driver":
            with st.form("driver_form"):
                col1, col2 = st.columns(2)
                with col1:
                    first_name = st.text_input("First Name*")
                    license_number = st.text_input("License Number*")
                    hire_date = st.date_input("Hire Date*", value=date.today())
                with col2:
                    last_name = st.text_input("Last Name*")
                    phone = st.text_input("Phone*")
                    is_active = st.checkbox("Active", value=True)
                
                submit = st.form_submit_button("Insert Driver")
                
                if submit:
                    if all([first_name, last_name, license_number, phone]):
                        conn = get_mysql_connection("ride_mgmt_oltp")
                        if conn:
                            try:
                                driver_id = insert_driver(conn, {
                                    'first_name': first_name,
                                    'last_name': last_name,
                                    'license_number': license_number,
                                    'phone': phone,
                                    'hire_date': hire_date,
                                    'is_active': is_active
                                })
                                st.success(f"✅ Driver inserted successfully! ID: {driver_id}")
                                conn.close()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please fill all required fields")
        
        elif insert_type == "Vehicle":
            with st.form("vehicle_form"):
                col1, col2 = st.columns(2)
                with col1:
                    license_plate = st.text_input("License Plate*")
                    make = st.text_input("Make*")
                    year = st.number_input("Year*", min_value=1990, max_value=2025, value=2020)
                    vehicle_type = st.selectbox("Type*", ["Standard", "Premium", "SUV", "Luxury"])
                with col2:
                    model = st.text_input("Model*")
                    color = st.text_input("Color*")
                    is_active = st.checkbox("Active", value=True)
                
                submit = st.form_submit_button("Insert Vehicle")
                
                if submit:
                    if all([license_plate, make, model, color]):
                        conn = get_mysql_connection("ride_mgmt_oltp")
                        if conn:
                            try:
                                vehicle_id = insert_vehicle(conn, {
                                    'license_plate': license_plate,
                                    'make': make,
                                    'model': model,
                                    'year': year,
                                    'color': color,
                                    'vehicle_type': vehicle_type,
                                    'is_active': is_active
                                })
                                st.success(f"✅ Vehicle inserted successfully! ID: {vehicle_id}")
                                conn.close()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please fill all required fields")
        
        elif insert_type == "Ride":
            with st.form("ride_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    customer_id = st.number_input("Customer ID*", min_value=1)
                    driver_id = st.number_input("Driver ID*", min_value=1)
                    vehicle_id = st.number_input("Vehicle ID*", min_value=1)
                
                with col2:
                    pickup_date = st.date_input("Pickup Date*", value=date.today())
                    pickup_time= st.time_input("Pickup Time*", value=datetime.now().time())
                    pickup_lat = st.number_input("Pickup Latitude*", format="%.6f", value=40.758896)
                    pickup_lon = st.number_input("Pickup Longitude*", format="%.6f", value=-73.985130)
                
                with col3:
                    dropoff_lat = st.number_input("Dropoff Latitude*", format="%.6f", value=40.748817)
                    dropoff_lon = st.number_input("Dropoff Longitude*", format="%.6f", value=-73.985428)
                    trip_distance = st.number_input("Distance (miles)*", min_value=0.0, value=2.5, step=0.1)
                
                pickup_datetime = datetime.combine(pickup_date, pickup_time)
                
                col4, col5 = st.columns(2)
                with col4:
                    fare_amount = st.number_input("Fare Amount*", min_value=0.0, value=15.0, step=0.5)
                    tip_amount = st.number_input("Tip Amount", min_value=0.0, value=3.0, step=0.5)
                
                with col5:
                    payment_method_id = st.number_input("Payment Method ID*", min_value=1, value=1)
                    total_amount = st.number_input("Total Amount*", min_value=0.0, value=fare_amount + tip_amount, step=0.5)
                
                submit = st.form_submit_button("Insert Ride")
                
                if submit:
                    conn = get_mysql_connection("ride_mgmt_oltp")
                    if conn:
                        try:
                            ride_id = insert_ride(conn, {
                                'customer_id': int(customer_id),
                                'driver_id': int(driver_id),
                                'vehicle_id': int(vehicle_id),
                                'pickup_datetime': pickup_datetime,
                                'dropoff_datetime': None,
                                'trip_distance': trip_distance,
                                'pickup_longitude': pickup_lon,
                                'pickup_latitude': pickup_lat,
                                'dropoff_longitude': dropoff_lon,
                                'dropoff_latitude': dropoff_lat,
                                'fare_amount': fare_amount,
                                'tip_amount': tip_amount,
                                'total_amount': total_amount,
                                'payment_method_id': int(payment_method_id)
                            })
                            st.success(f"✅ Ride inserted successfully! ID: {ride_id}")
                            conn.close()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    # ===== TAB 2: Batch Upload =====
    with tab2:
        st.subheader("Batch Upload from CSV")
        
        st.markdown("""
        Upload a CSV file with ride data. Required columns:
        - customer_id, driver_id, vehicle_id
        - pickup_datetime, pickup_latitude, pickup_longitude
        - dropoff_latitude, dropoff_longitude
        - trip_distance, fare_amount, total_amount, payment_method_id
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Insert Batch"):
                    conn = get_mysql_connection("ride_mgmt_oltp")
                    if conn:
                        try:
                            count = batch_insert_rides(conn, df)
                            st.success(f"✅ Successfully inserted {count} rides!")
                            conn.close()
                        except Exception as e:
                            st.error(f"Error during batch insert: {str(e)}")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    # ===== TAB 3: Support Tickets =====
    with tab3:
        st.subheader("Insert Support Ticket")
        
        with st.form("ticket_form"):
            col1, col2 = st.columns(2)
            with col1:
                ride_id = st.number_input("Ride ID*", min_value=1)
                customer_id = st.number_input("Customer ID*", min_value=1)
                driver_id = st.number_input("Driver ID (optional)", min_value=0, value=0)
                issue_type = st.selectbox("Issue Type*", ["Driver", "Vehicle", "Fare", "App", "Other"])
            
            with col2:
                status = st.selectbox("Status", ["Open", "In Progress", "Resolved", "Closed"])
                priority = st.selectbox("Priority", ["Low", "Medium", "High"])
                assigned_agent_id = st.number_input("Assigned Agent ID (optional)", min_value=0, value=0)
            
            description = st.text_area("Description*")
            
            submit = st.form_submit_button("Submit Ticket")
            
            if submit:
                if ride_id and customer_id and description:
                    conn = get_mysql_connection("ride_review_oltp")
                    if conn:
                        try:
                            ticket_id = insert_support_ticket(conn, {
                                'ride_id': int(ride_id),
                                'customer_id': int(customer_id),
                                'driver_id': int(driver_id) if driver_id > 0 else None,
                                'issue_type': issue_type,
                                'description': description,
                                'status': status,
                                'priority': priority,
                                'assigned_agent_id': int(assigned_agent_id) if assigned_agent_id > 0 else None
                            })
                            st.success(f"✅ Support ticket created! ID: {ticket_id}")
                            conn.close()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please fill all required fields")