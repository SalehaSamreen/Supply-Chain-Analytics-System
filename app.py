import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime
import os

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="AI Smart Supply Chain Management", layout="wide")
st.title("🧠 AI Smart Supply Chain Management")

# ------------------------
# Load CSV files with safety checks
# ------------------------
@st.cache_data
def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"❌ File not found: {file_path}")
        st.stop()

# Load data
inventory_df = load_csv('data/inventory_data.csv')
product_catalog_df = load_csv('data/product_catalog.csv')
sales_df = load_csv('data/sales_data.csv')

# ------------------------
# Merge sales and product catalog
# ------------------------
merged_products = pd.merge(
    sales_df,
    product_catalog_df,
    on='product_id',
    suffixes=('_sales', '_catalog')
)

# ------------------------
# Product Selection
# ------------------------
product_names = merged_products['product_name_catalog'].unique()
selected_product = st.selectbox("📦 Select a Product to Forecast", sorted(product_names))

# Filter for selected product
product_data = merged_products[merged_products['product_name_catalog'] == selected_product]

# ------------------------
# Prepare Forecast Data
# ------------------------
forecast_df = product_data[['date', 'units_sold']].copy()
forecast_df['date'] = pd.to_datetime(forecast_df['date'], errors='coerce')
forecast_df.dropna(subset=['date'], inplace=True)
forecast_df = forecast_df.rename(columns={"date": "ds", "units_sold": "y"})

# ------------------------
# Forecasting with Prophet
# ------------------------
if len(forecast_df) < 2:
    st.warning("⚠ Not enough historical data to forecast this product.")
else:
    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # ------------------------
    # Plot Forecast
    # ------------------------
    st.subheader("📈 7-Day Demand Forecast")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # ------------------------
    # Inventory Check
    # ------------------------
    forecasted_demand = forecast.tail(7)['yhat'].sum()
    selected_product_id = product_data['product_id'].iloc[0]
    inventory_status = inventory_df[inventory_df['product_id'] == selected_product_id]
    current_stock = inventory_status['stock_level'].sum()

    st.subheader("📦 Current Inventory Status")
    st.write(f"Total Current Stock Across Warehouses: {int(current_stock)} units")

    st.subheader("📊 Required Inventory Based on Forecast")
    st.write(f"Forecasted Demand for Next 7 Days: {int(forecasted_demand)} units")

    # Inventory decision
    if current_stock >= forecasted_demand:
        st.success("✅ Current inventory is sufficient to meet forecasted demand.")
    else:
        shortage = forecasted_demand - current_stock
        st.error(f"⚠ Inventory shortage of {int(shortage)} units! Consider restocking.")