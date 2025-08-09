import os
import pandas as pd
from prophet import Prophet
from tqdm import tqdm  # For progress bar
import warnings

warnings.filterwarnings("ignore")

# Ensure output directory exists
os.makedirs("Multiforecasts", exist_ok=True)

# Load sales data
df = pd.read_csv("Data/sales_data.csv")

# Ensure date column is in datetime format
df["date"] = pd.to_datetime(df["date"])

# Get unique product IDs
unique_products = df["product_id"].unique()

# Loop through each product and forecast
print(f"Generating forecasts for {len(unique_products)} products...\n")

for pid in tqdm(unique_products):
    # Filter product-specific sales
    df_product = df[df["product_id"] == pid]

    # Group by date and aggregate units_sold
    df_grouped = df_product.groupby("date")["units_sold"].sum().reset_index()
    df_grouped.columns = ["ds", "y"]
    df_grouped["ds"] = pd.to_datetime(df_grouped["ds"])

    # Skip products with too little data
    if len(df_grouped) < 30:
        continue

    try:
        # Initialize and fit Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(df_grouped)

        # Predict next 60 days
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)

        # Add product_id to forecast output
        forecast["product_id"] = pid

        # Save forecast to CSV
        forecast[["ds", "yhat", "yhat_lower","yhat_upper", "product_id"]].to_csv( f"Multiforecasts/forecast_{pid}.csv", index=False )
    except Exception as e:
        print(f"⚠ Skipped {pid} due to error: {e}")
        continue

print("\n✅ All forecasts generated and saved in the 'forecasts/' directory.")