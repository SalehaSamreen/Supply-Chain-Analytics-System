import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os 

os.makedirs("SinglePrediction", exist_ok=True)
#load data
df=pd.read_csv("data/sales_data.csv")

#Filter sales data for one product
product_id="FOO_09"
df_product = df[df["product_id"] == product_id]

#Group[by date to get daily sales]
df_grouped = df_product.groupby("date")["units_sold"].sum().reset_index()
print(df_grouped.head())
df_grouped.columns=["ds","y"] #rename for prophet

#Train the model
model=Prophet(daily_seasonality=True)
model.fit(df_grouped)

##forecaste
future = model.make_future_dataframe(periods=7)
forecast=model.predict(future)

#plot forecast
fig = model.plot(forecast)
plt.title(f"Demand Forecast for {product_id}")
plt.show()

#save forecast to csv
forecast.to_csv(f"SinglePrediction/forecast_{product_id}.csv", index=False)