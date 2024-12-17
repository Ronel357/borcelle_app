import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

# Initialize PostgreSQL connection
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

# Query execution function
def run_query(query, conn):
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()
    return pd.DataFrame(data, columns=columns)

# Load CSS for custom styling
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("./assets/style1.css")

# Sales Forecasting Page
st.title("Sales Forecasting by Region")

# Initialize PostgreSQL connection
conn = init_connection()

# Query raw sales data with region from the database
query_raw_sales = """
SELECT 
    fs.order_date, 
    fs.sales,
    dc.region
FROM fact_sales fs
JOIN dim_customer dc ON fs.customer_id = dc.customer_id
ORDER BY fs.order_date;
"""
sales_data = run_query(query_raw_sales, conn)

# Close the connection
conn.close()

# Display the raw sales data to verify it
st.subheader("Raw Sales Data")
st.write(sales_data.head())

# Feature engineering
sales_data["order_date"] = pd.to_datetime(sales_data["order_date"])
sales_data["month"] = sales_data["order_date"].dt.to_period("M")

# Aggregating sales data by region and month
monthly_sales = sales_data.groupby(["region", "month"])["sales"].sum().reset_index()

# Display the aggregated monthly sales data
st.subheader("Aggregated Monthly Sales Data")
st.write(monthly_sales.head())

# Allow the user to select a region
region = st.selectbox("Select Region", monthly_sales["region"].unique())
region_sales = monthly_sales[monthly_sales["region"] == region]

# Allow the user to filter data by date range
start_date, end_date = st.date_input(
    "Select Date Range",
    [region_sales["month"].min().to_timestamp(), region_sales["month"].max().to_timestamp()],
)
filtered_sales = region_sales[ 
    (region_sales["month"].dt.to_timestamp() >= pd.to_datetime(start_date)) & 
    (region_sales["month"].dt.to_timestamp() <= pd.to_datetime(end_date))
]

st.subheader(f"Filtered Sales Data for {region}")
st.write(filtered_sales)

# Prepare data for Ridge Regression
filtered_sales["month"] = filtered_sales["month"].astype(str)
filtered_sales["month"] = pd.to_datetime(filtered_sales["month"])
filtered_sales["month_num"] = (filtered_sales["month"] - filtered_sales["month"].min()).dt.days

# Create lag features
for lag in range(1, 4):
    filtered_sales[f"sales_lag_{lag}"] = filtered_sales["sales"].shift(lag)

# Add rolling statistics
filtered_sales["rolling_mean_3"] = filtered_sales["sales"].rolling(window=3).mean()
filtered_sales["rolling_std_3"] = filtered_sales["sales"].rolling(window=3).std()

# Drop rows with NaN values (introduced by lagging)
filtered_sales = filtered_sales.dropna()

# Define features and target
X = filtered_sales[["month_num", "sales_lag_1", "sales_lag_2", "sales_lag_3", "rolling_mean_3", "rolling_std_3"]]
y = filtered_sales["sales"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Ridge Regression model
param_grid = {
    "alpha": [0.1, 1.0, 10.0, 100.0]
}
ridge_model = Ridge()
grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5, scoring="r2")
grid_search.fit(X_train, y_train)

# Get the best model
best_ridge_model = grid_search.best_estimator_
y_pred = best_ridge_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader(f"Model Performance for {region} (Ridge Regression)")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.4f}")

# Allow user to select future years for forecasting
future_years = st.slider("Select the number of future years to predict:", 1, 5, 1)

# Generate future dates
last_month = filtered_sales["month"].max()
future_months = pd.date_range(
    start=last_month + pd.offsets.MonthBegin(1), 
    periods=future_years * 12, 
    freq="MS"
)
future_data = pd.DataFrame({"month": future_months})
future_data["month_num"] = (future_data["month"] - filtered_sales["month"].min()).dt.days

# Initialize future data with lag features using the last known values
last_known_sales = list(filtered_sales["sales"][-3:])
future_predictions = []

for i in range(len(future_months)):
    # Create lag features for the current month
    lag_1 = last_known_sales[-1] if len(last_known_sales) >= 1 else 0
    lag_2 = last_known_sales[-2] if len(last_known_sales) >= 2 else 0
    lag_3 = last_known_sales[-3] if len(last_known_sales) >= 3 else 0
    
    rolling_mean_3 = np.mean(last_known_sales) if len(last_known_sales) >= 3 else 0
    rolling_std_3 = np.std(last_known_sales) if len(last_known_sales) >= 3 else 0
    
    # Construct feature row for prediction
    feature_row = pd.DataFrame({
        "month_num": [future_data.loc[i, "month_num"]], 
        "sales_lag_1": [lag_1], 
        "sales_lag_2": [lag_2], 
        "sales_lag_3": [lag_3], 
        "rolling_mean_3": [rolling_mean_3], 
        "rolling_std_3": [rolling_std_3]
    })
    
    # Scale the feature row and predict sales
    feature_row_scaled = scaler.transform(feature_row)
    predicted_sales = best_ridge_model.predict(feature_row_scaled)[0]
    future_predictions.append(predicted_sales)
    
    # Update last_known_sales with the predicted value for the next iteration
    last_known_sales.append(predicted_sales)
    if len(last_known_sales) > 3:  # Keep the window size consistent
        last_known_sales.pop(0)

# Add predicted sales to the future data
future_data["predicted_sales"] = future_predictions

st.subheader(f"Future Sales Predictions for {region}")
st.write(future_data)

# Visualize sales data
plt.figure(figsize=(12, 6))
plt.plot(filtered_sales["month"], filtered_sales["sales"], label="Actual Sales", marker="o")
plt.plot(future_data["month"], future_data["predicted_sales"], label="Predicted Sales", marker="x", linestyle="--")
plt.title(f"Sales Forecasting for {region} (Including Future Predictions)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)
