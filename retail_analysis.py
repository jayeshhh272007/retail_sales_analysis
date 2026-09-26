import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("Retail.csv")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())

# Filling  missing SalesRegion with "Unknown"
df["SalesRegion"] = df["SalesRegion"].fillna("Unknown")

# Filling  missing OrderQuantity with the median
df["OrderQuantity"] = pd.to_numeric(df["OrderQuantity"], errors="coerce")
df["OrderQuantity"] = df["OrderQuantity"].fillna(df["OrderQuantity"].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Convert Orderdate to datetime
df["Orderdate"] = pd.to_datetime(df["Orderdate"])

# Create time-based features
df["Month"] = df["Orderdate"].dt.month
df["Quarter"] = df["Orderdate"].dt.quarter
df["Weekday"] = df["Orderdate"].dt.day_name()
df["Total Revenue"] = df["SalesAmount"]
print("\nNew Columns:")
print(df[["Orderdate", "Month", "Quarter", "Weekday"]].head())
# Save cleaned dataset
df.to_csv("cleaned_retail_sales.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ==============================
# EDA - Visualization 1
# Monthly Sales Trend
# ==============================

# Monthly sales trend
df["YearMonth"] = df["Orderdate"].dt.to_period("M")

monthly_sales = df.groupby("YearMonth")["Total Revenue"].sum()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index.astype(str), monthly_sales.values)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==============================
# EDA - Visualization 2
# Product Category Performance
# ==============================

category_sales = df.groupby("Category")["Total Revenue"].sum()

plt.figure(figsize=(10, 6))
category_sales.plot(kind="bar")

plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==============================
# EDA - Visualization 3
# Revenue by Sales Region
# ==============================

region_sales = df.groupby("SalesRegion")["Total Revenue"].sum()

plt.figure(figsize=(10, 6))
region_sales.plot(kind="bar")

plt.title("Revenue by Sales Region")
plt.xlabel("Sales Region")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==============================
# EDA - Visualization 4
# Product Subcategory Distribution
# ==============================

subcategory_count = df["Subcategory"].value_counts()

plt.figure(figsize=(12, 6))
subcategory_count.plot(kind="bar")

plt.title("Product Subcategory Distribution")
plt.xlabel("Product Subcategory")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==============================
# EDA - Visualization 5
# Correlation Heatmap
# ==============================

numeric_columns = [
    "ListPrice",
    "OrderQuantity",
    "UnitPrice",
    "SalesAmount",
    "DiscountAmount",
    "TaxAmount",
    "Freight"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 7))
plt.imshow(correlation, cmap="coolwarm", aspect="auto")

plt.colorbar(label="Correlation")
plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=45)
plt.yticks(range(len(numeric_columns)), numeric_columns)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ==============================
# EDA - Key Numerical Insights
# ==============================

print("\n========== KEY INSIGHTS ==========")

# 1. Best performing category
best_category = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
print("\nRevenue by Category:")
print(best_category)

# 2. Best performing region
best_region = df.groupby("SalesRegion")["Total Revenue"].sum().sort_values(ascending=False)
print("\nRevenue by Sales Region:")
print(best_region)

# 3. Best performing subcategory
best_subcategory = df.groupby("Subcategory")["Total Revenue"].sum().sort_values(ascending=False)
print("\nRevenue by Subcategory:")
print(best_subcategory.head(10))

# 4. Monthly revenue
print("\nMonthly Revenue:")
print(monthly_sales)

# 5. Correlation with SalesAmount
print("\nCorrelation with Sales Amount:")
print(correlation["SalesAmount"].sort_values(ascending=False))

# ==============================
# MILESTONE 3 - PREDICTIVE MODELING
# Linear Regression
# ==============================

print("\n========== PREDICTIVE MODELING ==========")

# Convert monthly data into a DataFrame
monthly_df = monthly_sales.reset_index()
monthly_df.columns = ["YearMonth", "Revenue"]

# Create a numerical time index
monthly_df["TimeIndex"] = range(len(monthly_df))

print("\nMonthly Data:")
print(monthly_df)

# Split data into training and testing sets
split_point = int(len(monthly_df) * 0.8)

train = monthly_df.iloc[:split_point]
test = monthly_df.iloc[split_point:]

# Features and target
X_train = train[["TimeIndex"]]
y_train = train["Revenue"]

X_test = test[["TimeIndex"]]
y_test = test["Revenue"]

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ==============================
# Forecast Next 3 Months
# ==============================

# Train model using all available monthly data
X_all = monthly_df[["TimeIndex"]]
y_all = monthly_df["Revenue"]

model.fit(X_all, y_all)

# Create TimeIndex for next 3 months
future_index = np.arange(
    len(monthly_df),
    len(monthly_df) + 3
).reshape(-1, 1)

# Predict future revenue
future_predictions = model.predict(future_index)

# Get the last available month
last_month = monthly_df["YearMonth"].iloc[-1]

# Create future month names
future_months = pd.period_range(
    start=last_month + 1,
    periods=3,
    freq="M"
)

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    "Month": future_months.astype(str),
    "Forecasted Revenue": future_predictions
})

print("\n========== NEXT 3 MONTHS FORECAST ==========")
print(forecast_df)

# Save forecast results
forecast_df.to_csv("forecast_results.csv", index=False)

print("\nForecast results saved successfully!")