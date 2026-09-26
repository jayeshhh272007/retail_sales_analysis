# Retail Sales Forecasting & Customer Purchase Analysis 
 
## Overview 
Retail Sales Forecasting and Customer Purchase Analysis is a Data Science project that looks at past retail sales data. The goal is to find out how much money is made, how well products do and how sales vary by area. The project has steps like cleaning the data making new features looking at the data to learn more making charts and using Linear Regression to make predictions. A forecast, for the three months of sales is made and shown in a Power BI dashboard that people can interact with. 
 
## Technologies Used 
- Python 
- Pandas 
- NumPy 
- Matplotlib 
- Scikit-learn 
- Power BI 
 
## Key Features 
- Data Cleaning 
- Exploratory Data Analysis 
- Sales Trend Analysis 
- Linear Regression 
- 3-Month Revenue Forecasting 
- Power BI Dashboard 
 
## Project Files 
1. Retail.csv – Original retail sales dataset used for the analysis.

2. Cleaned_retail_sales.csv – Cleaned dataset after preprocessing and feature engineering.

3. Retail_analysis.py – Python script containing data cleaning, EDA, visualizations, Linear Regression, model evaluation, and revenue forecasting.

4. Forecast_results.csv – Contains the predicted revenue for the next three months.

5. Skillflow_dashboard.pbix – Power BI dashboard containing KPIs, sales trends, category and regional analysis, and forecast results.

6. Updated_Retail_Sales_Forecasting_Final_Report.pdf – Detailed project report covering methodology, analysis, results, insights, and conclusion. 
 
## Results

* Category Performance: Bikes generated the highest revenue among the analyzed product categories, making it the strongest category in the dataset.

* Subcategory Performance: Road Bikes and Mountain Bikes were the major revenue-generating subcategories.

* Regional Performance: The Southwest region recorded the highest revenue among the analyzed sales regions.

* Sales Trends: Monthly revenue showed fluctuations across the available period rather than following a completely consistent trend.

* Correlation Analysis: Unit Price and List Price showed relatively strong positive correlations with Sales Amount, indicating a noticeable relationship between these variables in the dataset.

* Forecasting: A Linear Regression model was developed to forecast future monthly revenue. The model generated revenue estimates for the next three months: July, August, and September 2013.

* Forecast Output: The predicted revenue was approximately 3.02M for July, 3.05M for August, and 3.08M for September 2013, showing a gradual increase in the model-generated estimates.

* Model Evaluation: The model achieved an R² score of -0.1877, indicating that the simple time-based Linear Regression model did not explain the held-out revenue variation well. Therefore, the model is considered a baseline forecasting approach rather than a high-accuracy forecasting solution.
