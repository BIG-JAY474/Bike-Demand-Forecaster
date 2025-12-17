# Bike-Demand-Forecaster
🚲 Bike Rental Demand Forecaster
End-to-End Machine Learning Project | UCI Machine Learning Repository


📌 Project Overview
This project predicts daily bike rental demand using the UCI Bike Sharing Dataset. By analyzing seasonal patterns and weather conditions, the system provides actionable forecasts to optimize bike distribution and maintenance schedules.

📈 The 6-Stage ML Workflow

1. Problem Definition
Goal: Build a regression model to forecast daily total bike rentals.
Business Impact: Minimize station "stockouts" and reduce overhead costs.
Target Metric: Mean Absolute Error (MAE) < 850 rentals.

3. Data Collection & Preparation
Source: UCI Machine Learning Repository.
Key Actions:
Loaded 731 daily records (day.csv).
Converted categorical features (season, weather) into proper data types.
Validated data integrity (no missing values found).

4. EDA & Feature Engineering
Through Exploratory Data Analysis, I discovered that temperature and seasonality are the strongest predictors.
Key Feature Engineered: is_holiday_week.
Why? I observed that standard "holiday" flags only cover single days, while bike demand stays low for the entire week surrounding Christmas and New Year. Adding this flag significantly improved performance during the holiday season.
![Performance Plot]
(Actual vs Predicted Rentals)

6. Model Selection & Training
Model: XGBoost Regressor (Extreme Gradient Boosting).
Training Strategy:
Time-Series Split: Trained on historical data from 2011 to late 2012.
Validation: Reserved the final 60 days (Nov/Dec 2012) for testing to prevent "cheating" with future data.
Implementation: Native categorical support enabled via enable_categorical=True.

7. Evaluation & Testing
The final model was evaluated against the unseen 60-day test window:
Mean Absolute Error (MAE): 842.83
R-Squared (R²): 0.53
Analysis: The model successfully captures the sharp demand "dips" during winter storms and the end-of-year holiday period.

8. Deployment & Monitoring
API: A FastAPI server (api_app.py) was developed to serve real-time predictions.
Monitoring Strategy: I recommend tracking Data Drift (monitoring temperature changes) and Model Drift (tracking weekly MAE) to determine when the model requires retraining.
📂 Repository Structure
demand_forecaster.ipynb: Complete code for EDA, engineering, and model evaluation.
api_app.py: FastAPI implementation for real-time inference.
demand_forecaster_model.joblib: The production-ready serialized model.
day.csv: The official dataset from UCI.
requirements.txt: Environment dependencies.
🛠️ How to Use
Clone: git clone github.com[YOUR_USERNAME]/bike-sharing-forecaster.git
Install: pip install -r requirements.txt
Run: Open demand_forecaster.ipynb to view the analysis, or run the API:
bash
uvicorn api_app:app --reload
Use code with caution.

Author: [Your Name]
Date: December 2025
Final Checklist for GitHub:
Main Image: If you have a screenshot of your "Actual vs. Predicted" plot, add it right under the "Evaluation" section.
Interactive Docs: Mention that users can see the API docs at /docs once they run the server.
Clean Code: Ensure your .ipynb file has markdown headings so it’s easy for a recruiter to read without running it.
Congratulations on finishing your first professional ML portfolio project!
AI responses may include mistakes. For financial advice, consult a professional. Learn more



