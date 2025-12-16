# api_app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load the trained model from the joblib file
model = joblib.load('demand_forecaster_model.joblib')

# Initialize the FastAPI app
app = FastAPI(
    title="Bike Rental Demand Forecaster API",
    description="An API for predicting daily bike rental demand using the trained XGBoost model."
)

# Define the input data structure using Pydantic BaseModel
# These fields must match exactly the features used during training (our 'features' list)
class PredictionInput(BaseModel):
    season: int
    year: int
    month: int
    holiday: int
    weekday: int
    workingday: int
    weather_condition: int
    temp: float
    humidity: float
    windspeed: float
    is_weekend: int
    is_holiday_week: int # The new feature we added

    # Add a sample input example for the automatic documentation (Swagger UI)
    model_config = {
        "json_schema_extra": {
            "example": {
                "season": 4, # Winter
                "year": 1,   # 2012
                "month": 12,
                "holiday": 0,
                "weekday": 3, # Wednesday
                "workingday": 1,
                "weather_condition": 2, # Misty/Rain
                "temp": 0.35,
                "humidity": 0.53,
                "windspeed": 0.15,
                "is_weekend": 0,
                "is_holiday_week": 1 # Example of our engineered feature in action
            }
        }
    }

# Define the root endpoint (for health check)
@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": True}

# Define the prediction endpoint using a POST request
@app.post("/predict_demand/")
def predict_demand(data: PredictionInput):
    # Convert input Pydantic model to a Pandas DataFrame row for the model
    # The data must be in the exact order the model expects
    input_data_df = pd.DataFrame([data.model_dump()])
    
    # Ensure categorical dtypes are set correctly for XGBoost to handle natively
    categorical_cols = ['season', 'year', 'month', 'holiday', 'weekday', 'workingday', 'weather_condition', 'is_weekend', 'is_holiday_week']
    for col in categorical_cols:
        input_data_df[col] = input_data_df[col].astype('category')

    # Make the prediction
    prediction = model.predict(input_data_df)
    
    # Return the prediction as a JSON response
    return {"predicted_rentals": float(prediction[0])}
