import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

app = FastAPI(title="Smart Crop API - Offline Climate Edition")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart Crop API - Offline Climate Edition")

# Add this CORS middleware block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

CLIMATE_DB = {
    "Pune": {
        1:  {"temp": 21.5, "humidity": 52.0, "rainfall": 0.0},
        2:  {"temp": 23.5, "humidity": 41.0, "rainfall": 0.5},
        3:  {"temp": 27.0, "humidity": 35.0, "rainfall": 1.5},
        4:  {"temp": 29.5, "humidity": 38.0, "rainfall": 12.0},
        5:  {"temp": 30.0, "humidity": 52.0, "rainfall": 35.0},
        6:  {"temp": 27.5, "humidity": 76.0, "rainfall": 155.0},
        7:  {"temp": 25.5, "humidity": 86.0, "rainfall": 280.0},
        8:  {"temp": 25.0, "humidity": 85.0, "rainfall": 210.0},
        9:  {"temp": 25.5, "humidity": 82.0, "rainfall": 125.0},
        10: {"temp": 26.5, "humidity": 68.0, "rainfall": 65.0},
        11: {"temp": 24.0, "humidity": 58.0, "rainfall": 15.0},
        12: {"temp": 21.5, "humidity": 55.0, "rainfall": 2.0}
    }
}

class UserInput(BaseModel):
    N: float = Field(..., description="Nitrogen")
    P: float = Field(..., description="Phosphorous")
    K: float = Field(..., description="Potassium")
    ph: float = Field(..., description="pH value")
    city: str = Field(..., description="City name (e.g., 'Pune')")
    planting_month: int = Field(..., ge=1, le=12, description="Month 1-12")

def get_seasonal_climate(city: str, planting_month: int, season_length_months: int = 3):
    if city not in CLIMATE_DB:
        raise HTTPException(status_code=400, detail=f"No climate data for {city}. Available: {list(CLIMATE_DB.keys())}")
    
    city_data = CLIMATE_DB[city]
    total_temp, total_humidity, total_rainfall = 0.0, 0.0, 0.0
    
    for i in range(season_length_months):
        current_month = ((planting_month - 1 + i) % 12) + 1 
        weather = city_data[current_month]
        
        total_temp += weather["temp"]
        total_humidity += weather["humidity"]
        total_rainfall += weather["rainfall"]
        
    return (
        round(total_temp / season_length_months, 2), 
        round(total_humidity / season_length_months, 2), 
        round(total_rainfall / season_length_months, 2)
    )

# 5. The Prediction Endpoint
@app.post("/predict")
def predict_crop(payload: UserInput):
    try:
        # Step A: Get the local climate math
        avg_temp, avg_hum, avg_rain = get_seasonal_climate(payload.city, payload.planting_month)
        
        # Step B: Combine frontend data with database climate data
        combined_data = {
            "N": payload.N,
            "P": payload.P,
            "K": payload.K,
            "temperature": avg_temp,
            "humidity": avg_hum,
            "ph": payload.ph,
            "rainfall": avg_rain
        }
        
        input_df = pd.DataFrame([combined_data])
        prediction = model.predict(input_df)
        
        result = int(prediction[0]) if isinstance(prediction[0], (np.integer, np.ndarray)) else prediction[0]
            
        return {
            "success": True,
            "weather_used": {
                "avg_temperature": avg_temp,
                "avg_humidity": avg_hum,
                "avg_rainfall": avg_rain,
                "season_length": "3 months"
            },
            "prediction": result
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
