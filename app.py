from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "motor_current_A", "rated_motor_current_A", "vibration_mm_s",
    "belt_age_months", "parcels_per_minute", "dust_accumulation_index",
    "days_since_maintenance", "humidity_percent",
    "motor_load_ratio", "belt_wear_pressure_index"
]

app = FastAPI(title="Conveyor Belt Jam Prediction API", version="1.0")

class ConveyorInput(BaseModel):
    motor_current_A: float = Field(gt=0)
    rated_motor_current_A: float = Field(gt=0)
    vibration_mm_s: float = Field(ge=0)
    belt_age_months: float = Field(ge=0)
    parcels_per_minute: float = Field(ge=0)
    dust_accumulation_index: float = Field(ge=0, le=100)
    days_since_maintenance: float = Field(ge=0)
    humidity_percent: float = Field(ge=0, le=100)

@app.get("/health")
def health():
    return {"status": "healthy", "model": "ConveyorJamPrediction"}

@app.post("/predict")
def predict(data: ConveyorInput):
    row = data.dict()
    row["motor_load_ratio"] = row["motor_current_A"] / row["rated_motor_current_A"]
    row["belt_wear_pressure_index"] = (
        row["belt_age_months"] / 60.0 *
        (1.0 + row["dust_accumulation_index"] / 100.0)
    )
    frame = pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    return {
        "prediction": "Jam Risk" if prediction == 1 else "Normal Operation",
        "jam_probability": round(probability, 4)
    }
