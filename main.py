
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load trained Penguins model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Penguins Random Forest API")


# Input features
class PenguinInput(BaseModel):
    culmen_length_mm: float = Field(gt=0)
    culmen_depth_mm: float = Field(gt=0)
    flipper_length_mm: float = Field(gt=0)
    body_mass_g: float = Field(gt=0)


@app.get("/")
def ready_to_serve():
    return {"status": "Ready to serve"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Endpoint for making predictions
@app.post("/predict")
def predict(data: PenguinInput):
    features = pd.DataFrame([{
        "culmen_length_mm": data.culmen_length_mm,
        "culmen_depth_mm": data.culmen_depth_mm,
        "flipper_length_mm": data.flipper_length_mm,
        "body_mass_g": data.body_mass_g
    }])

    prediction = str(model.predict(features)[0])

    print("Prediction is", prediction)

    return {
        "prediction": prediction,
        "class_name": prediction
    }
