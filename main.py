from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI(title="API Pipeline Taller 3")
pipeline = joblib.load("pipeline.joblib")

class Entrada(BaseModel):
    sensor_temperatura: Optional[float] = None   # el pipeline imputa los nulos
    sensor_presion: float

@app.get("/")
def home():
    return {"mensaje": "API del pipeline funcionando"}

@app.post("/predict")
def predict(data: Entrada):
    try:
        df = pd.DataFrame([data.model_dump()]).astype(float)   # None -> NaN
        clase = int(pipeline.predict(df)[0])
        prob = float(pipeline.predict_proba(df)[0][1])
        return {"prediccion": clase, "probabilidad_clase_1": round(prob, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))