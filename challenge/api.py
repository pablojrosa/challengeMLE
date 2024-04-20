import fastapi
from fastapi import HTTPException
from model import DelayModel
from pydantic import BaseModel
import pandas as pd
import os

app = fastapi.FastAPI()
delay_model = DelayModel()

# Definir el esquema de solicitud para el endpoint de predicción
class PredictRequest(BaseModel):
    features: dict  # Asumiendo que los features vienen en formato diccionario

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'trained_model', 'xgboost_model.pkl')
    delay_model.load_model(model_path)

    try:
        # Convertir el diccionario de features en un DataFrame
        features_df = pd.DataFrame([request.features])  # Usa una lista para convertir directamente
        # Realizar la predicción
        prediction = delay_model.predict(features=features_df)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))