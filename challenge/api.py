import fastapi
from fastapi import HTTPException
from challenge.model import DelayModel
from pydantic import BaseModel
import pandas as pd
import os
import uvicorn

app = fastapi.FastAPI()
delay_model = DelayModel()

class PredictRequest(BaseModel):
    features: dict

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'trained_model', 'xgboost_model.pkl')
    delay_model.load_model(model_path)

    try:
        features_df = pd.DataFrame([request.features])
        prediction = delay_model.predict(features=features_df)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
