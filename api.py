from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# load model
model = joblib.load("model.pkl")   # adjust path if needed

@app.get("/")
def home():
    return {"message": "ML API running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # example: expects features array
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features).tolist()

        return {
            "prediction": prediction
        }
    except Exception as e:
        return {"error": str(e)}
