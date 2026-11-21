
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

model = joblib.load(
    "model_artifacts/random_forest_pipeline.joblib"
)

app = FastAPI(
    title="Online Shoppers Purchase Prediction",
    description="API для прогнозирования намерения совершить покупку"
)


@app.get("/")
def home():
    return {
        "message": "Online Shoppers ML API is running"
    }


@app.post("/predict")
def predict(data: dict):
    try:
        # Формируем DataFrame из входных данных
        X = pd.DataFrame([data])

        # Проверяем наличие необходимых признаков
        expected_features = list(model.feature_names_in_)
        missing = set(expected_features) - set(X.columns)

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {list(missing)}"
            )

        X = X[expected_features]

        prediction = int(model.predict(X)[0])
        probability = float(
            model.predict_proba(X)[0][1]
        )

        return {
            "prediction": prediction,
            "purchase_probability": probability,
            "result": (
                "Purchase" if prediction == 1
                else "No purchase"
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
