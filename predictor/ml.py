from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings


MODEL_PATH = Path(settings.BASE_DIR) / "models" / "ckd_model.joblib"


@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file is missing. Run: python train_model.py")
    return joblib.load(MODEL_PATH)


def predict_ckd(patient_data: dict) -> dict:
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    row = pd.DataFrame([{column: patient_data.get(column) for column in feature_columns}])

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    has_ckd = prediction == 1
    return {
        "prediction": "ckd" if has_ckd else "notckd",
        "probability": round(probability * 100, 2),
        "message": (
            "You have CKD please have proper medical Consultation"
            if has_ckd
            else "You do not have CKD"
        ),
        "status": "danger" if has_ckd else "safe",
    }
