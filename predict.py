import json
from typing import Literal

import joblib
from fastapi import Body, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

app = FastAPI()

# Load the trained pipeline (DictVectorizer + XGBClassifier/XGBRegressor)
pipeline = joblib.load("pipeline.bin")

THRESHOLD = 0.46


class StudentFeatures(BaseModel):
    """Enforces the exact feature schema"""

    age: int
    gender: Literal["male", "female"]
    study_hours: float
    class_attendance: float
    tuition: float
    assignment_load: float
    sleep_hours: float
    physical_exercise: float
    social_media_use: float
    screen_time: float
    family_income_level: Literal["low", "medium", "high"]
    peer_pressure: float
    family_support: float
    university_type: Literal["national", "private", "public"]
    model_config = ConfigDict(extra="forbid")


@app.post("/predict")
def predict(payload: StudentFeatures | list[StudentFeatures] = Body(...)) -> StreamingResponse:
    """Runs inference via an XGBoost pipeline and streams JSONL predictions.

    Args:
        payload: Single student record or batch of records.

    Returns:
        Stream of JSONL records containing prediction and probability.
    """
    if isinstance(payload, StudentFeatures):
        payload = [payload]

    # Convert validated Pydantic models to plain dictionaries for DictVectorizer
    data_dicts = [item.model_dump() for item in payload]

    # Predict probabilities for the positive class (burnout)
    probabilities = pipeline.predict_proba(data_dicts)[:, 1]

    def generate():
        for prob in probabilities:
            prob = float(prob)
            record = {
                "burnout": prob >= THRESHOLD,
                "probability": round(prob, 3),
            }
            yield json.dumps(record) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("predict:app", host="127.0.0.1", port=8000, reload=True)
