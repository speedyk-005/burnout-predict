<div align="center">
  <h1><b>🦉 burnout-predict</b></h1>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/speedyk-005/burnout-predict/main/logo.jpeg" alt="Chunklet-py Logo" width="400"/>
</p>

## The Problem

Academic institutions increasingly face high rates of student burnout, driven by heavy course loads, sleep deprivation, excessive screen time, and psychological stress. While early intervention can prevent severe academic failure and psychological distress, educational institutions lack automated tools to flag students at high risk before they reach a breaking point.

Traditional assessments rely on manual, post-hoc evaluations that fail to identify elevated stress levels in real time. Survey-based telemetry (capturing academic demands, lifestyle habits, and socio-economic support) contains strong predictive signals, but without automated scoring, these metrics sit idle in raw datasets.


## The Machine Learning Solution

`burnout-predict` addresses this by providing a deployable end-to-end classification pipeline.

This API is designed to be integrated into:

* **Academic Advising Dashboards:** Automatically triaging students who may need academic extensions or counseling resources based on self-reported periodic surveys.
* **Personal Wellbeing Applications:** Powering client-facing tools that alert users when their current work-to-rest ratio crosses a dangerous threshold.

## Dataset

This project uses the University Student Stress Dataset from Mendeley Data ([DOI: 10.17632/rc5htd5dfr.1](https://data.mendeley.com/datasets/rc5htd5dfr/1)), specifically the `Raw_Responses` directory ([Direct File Link](https://data.mendeley.com/datasets/rc5htd5dfr/1/files/02e68a9a-6e92-425a-bc76-5a218eaf7b45)).

**Storage Location:** A copy of the raw dataset is committed under `data/raw/university_student_stress.csv` to ensure offline execution and end-to-end training reproducibility.

## Model

**Task.** Binary classification: predict burnout with `at_risk`, defined as a self-reported stress level of `Medium` or `High` versus `Low`. The survey records three levels; collapsing the top two leaves a near-balanced target, with roughly 59% of respondents positive.

**Cleaning.** `Timestamp` is dropped, the survey-question headers are reduced to their field name, rows are restricted to ages 19-24, and categorical values are lowercased so `Male` and `male` don't become separate categories.

**Features.** 14 survey columns: age, gender, study hours, class attendance, tuition, assignment load, sleep hours, physical exercise, social media use, screen time, family income level, peer pressure, family support, university type. `DictVectorizer` one-hot encodes the categorical ones, producing 21 columns.

**Excluded columns.** `stress_score` and `anxiety_level` measure stress directly and would leak the label; `exam_frequency` has no defined time window, so its scale is ambiguous across respondents.

**Model.** `DictVectorizer` + `XGBClassifier` in a single scikit-learn `Pipeline`, so the vectorizer's feature schema is saved with the model and inference takes plain dicts. The configuration is selected in `notebook.ipynb` over logistic regression and random forest.

**Evaluation.** A held-out 20% split (`SEED=42`): 600 students, 361 positive.

| Metric | Value |
| --- | --- |
| ROC AUC | 0.897 |
| Accuracy | 0.808 |
| Recall | 0.834 |
| Precision | 0.846 |
| F1 | 0.840 |

**Decision threshold.** The cutoff is **0.48**, tuned by sweep in the notebook rather than left at the sklearn default of 0.5. It flags 356 of 600 students instead of 351; three more true positives for two extra false positives, lifting recall from 0.826 to 0.834. The gain over 0.5 is small, and 60 of the 361 at-risk students still fall below the cutoff.

## Installation

**Requirements:** Python 3.11+ (3.12 is used in development) and [uv](https://docs.astral.sh/uv/getting-started/installation/). No GPU is needed; training takes a few seconds on CPU.

### 1. Clone the repository

```bash
git clone https://github.com/speedyk-005/burnout-predict.git
cd burnout-predict
```

### 2. Create the environment

```bash
uv sync --extra dev
```

This creates `.venv` and installs the exact versions pinned in `uv.lock`. The `dev` extra adds `ruff` for linting; drop `--extra dev` if you only want to run the service.

<details>
<summary>Using plain <code>pip</code> instead of <code>uv</code></summary>

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```
</details>

## Usage

### 1. Train the model

```bash
uv run python train.py
```

Reads `data/raw/university_student_stress.csv`, fits the `DictVectorizer` + `XGBClassifier` pipeline, writes `data/test.csv` and `pipeline.bin`. Expected output:

```
Training...
Test split saved as `data/test.csv`

 == Metrics ==

roc_auc_score: 0.897
accuracy_score: 0.808
recall_score: 0.834
f1_score: 0.84

Model pipeline saved as: `pipeline.bin`
Open with joblib.load('pipeline.bin')
```

> [!Note]
> `pipeline.bin` is the artifact the API loads, so this step must run before the service starts.

### 2. Start the API

```bash
uv run python predict.py
```

Serves on `http://0.0.0.0:8000`. Endpoints: `GET /health` (liveness probe) and `POST /predict`; both are documented interactively at `http://127.0.0.1:8000/docs`.

### 3. Verify it works

Input format:

```python
class StudentFeatures(BaseModel):
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
```

With the server running, in a second terminal:

```bash
uv run python test.py
```

Expected output:

```
Firing request to http://127.0.0.1:8000/predict...

Streamed Response Records:
{'at_risk': True, 'probability': 0.987}
```

Or with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "age": 20, "gender": "male", "study_hours": 8,
        "class_attendance": 90.0, "tuition": 1500.0, "assignment_load": 9,
        "sleep_hours": 6.0, "physical_exercise": 4, "social_media_use": 9,
        "screen_time": 7.5, "family_income_level": "low",
        "peer_pressure": 3.0, "family_support": 3.0,
        "university_type": "national"
      }'
```

### Docker

`pipeline.bin` must exist before the image is built (run step 1 first).

```bash
docker build -t burnout-predict .
docker run --rm -p 8000:8000 burnout-predict
```

The image needs no network access at runtime: both the model and its feature schema are baked in. Dependencies are installed with `uv sync --frozen`, so the image uses the exact versions pinned in `uv.lock`; the same ones as your local `.venv`.
