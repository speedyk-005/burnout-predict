<div align="center">
  <h1><b>🦉 burnout-predict</b></h1>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/speedyk-005/burnout-predict/main/logo.jpeg" alt="Chunklet-py Logo" width="400"/>
</p>

<p align="center">
  <b>Demo url: https://burnout-predict-1944.onrender.com/docs</b>
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

**Task.** Binary classification: predict `at_risk`, meaning a student is trending toward burnout. The survey records self-reported stress on three levels rather than a clinical burnout diagnosis, so `Medium` or `High` stress is treated as the operational proxy for burnout risk; collapsing the top two leaves a near-balanced target with roughly 59% of respondents positive.

**Cleaning.** `Timestamp` is dropped, the survey-question headers are reduced to their field name, rows are restricted to ages 19-24, and categorical values are lowercased so `Male` and `male` don't become separate categories.

**Features.** 14 survey columns: age, gender, study hours, class attendance, tuition, assignment load, sleep hours, physical exercise, social media use, screen time, family income level, peer pressure, family support, university type. `DictVectorizer` one-hot encodes the categorical ones, producing 21 columns.

**Excluded columns.** `stress_score` and `anxiety_level` measure stress directly and would leak the label; `exam_frequency` has no defined time window, so its scale is ambiguous across respondents.

**Model.** `DictVectorizer` + `XGBClassifier` in a single scikit-learn `Pipeline`, so the vectorizer's feature schema is saved with the model and inference takes plain dicts. The configuration is selected in `notebook.ipynb` over logistic regression and random forest.

**Evaluation.** A held-out 20% split (`SEED=42`): 600 students, 361 positive.

| Metric | Value |
| --- | --- |
| ROC AUC | 0.897 |
| Accuracy | 0.810 |
| Recall | 0.839 |
| Precision | 0.844 |
| F1 | 0.842 |

**Decision threshold.** The cutoff is **0.47**, tuned by sweep in the notebook rather than left at the sklearn default of 0.5. It flags 359 of 600 students instead of 351; five more true positives for three extra false positives, lifting recall from 0.825 to 0.839. The gain over 0.5 is small, and 58 of the 361 at-risk students still fall below the cutoff.

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

Input format: one JSON object per student (or a JSON array of them), with all 14 fields required.

| Field | Type | Accepted values |
| --- | --- | --- |
| `age` | integer | |
| `gender` | string | `male`, `female` |
| `study_hours` | number | |
| `class_attendance` | number | percentage |
| `tuition` | number | |
| `assignment_load` | number | |
| `sleep_hours` | number | |
| `physical_exercise` | number | |
| `social_media_use` | number | |
| `screen_time` | number | |
| `family_income_level` | string | `low`, `medium`, `high` |
| `peer_pressure` | number | |
| `family_support` | number | |
| `university_type` | string | `national`, `private`, `public` |

Unknown fields are rejected with `422`, and the same schema is rendered interactively at `/docs`.

### Live demo

The model is deployed on Render: **https://burnout-predict-1944.onrender.com**

| Endpoint | Purpose |
| --- | --- |
| [`/docs`](https://burnout-predict-1944.onrender.com/docs) | Interactive OpenAPI documentation |
| [`/health`](https://burnout-predict-1944.onrender.com/health) | Liveness probe, returns `{"status": "ok"}` |
| `POST /predict` | Returns one NDJSON record per input |

```bash
curl -X POST https://burnout-predict-1944.onrender.com/predict \
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

Response:

```
{"at_risk": true, "probability": 0.987}
```

> [!Note]
> The demo runs on Render's free plan, which sleeps after 15 minutes without traffic. The first request after that takes about a minute while the instance wakes up; later requests answer in around a second.

The steps below run the same service locally.

### Local

#### 1. Train the model

```bash
uv run python train.py
```

Reads `data/raw/university_student_stress.csv`, fits the `DictVectorizer` + `XGBClassifier` pipeline, writes `data/test.csv` and `pipeline.bin`. Expected output:

```
Training...
Test split saved as `data/test.csv`

 == Metrics ==

roc_auc_score: 0.897
accuracy_score: 0.81
recall_score: 0.839
f1_score: 0.842

Model pipeline saved as: `pipeline.bin`
Open with joblib.load('pipeline.bin')
```

#### 2. Start the API

```bash
uv run python predict.py
```

Serves on `http://0.0.0.0:8000`. Endpoints: `GET /health` (liveness probe) and `POST /predict`; both are documented interactively at `http://127.0.0.1:8000/docs`.

#### 3. Verify it works

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

## Deployment

The service runs on Render's free plan, built from the [`Dockerfile`](Dockerfile) in this repo.

### Deploy

1. In the [Render Dashboard](https://dashboard.render.com), click **New > Web Service**.
2. Connect this repository and pick the branch to track.
3. Select **Docker** as the language, set the health check path to `/health`, and choose the **Free** plan.
4. Render builds the `Dockerfile` from the repo root and serves the result at `https://<service-name>.onrender.com`.

Every later push to that branch rebuilds the image and redeploys.

Render injects `PORT`, which `predict.py` already reads, and the image is built from the `pipeline.bin` committed to the repo, so no build-time training step is needed.

