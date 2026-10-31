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
