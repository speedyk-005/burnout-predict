"""Training script for the student burnout risk classifier.

Extracted from `notebook.ipynb` ("Student Burnout Risk Detection") in this
repository, where the EDA, feature selection, and model/threshold tuning were
done.

Only the final pipeline is kept here:
  - clean the raw survey data, split it
  - fit a `DictVectorizer` + `XGBClassifier` pipeline
  - print test metrics, and persist the pipeline to `pipeline.bin`.

The `THRESHOLD` cutoff below (0.48) came out of the threshold sweep in the
notebook, not from the default 0.5.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RAW_DATA_PATH = "data/raw/university_student_stress.csv"
SEED = 42
TEST_SIZE = 0.2
XGB_PARAMS = {
    "objective": "binary:logistic",
    "min_child_weight": 20,
    "max_depth": 4,
    "eta": 0.1,
    "nthread": 8,
    "seed": SEED,
    "verbosity": 1,
    "n_estimators": 100,
}
THRESHOLD = 0.48


def load_and_clean_dataset(path: str) -> pd.DataFrame:
    # Read the file with codec `latin1` codec to prevent decoding error
    df = pd.read_csv(path, encoding="latin1")

    df_clean = df.drop("Timestamp", axis=1)

    # The column names are messy; let's keep the necessary parts.
    df_clean.columns = df_clean.columns.str.split().str[1].str.lower()

    # Ensure age range
    df_clean = df_clean[(df_clean["age"] >= 19) & (df_clean["age"] <= 24)]

    # Normalize income, gender and university type
    df_clean["gender"] = df_clean["gender"].str.lower()
    df_clean["family_income_level"] = df_clean["family_income_level"].str.lower()
    df_clean["university_type"] = df_clean["university_type"].str.split().str[0].str.lower()

    # Let's convert the label from ternary to binary
    df_clean["at_risk"] = df_clean["stress_level"].map({"Low": 0, "Medium": 1, "High": 1})

    df_clean = df_clean.drop(
        columns=[
            "stress_level",
            "stress_score",
            "exam_frequency",  # Lacks a temporal anchor
            "anxiety_level",  # Related to stress too much
        ]
    )
    return df_clean


def train_test_split_df(df: pd.DataFrame):
    """Splits the input dataframe into training and test sets

    Then writes the test set to disk.
    """
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    df_test.to_csv("data/test.csv", index=False)
    print("Test split saved as `data/test.csv`")

    return df_train, df_test


def train_model_pipeline(df_train: pd.DataFrame) -> Pipeline:
    X_train_records = df_train.drop("at_risk", axis=1).to_dict("records")
    y_train = df_train["at_risk"].values

    pipeline = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", XGBClassifier(**XGB_PARAMS)),
        ]
    )

    pipeline.fit(X_train_records, y_train)

    return pipeline


def print_model_metrics(model, df_test):
    X_test_records = df_test.drop("at_risk", axis=1).to_dict("records")
    y_test = df_test["at_risk"].values

    # Predict the probability of burnout, then binarize with the tuned cutoff
    y_pred = model.predict_proba(X_test_records)[:, 1]
    y_pred_binary = (y_pred >= THRESHOLD).astype(int)

    print("\n == Metrics == \n")
    print("roc_auc_score:", round(roc_auc_score(y_test, y_pred), 3))
    print("accuracy_score:", round(accuracy_score(y_test, y_pred_binary), 3))
    print("recall_score:", round(recall_score(y_test, y_pred_binary), 3))
    print("f1_score:", round(f1_score(y_test, y_pred_binary), 3))
    print()


if __name__ == "__main__":
    print("Training...")
    df_clean = load_and_clean_dataset(RAW_DATA_PATH)
    df_train, df_test = train_test_split_df(df_clean)

    pipeline = train_model_pipeline(df_train)
    print_model_metrics(pipeline, df_test)

    # Save the entire pipeline
    print("Model pipeline saved as: `pipeline.bin`")
    print("Open with joblib.load('pipeline.bin')")
    joblib.dump(pipeline, "pipeline.bin")
