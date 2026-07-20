from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/kidney_disease.csv")
MODEL_PATH = Path("models/ckd_model.joblib")
METRICS_PATH = Path("reports/metrics.json")
TARGET = "classification"
DROP_COLUMNS = ["id"]
NUMERIC_COLUMNS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
]


def load_and_clean_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python scripts/download_data.py"
        )

    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # The Kaggle/UCI CKD file contains a few tab-prefixed labels.
    df = df.replace(
        {
            "\t?": None,
            "?": None,
            "\tyes": "yes",
            "\tno": "no",
            " yes": "yes",
            "\tckd": "ckd",
        }
    )
    df[TARGET] = df[TARGET].astype(str).str.strip()

    target_values = df.pop(TARGET)
    y = target_values.map({"ckd": 1, "notckd": 0})
    if y.isna().any():
        bad_labels = sorted(target_values[y.isna()].dropna().unique())
        raise ValueError(f"Unknown target labels found: {bad_labels}")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df, y.astype(int)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = [c for c in NUMERIC_COLUMNS if c in X.columns]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    X, y = load_and_clean_data()
    pipeline = build_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_columns": list(X.columns),
            "target_mapping": {"notckd": 0, "ckd": 1},
        },
        MODEL_PATH,
    )

    metrics = {
        "dataset_rows": int(len(X)),
        "dataset_columns": int(X.shape[1]),
        "test_accuracy": round(float(test_accuracy), 4),
        "cross_validation_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cross_validation_accuracy_scores": [round(float(score), 4) for score in cv_scores],
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["notckd", "ckd"],
            output_dict=True,
        ),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.2%}")


if __name__ == "__main__":
    main()
