from pathlib import Path
import argparse
import json

import joblib
import pandas as pd


MODEL_PATH = Path("models/ckd_model.joblib")
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


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run: python train_model.py")
    return joblib.load(path)


def clean_rows(rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = rows.copy()
    rows = rows.replace(
        {
            "\t?": None,
            "?": None,
            "\tyes": "yes",
            "\tno": "no",
            " yes": "yes",
            "\tckd": "ckd",
        }
    )

    if "classification" in rows.columns:
        rows = rows.drop(columns=["classification"])
    if "id" in rows.columns:
        rows = rows.drop(columns=["id"])

    for column in feature_columns:
        if column not in rows.columns:
            rows[column] = None
    rows = rows[feature_columns]

    for column in NUMERIC_COLUMNS:
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")

    return rows


def predict_from_csv(csv_path: Path) -> pd.DataFrame:
    bundle = load_model()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    rows = pd.read_csv(csv_path)
    rows = clean_rows(rows, feature_columns)

    predictions = model.predict(rows)
    probabilities = model.predict_proba(rows)[:, 1]

    result = rows.copy()
    result["prediction"] = ["ckd" if value == 1 else "notckd" for value in predictions]
    result["ckd_probability"] = probabilities.round(4)
    return result


def predict_from_json(payload: str) -> dict:
    bundle = load_model()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    values = json.loads(payload)
    row = pd.DataFrame([{column: values.get(column) for column in feature_columns}])
    row = clean_rows(row, feature_columns)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    return {
        "prediction": "ckd" if prediction == 1 else "notckd",
        "ckd_probability": round(probability, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict chronic kidney disease risk.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=Path, help="CSV file containing patient rows.")
    group.add_argument("--json", help="Single patient JSON object.")
    group.add_argument("--json-file", type=Path, help="JSON file containing one patient object.")
    args = parser.parse_args()

    if args.csv:
        print(predict_from_csv(args.csv).to_string(index=False))
    elif args.json_file:
        print(json.dumps(predict_from_json(args.json_file.read_text(encoding="utf-8")), indent=2))
    else:
        print(json.dumps(predict_from_json(args.json), indent=2))


if __name__ == "__main__":
    main()
