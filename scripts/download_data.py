from pathlib import Path
import shutil

import kagglehub


DATASET_SLUG = "mansoordaku/ckdisease"
DATA_FILE = "kidney_disease.csv"
OUTPUT_DIR = Path("data/raw")
OUTPUT_PATH = OUTPUT_DIR / DATA_FILE


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    source_path = dataset_dir / DATA_FILE

    if not source_path.exists():
        csv_files = list(dataset_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in Kaggle dataset: {dataset_dir}")
        source_path = csv_files[0]

    shutil.copy2(source_path, OUTPUT_PATH)
    print(f"Downloaded Kaggle dataset '{DATASET_SLUG}' to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
