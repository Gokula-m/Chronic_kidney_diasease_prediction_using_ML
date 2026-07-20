# Chronic Kidney Disease Prediction

Machine learning project for predicting chronic kidney disease from the Kaggle dataset
`mansoordaku/ckdisease`.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Download the dataset

```powershell
python scripts/download_data.py
```

This downloads `kidney_disease.csv` from Kaggle using `kagglehub` and copies it to
`data/raw/kidney_disease.csv`.

## Train the model

```powershell
python train_model.py
```

The training script cleans missing/tab-prefixed values, imputes missing numeric and
categorical features, one-hot encodes categorical features, trains a Random Forest
classifier, and saves:

- `models/ckd_model.joblib`
- `reports/metrics.json`

Current measured result on the downloaded dataset:

- Test accuracy: 100.00%
- 5-fold cross-validation mean accuracy: 99.25%

## Predict

Predict from a CSV:

```powershell
python predict.py --csv data/raw/kidney_disease.csv
```

Predict one patient from JSON:

```powershell
python predict.py --json "{\"age\":48,\"bp\":80,\"sg\":1.02,\"al\":1,\"su\":0,\"rbc\":\"normal\",\"pc\":\"normal\",\"pcc\":\"notpresent\",\"ba\":\"notpresent\",\"bgr\":121,\"bu\":36,\"sc\":1.2,\"sod\":null,\"pot\":null,\"hemo\":15.4,\"pcv\":44,\"wc\":7800,\"rc\":5.2,\"htn\":\"yes\",\"dm\":\"yes\",\"cad\":\"no\",\"appet\":\"good\",\"pe\":\"no\",\"ane\":\"no\"}"
```

Or put the patient values in a JSON file and run:

```powershell
python predict.py --json-file sample_patient.json
```

## Run the Django web interface

```powershell
python manage.py migrate
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

Patients can enter their medical values in the form. The page shows:

- `You have CKD please have proper medical Consultation`
- `You do not have CKD`

## Note

This is an educational ML project. It should not be used as a medical diagnosis
tool without clinical validation.
