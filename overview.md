 Chronic Kidney Disease Prediction Project Overview

This project predicts whether a patient has Chronic Kidney Disease (CKD) using machine learning and provides a Django web interface where a user can enter medical values and see the prediction result.

The final output shown in the web app is:

- `You have CKD please have proper medical Consultation`
- `You do not have CKD`

This project has two major parts:

1. Machine Learning part: dataset download, preprocessing, model training, evaluation, and saving the trained model.
2. Django Web part: patient input form, loading the trained model, prediction, and displaying the result.

## Project Goal

The goal is to build a supervised machine learning classification model that predicts one of two classes:

- `ckd`: patient has chronic kidney disease
- `notckd`: patient does not have chronic kidney disease

This is a binary classification problem because there are only two possible outputs.

## Dataset

The dataset is downloaded from Kaggle:

`mansoordaku/ckdisease`

The downloaded file is:

`data/raw/kidney_disease.csv`

The dataset contains 400 patient records. After removing the `id` column and target column, the model uses 24 input features.

Important medical features include:

- `age`: patient age
- `bp`: blood pressure
- `sg`: specific gravity
- `al`: albumin
- `su`: sugar
- `rbc`: red blood cells
- `pc`: pus cell
- `pcc`: pus cell clumps
- `ba`: bacteria
- `bgr`: blood glucose random
- `bu`: blood urea
- `sc`: serum creatinine
- `sod`: sodium
- `pot`: potassium
- `hemo`: hemoglobin
- `pcv`: packed cell volume
- `wc`: white blood cell count
- `rc`: red blood cell count
- `htn`: hypertension
- `dm`: diabetes mellitus
- `cad`: coronary artery disease
- `appet`: appetite
- `pe`: pedal edema
- `ane`: anemia

The target column is:

`classification`

It contains:

- `ckd`
- `notckd`

## Why Preprocessing Is Needed

Real-world datasets usually contain missing values, text values, and inconsistent formatting. This dataset has examples like:

- `?`
- `\t?`
- `\tyes`
- `\tno`
- `\tckd`

These values can break the model if they are not cleaned.

In this project, preprocessing does the following:

1. Removes the `id` column because it is only an identifier and has no medical meaning.
2. Cleans tab-prefixed values like `\tyes`, `\tno`, and `\tckd`.
3. Converts missing symbols like `?` and `\t?` into missing values.
4. Converts numeric columns into real numeric data types.
5. Fills missing numeric values using median imputation.
6. Fills missing categorical values using most frequent value imputation.
7. Converts categorical columns into numeric form using one-hot encoding.

## Numeric and Categorical Features

Machine learning models need numeric input. Some columns are already numeric, while others are text categories.

Numeric columns:

- `age`
- `bp`
- `sg`
- `al`
- `su`
- `bgr`
- `bu`
- `sc`
- `sod`
- `pot`
- `hemo`
- `pcv`
- `wc`
- `rc`

Categorical columns:

- `rbc`
- `pc`
- `pcc`
- `ba`
- `htn`
- `dm`
- `cad`
- `appet`
- `pe`
- `ane`

For numeric columns, missing values are filled using the median.

For categorical columns, missing values are filled using the most frequent category.

Categorical values are converted using one-hot encoding. For example, a column like `htn` with values `yes` and `no` becomes separate numeric columns internally.

## Algorithm Used

The algorithm used is:

`RandomForestClassifier`

Random Forest is an ensemble learning algorithm. It builds many decision trees and combines their predictions.

In this project:

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)
```

Meaning:

- `n_estimators=300`: the model builds 300 decision trees.
- `random_state=42`: keeps the result reproducible.
- `class_weight="balanced"`: helps the model handle class imbalance by giving suitable weight to each class.

## Why Random Forest Was Selected

Random Forest is suitable for this project because:

1. It works well for tabular medical datasets.
2. It can handle both simple and complex patterns.
3. It is less likely to overfit than a single decision tree.
4. It usually gives strong accuracy without too much tuning.
5. It works well with preprocessed numeric and one-hot encoded categorical features.
6. It can handle non-linear relationships between symptoms and disease outcome.

Interview answer:

I selected Random Forest because CKD prediction is a tabular binary classification problem with both numeric and categorical medical features. Random Forest is robust, handles non-linear relationships well, reduces overfitting by combining multiple decision trees, and achieved high accuracy on this dataset.

## Model Pipeline

The model is saved as a complete scikit-learn pipeline.

The pipeline contains:

1. Numeric preprocessing:
   - median imputation
   - standard scaling
2. Categorical preprocessing:
   - most frequent imputation
   - one-hot encoding
3. Classifier:
   - Random Forest

This is useful because the same preprocessing steps used during training are also used during prediction.

## Model Training Flow

File:

`train_model.py`

Steps:

1. Load `data/raw/kidney_disease.csv`.
2. Drop the `id` column.
3. Clean inconsistent values.
4. Separate input features `X` and target `y`.
5. Convert target labels:
   - `ckd` becomes `1`
   - `notckd` becomes `0`
6. Build preprocessing pipeline.
7. Split dataset into training and testing sets.
8. Train Random Forest model.
9. Evaluate model using accuracy, confusion matrix, and classification report.
10. Save model to `models/ckd_model.joblib`.
11. Save metrics to `reports/metrics.json`.

## Train/Test Split

The dataset is split like this:

```python
test_size=0.2
random_state=42
stratify=y
```

Meaning:

- 80% of data is used for training.
- 20% of data is used for testing.
- `random_state=42` makes the split reproducible.
- `stratify=y` keeps the same ratio of CKD and non-CKD classes in both train and test data.

## Cross Validation

The project uses 5-fold cross-validation:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Cross-validation divides the dataset into 5 parts. The model trains on 4 parts and tests on 1 part. This repeats 5 times.

The cross-validation scores are:

```text
0.9875
1.0
0.9875
0.9875
1.0
```

Mean cross-validation accuracy:

```text
99.25%
```

Interview answer:

I used cross-validation to check that the model performs consistently across different subsets of the data and is not only performing well on one lucky train-test split.

## Model Accuracy

The project achieved:

```text
Test accuracy: 100.00%
5-fold cross-validation mean accuracy: 99.25%
```

Accuracy means:

```text
Correct predictions / Total predictions
```

In the test set, there were 80 records. The model predicted all 80 correctly.

Important interview note:

Even though the accuracy is very high, this is a small dataset with only 400 records. In real medical systems, the model must be tested on larger, more diverse clinical data before actual medical use.

## Confusion Matrix

From `reports/metrics.json`:

```json
[
  [30, 0],
  [0, 50]
]
```

This means:

```text
                 Predicted notckd    Predicted ckd
Actual notckd          30                0
Actual ckd              0               50
```

Explanation:

- 30 non-CKD patients were correctly predicted as non-CKD.
- 50 CKD patients were correctly predicted as CKD.
- 0 non-CKD patients were wrongly predicted as CKD.
- 0 CKD patients were wrongly predicted as non-CKD.

Terms:

- True Positive: CKD patient predicted as CKD.
- True Negative: non-CKD patient predicted as non-CKD.
- False Positive: non-CKD patient predicted as CKD.
- False Negative: CKD patient predicted as non-CKD.

In medical prediction, false negatives are especially dangerous because a patient who actually has CKD may be told they do not have it.

## Classification Report

The classification report contains:

- precision
- recall
- f1-score
- support

Current report:

```json
"notckd": {
  "precision": 1.0,
  "recall": 1.0,
  "f1-score": 1.0,
  "support": 30.0
},
"ckd": {
  "precision": 1.0,
  "recall": 1.0,
  "f1-score": 1.0,
  "support": 50.0
}
```

### Precision

Precision answers:

When the model predicts a class, how often is it correct?

For CKD:

```text
Precision = Correct CKD predictions / All predicted CKD cases
```

Precision of `1.0` means every patient predicted as CKD was actually CKD in the test set.

### Recall

Recall answers:

Out of all actual patients in a class, how many did the model find correctly?

For CKD:

```text
Recall = Correct CKD predictions / Actual CKD cases
```

Recall of `1.0` means all actual CKD patients in the test set were identified correctly.

### F1-Score

F1-score is the balance between precision and recall.

It is useful when we want one score that considers both false positives and false negatives.

F1-score of `1.0` means perfect precision and recall in the test set.

### Support

Support means the number of actual samples in each class.

In this project:

- `notckd` support is 30
- `ckd` support is 50

This means the test set had 30 non-CKD records and 50 CKD records.

### Macro Average

Macro average calculates the average score across both classes equally.

It treats `ckd` and `notckd` as equally important, regardless of how many samples each class has.

### Weighted Average

Weighted average calculates the average score based on the number of samples in each class.

Classes with more samples have more influence.

## Reason for Important Files

### `data/raw/kidney_disease.csv`

This is the raw dataset downloaded from Kaggle.

It contains patient medical data and the target label `classification`.

### `scripts/download_data.py`

This script downloads the CKD dataset from Kaggle using `kagglehub`.

It copies the dataset into:

`data/raw/kidney_disease.csv`

This makes the dataset download reproducible.

### `train_model.py`

This is the main machine learning training file.

It:

- loads the dataset
- cleans the data
- preprocesses numeric and categorical columns
- trains the Random Forest model
- evaluates the model
- saves the trained model
- saves metrics

### `models/ckd_model.joblib`

This is the saved trained model file.

`joblib` is used because it is efficient for saving and loading scikit-learn models.

The file contains:

- trained Random Forest pipeline
- preprocessing steps
- feature column order
- target mapping

The Django app loads this file to make predictions.

### `reports/metrics.json`

This file stores model evaluation results in JSON format.

JSON means JavaScript Object Notation. It is a structured data format that is easy for humans to read and easy for programs to parse.

This file contains:

- dataset row count
- dataset column count
- test accuracy
- cross-validation scores
- confusion matrix
- classification report

Reason for using JSON:

- stores metrics in a structured way
- easy to read
- easy to use later in dashboards, reports, or APIs
- keeps results separate from code

### `sample_patient.json`

This file contains one sample patient's medical values.

It is used to test prediction from JSON input:

```powershell
python predict.py --json-file sample_patient.json
```

Reason for this file:

- makes CLI testing easier
- avoids PowerShell JSON escaping problems
- provides an example input format

### `predict.py`

This is a command-line prediction script.

It can predict using:

- CSV input
- JSON text
- JSON file

Examples:

```powershell
python predict.py --csv data/raw/kidney_disease.csv
python predict.py --json-file sample_patient.json
```

This file is useful for testing the model without opening the Django website.

### `requirements.txt`

This file lists Python packages needed for the project.

It includes:

- `pandas`
- `scikit-learn`
- `joblib`
- `kagglehub`
- `Django`

Install them using:

```powershell
python -m pip install -r requirements.txt
```

### `README.md`

This is the basic project guide.

It explains:

- how to install dependencies
- how to download data
- how to train the model
- how to predict
- how to run the Django web app

### `db.sqlite3`

This is Django's default SQLite database file.

It was created after running:

```powershell
python manage.py migrate
```

For this project, the prediction does not require storing patients in the database. However, Django creates this database for built-in apps like:

- admin
- auth
- sessions
- contenttypes

Interview answer:

SQLite is used here because this is a small local demo project. It is simple, file-based, and does not need separate database server setup.

### `manage.py`

This is Django's command-line utility.

It is used to run Django commands like:

```powershell
python manage.py runserver
python manage.py migrate
python manage.py check
```

## Django Project Files

### `ckd_web/settings.py`

This is the main Django settings file.

It contains:

- installed apps
- middleware
- database settings
- template settings
- static file settings
- allowed hosts

Important setting:

```python
INSTALLED_APPS = [
    'predictor',
    ...
]
```

This registers the `predictor` app in Django.

Another important setting:

```python
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]
```

This allows the app to run locally and also allows Django test client checks.

### `ckd_web/urls.py`

This file maps main project URLs.

It includes the predictor app at the homepage:

```python
path('', include('predictor.urls'))
```

So when the user opens:

```text
http://127.0.0.1:8000/
```

Django sends the request to the predictor app.

### `ckd_web/wsgi.py`

This is used when deploying Django using WSGI servers.

For local development, we usually do not edit this file.

### `ckd_web/asgi.py`

This is used when deploying Django using ASGI servers.

ASGI supports asynchronous applications.

For this local project, we usually do not edit this file.

## Django App Files

The app name is:

`predictor`

### `predictor/forms.py`

This file defines the patient input form.

It uses Django forms to create fields such as:

- age
- blood pressure
- serum creatinine
- hemoglobin
- hypertension
- diabetes mellitus
- anemia

Numeric fields use `FloatField`.

Categorical fields use `ChoiceField`.

Example:

```python
htn = forms.ChoiceField(label="Hypertension", choices=YES_NO_CHOICES)
```

Reason for using Django forms:

- validates input
- creates form fields easily
- handles submitted data safely
- keeps form logic separate from view logic

### `predictor/ml.py`

This file connects Django with the trained ML model.

It:

- loads `models/ckd_model.joblib`
- receives patient input
- creates a pandas DataFrame
- calls `model.predict()`
- calls `model.predict_proba()`
- returns the final message and probability

Important function:

```python
predict_ckd(patient_data)
```

This function returns:

- prediction class
- CKD probability
- display message
- result status

It uses `lru_cache` so the model is loaded only once and reused.

Interview answer:

I separated ML prediction logic into `ml.py` so the Django view does not directly handle model loading. This improves code organization and avoids loading the model again for every request.

### `predictor/views.py`

This file handles web requests.

Flow:

1. User opens the page with GET request.
2. Django shows an empty form.
3. User submits the form with POST request.
4. Django validates the form.
5. The cleaned data is sent to `predict_ckd()`.
6. The result is sent back to the template.
7. The template displays the CKD or non-CKD message.

### `predictor/urls.py`

This file maps app URLs.

It maps the home page:

```python
path("", views.home, name="home")
```

### `predictor/templates/predictor/home.html`

This is the HTML file for the user interface.

It contains:

- page title
- patient input form
- grouped sections for medical values
- submit button
- result display area

The result area changes based on prediction:

- red style for CKD
- green style for non-CKD

### `predictor/static/predictor/styles.css`

This file contains the CSS styling for the Django page.

It controls:

- layout
- colors
- form grid
- result panel
- mobile responsiveness
- buttons
- input design

### `predictor/models.py`

This file is normally used for Django database models.

In this project, it is not used because we are not saving patient records to the database.

### `predictor/admin.py`

This file is used to register database models in Django admin.

It is currently not important because there are no custom database models.

### `predictor/tests.py`

This file is reserved for Django tests.

Future tests can be added here to test:

- homepage loads
- form submission works
- CKD result displays correctly

### `predictor/apps.py`

This file contains Django app configuration.

It tells Django about the `predictor` app.

### `predictor/migrations/`

This folder stores database migration files for the app.

Since this app has no database models, only the initial migration package exists.

### `__pycache__` folders

These are automatically created by Python.

They store compiled Python bytecode files to make imports faster.

They are not written manually and are not important for explaining the project logic.

## How the Django Prediction Works

Complete flow:

1. User opens `http://127.0.0.1:8000/`.
2. Django routes the request through `ckd_web/urls.py`.
3. The request goes to `predictor/urls.py`.
4. The `home` view in `predictor/views.py` runs.
5. The patient form from `predictor/forms.py` is displayed.
6. User enters medical values and submits the form.
7. Django validates the form data.
8. Valid data is sent to `predictor/ml.py`.
9. `ml.py` loads the saved model from `models/ckd_model.joblib`.
10. The data is converted into a pandas DataFrame.
11. The model predicts CKD or non-CKD.
12. The prediction message is sent to the template.
13. `home.html` displays the result.

## Commands Used

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download dataset:

```powershell
python scripts/download_data.py
```

Train model:

```powershell
python train_model.py
```

Run command-line prediction:

```powershell
python predict.py --json-file sample_patient.json
```

Run Django migrations:

```powershell
python manage.py migrate
```

Run Django web app:

```powershell
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Interview Explanation

You can explain the project like this:

This is a chronic kidney disease prediction system built using machine learning and Django. I used a Kaggle CKD dataset containing 400 patient records with medical attributes such as blood pressure, serum creatinine, hemoglobin, blood urea, diabetes, hypertension, and anemia. The target variable is `classification`, which has two classes: CKD and not CKD.

First, I cleaned the dataset by handling missing values and inconsistent labels such as tab-prefixed values. I separated numeric and categorical columns. Numeric missing values were filled using median imputation, and categorical missing values were filled using the most frequent value. Categorical columns were converted into numeric format using one-hot encoding.

For the algorithm, I used Random Forest Classifier because it performs well on tabular data, handles non-linear relationships, reduces overfitting compared to a single decision tree, and gave strong accuracy on this dataset. The model was trained using an 80-20 train-test split and evaluated using accuracy, confusion matrix, classification report, and 5-fold cross-validation.

The trained model achieved 100% test accuracy and 99.25% mean cross-validation accuracy. The classification report showed precision, recall, and F1-score of 1.0 for both CKD and non-CKD classes on the test set. The model was saved using joblib so it can be reused without retraining.

Then I built a Django web interface where users can enter patient medical data. The Django form sends the input to the backend, where the saved model is loaded and used for prediction. The result is displayed clearly as either the patient has CKD and should consult a doctor, or the patient does not have CKD.

## Possible Interview Questions and Answers

### What type of machine learning problem is this?

It is a supervised binary classification problem because the model is trained using labeled data and predicts one of two classes: CKD or not CKD.

### Why did you use Random Forest?

Random Forest works well for tabular datasets, handles non-linear relationships, reduces overfitting by combining multiple decision trees, and gave high accuracy on this CKD dataset.

### What is preprocessing?

Preprocessing is preparing raw data for the model. In this project, preprocessing includes cleaning inconsistent values, handling missing data, converting numeric columns, imputing missing values, and encoding categorical variables.

### Why did you use one-hot encoding?

Machine learning models require numeric input. One-hot encoding converts categorical text values like `yes`, `no`, `normal`, and `abnormal` into numeric columns.

### Why did you save the model?

The model is saved so it can be loaded later for prediction without retraining every time. The Django app uses the saved model file directly.

### What is `metrics.json`?

It is a JSON file that stores evaluation results such as accuracy, cross-validation scores, confusion matrix, and classification report.

### What is precision?

Precision tells how many predicted positive cases are actually correct. For CKD, it tells how many patients predicted as CKD truly had CKD.

### What is recall?

Recall tells how many actual positive cases were correctly found by the model. For CKD, it tells how many real CKD patients were correctly detected.

### What is F1-score?

F1-score is the harmonic mean of precision and recall. It gives one balanced score when both false positives and false negatives matter.

### What is support?

Support is the number of actual samples for each class in the test set.

### What is the purpose of Django in this project?

Django provides the web interface. It allows users to enter patient data through a form, sends the data to the backend model, and displays the prediction result.

### Does this project use a database for prediction?

No. The prediction uses the saved machine learning model. The SQLite database exists because Django creates it for built-in features like admin, auth, and sessions.

### Can this be used in hospitals directly?

No. This is an educational project. For real medical use, the model must be validated with larger clinical datasets, expert review, privacy controls, and regulatory approval.

## Limitations

1. The dataset is small with only 400 records.
2. The model may not generalize to all real-world patients.
3. Missing values are estimated using imputation.
4. The model is not clinically validated.
5. The web app does not store patient history.
6. The result should not replace medical diagnosis.

## Future Improvements

1. Use a larger and more diverse dataset.
2. Add feature importance visualization.
3. Add patient record storage with user permission.
4. Add authentication for doctors/admins.
5. Add PDF report generation.
6. Compare multiple algorithms like Logistic Regression, SVM, XGBoost, and Gradient Boosting.
7. Add automated tests for the Django app.
8. Deploy the application online.

## Final Summary

This project is a complete machine learning and Django web application for chronic kidney disease prediction. It includes dataset download, data cleaning, preprocessing, Random Forest model training, evaluation, model saving, command-line prediction, and a user-friendly Django interface. The project demonstrates practical knowledge of machine learning, model evaluation, file organization, and web integration.
