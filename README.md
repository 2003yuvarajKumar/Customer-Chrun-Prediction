# Customer Churn Prediction (Telco)

Predicts which telecom customers are likely to leave (churn) using the Telco Customer Churn dataset and a Random Forest classifier.

Code adapted from the GeeksforGeeks tutorial
[Customer Churn Analysis Prediction - Python](https://www.geeksforgeeks.org/machine-learning/python-customer-churn-analysis-prediction/).

## Project structure

```
.
├── churn_prediction.py   # preprocessing, training, evaluation
├── download_data.py      # downloads the dataset into data/
├── requirements.txt
├── data/                 # dataset goes here (git-ignored)
└── outputs/              # generated plots and metrics
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
python churn_prediction.py
```

You can also pass your own CSV: `python churn_prediction.py path/to/file.csv`.

## Pipeline

1. Load the Telco Customer Churn CSV and inspect missing values and class balance.
2. Clean `TotalCharges` (blank strings → numeric, missing filled with the median).
3. Label-encode categorical columns, drop `customerID`.
4. 80/20 train/test split, standardize features (scaler fit on train only).
5. Train a `RandomForestClassifier`.
6. Evaluate with accuracy, a classification report, and a confusion matrix.

## Outputs

Saved to `outputs/`: `churn_distribution.png`, `confusion_matrix.png`, `metrics.txt`.

The original tutorial reports about **78% accuracy**. Its confusion matrix shows roughly half of the churners being missed, so accuracy alone overstates performance. Exact numbers may differ slightly on your machine.

## Ideas for improvement

- Handle class imbalance (class weights, SMOTE, threshold tuning).
- Compare against logistic regression and gradient boosting.
- Use one-hot encoding for nominal features and cross-validated hyperparameter search.
- Add EDA on churn drivers (contract type, tenure, monthly charges).
- Report recall, F1, and ROC-AUC for the churn class.

## Dataset

Telco Customer Churn (IBM sample dataset), as linked from the GeeksforGeeks tutorial.
