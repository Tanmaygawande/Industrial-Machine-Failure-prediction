# Industrial Machine Failure Prediction

**Tanmay Gawande**  
**Learn Depth Academy LLP — Track 1 Final Capstone, Problem 21**

## Problem selected
**Industrial Machine Failure Prediction** — Binary Classification.

The supplied capstone PDF defines this problem as predicting whether a machine is likely to experience a failure within a defined operating window. It specifies temperature, vibration, pressure, runtime and maintenance gap as expected inputs, and asks for a foundational classifier, evaluation, a Streamlit prototype and supporting documentation.

The supplied brief also specifies accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC where appropriate, plus source code, notebook, trained model, requirements, README, Streamlit application, technical paper and presentation.

## Dataset
The academy PDF does not include an assigned dataset. Therefore, this package contains a **reproducible synthetic dataset** for the educational prototype rather than pretending an external dataset was supplied.

- Rows: 5,000
- Inputs: temperature, vibration, pressure, runtime, maintenance_gap
- Target: failure (0/1)
- Random seed: 42

## Models and results
Two foundational models were compared.

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC_AUC |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| Logistic Regression |      0.912 |      0.7553 |   0.9181 | 0.8288 |    0.9794 |
| Decision Tree       |      0.787 |      0.5263 |   0.819  | 0.6408 |    0.8814 |

The final model is selected using F1-score first and ROC-AUC second. The saved pipeline is used by the Streamlit application.

## Run the project

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
python src/train_model.py
```

### Notebook
```bash
jupyter notebook
```
Open `notebooks/Industrial_Machine_Failure_Prediction.ipynb`.

### Streamlit
```bash
streamlit run src/app.py
```

## Project structure
```text
Industrial_Machine_Failure_Prediction_Tanmay_Gawande/
├── data/machine_failure_data.csv
├── models/machine_failure_model.joblib
├── notebooks/Industrial_Machine_Failure_Prediction.ipynb
├── reports/model_comparison.csv
├── reports/technical_paper.pdf
├── presentation/Industrial_Machine_Failure_Prediction.pptx
├── src/train_model.py
├── src/eda.py
├── src/app.py
├── requirements.txt
├── SOURCE_NOTE.txt
└── README.md
```

## Capstone coverage
This package covers the implementation-side deliverables from the supplied brief. The final demo/viva should be performed by the student, who should be able to explain the dataset, preprocessing, model choice, metrics and limitations.

## Limitation
This is an educational prototype. The synthetic dataset and model must not be used for real industrial maintenance decisions.
