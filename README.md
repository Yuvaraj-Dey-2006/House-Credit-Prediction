# House Credit Prediction

A machine learning project for predicting home-credit default risk from applicant, bureau, previous application, installment, credit card, and POS cash-balance features. The repository includes data preparation scripts, baseline model training, Optuna tuning, final model export, exploratory analysis plots, and an interactive Streamlit dashboard for applicant-level and batch risk scoring.

## 📌 Overview

This project builds a credit-risk classification workflow around the Home Credit dataset. The target is binary:

- `0`: applicant is not expected to default
- `1`: applicant is expected to default

The model pipeline focuses on imbalanced classification and evaluates models using ROC-AUC alongside accuracy, precision, recall, and F1 score. The Streamlit app loads saved model artifacts and provides an interactive interface for reviewing model performance, scoring one applicant, scoring uploaded CSV files, and viewing EDA plots.

## ✨ Features

- 🧹 End-to-end data cleaning pipeline for prepared train/test datasets
- 🤖 Baseline training for Elastic Net logistic regression, XGBoost, LightGBM, and CatBoost
- 🔁 Model-specific preprocessing pipelines saved for reproducible inference
- 🎯 Optuna hyperparameter tuning for the top baseline models by ROC-AUC
- 📦 Final CatBoost training script for deployment artifact creation
- 🖥️ Streamlit dashboard with:
  - 🔎 Single applicant scoring by applicant ID
  - 📄 Batch CSV upload and risk scoring
  - 🎚️ Adjustable high-risk probability threshold
  - 📊 Model metric comparison
  - 📈 EDA plot viewer
  - 🌗 Light/dark theme toggle

## 🗂️ Project Structure

```text
House Credit Prediction/
|-- .streamlit/
|   |-- app.py
|   |-- config.toml
|   `-- assets/
|-- Artifacts/
|   |-- baseline_models.joblib
|   |-- base_result.joblib
|   |-- best_params.joblib
|   |-- final_model.joblib
|   |-- optuna_studies.db
|   |-- preprocessors.joblib
|   `-- split_data.joblib
|-- EDA/
|   |-- home-credit-EDA.ipynb
|   `-- Plots/
|-- HCP kaggle Datasets/
|   `-- final_dataset_maker.py
|-- Model Training/
|   |-- baseline_training.py
|   |-- config.py
|   |-- data_cleaning.py
|   |-- final_training.py
|   `-- tune_models.py
|-- Processed Datasets/
|   |-- final_train.csv
|   |-- final_test.csv
|   |-- final_train_cleaned.csv
|   `-- final_test_cleaned.csv
|-- sample_batch_applicants.csv
|-- requirements.txt
`-- README.md
```

## ⚙️ Requirements

- 🐍 Python 3.10 or newer
- 📦 pip
- 💾 Local copies of the processed datasets and generated model artifacts

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are Streamlit, pandas, NumPy, scikit-learn, joblib, XGBoost, LightGBM, and CatBoost.

## 📁 Data and Artifacts

The raw Home Credit datasets are not included in this repository because of size and competition data restrictions. Download the dataset directly from the official Kaggle competition page:

https://www.kaggle.com/competitions/home-credit-default-risk/data

After downloading, place the Kaggle CSV files inside:

```text
HCP kaggle Datasets/
```

Large datasets and serialized model files are excluded from version control through `.gitignore`. To run the full workflow or dashboard, keep these files locally:

- `Processed Datasets/final_train.csv`
- `Processed Datasets/final_test.csv`
- `Processed Datasets/final_train_cleaned.csv`
- `Artifacts/baseline_models.joblib`
- `Artifacts/preprocessors.joblib`
- `Artifacts/base_result.joblib`

The Streamlit app requires the baseline model, preprocessor, and baseline result artifacts. If they are missing, run the training workflow before launching the app.

## 🚀 Reproducing the Pipeline

Run commands from the project root.

1. 📦 Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. ⬇️ Download the Home Credit dataset from Kaggle and place the CSV files inside `HCP kaggle Datasets/`.

3. 🏗️ Build the final prepared train/test datasets:

```bash
python "HCP kaggle Datasets/final_dataset_maker.py"
```

4. 📊 Run the EDA notebook and export plots to `EDA/Plots`:

```text
EDA/home-credit-EDA.ipynb
```

5. 🧭 Review or run the shared path configuration:

```bash
python "Model Training/config.py"
```

6. 🧹 Clean the prepared train/test datasets:

```bash
python "Model Training/data_cleaning.py"
```

7. 🤖 Train baseline models and save evaluation artifacts:

```bash
python "Model Training/baseline_training.py"
```

8. 🎯 Tune the top two baseline models with Optuna:

```bash
python "Model Training/tune_models.py"
```

9. 📦 Train and save the final deployment model:

```bash
python "Model Training/final_training.py"
```

## 🖥️ Running the Streamlit App

Start the dashboard from the project root:

```bash
streamlit run .streamlit/app.py
```

The app opens in the browser and provides four main views:

- 🔎 `Applicant Score`: choose an applicant ID from the loaded sample data and view default probability, risk band, and threshold decision.
- 📄 `Batch Score`: upload a CSV with matching feature columns and score many applicants at once.
- 📊 `Model Metrics`: compare baseline model performance.
- 📈 `EDA`: browse saved exploratory analysis plots from `EDA/Plots`.

A sample upload file is included at `sample_batch_applicants.csv`.

## 🧠 Model Details

The baseline training stage compares:

- Elastic Net logistic regression through `SGDClassifier`
- XGBoost classifier
- LightGBM classifier
- CatBoost classifier

The pipeline handles class imbalance with class weights or model-specific balancing options. Evaluation is performed on a held-out evaluation split and ranked primarily by ROC-AUC.

The tuning stage uses Optuna and only searches the top two baseline models by ROC-AUC to keep the search focused. Tuning results are saved to `Artifacts/best_params.joblib`.

## 📝 Notes

- ⚠️ The dashboard currently scores with the saved baseline model artifacts loaded from `Artifacts/baseline_models.joblib`.
- 🧩 Missing feature columns in uploaded batch files are filled with blank values before preprocessing.
- ✅ Model predictions are decision-support outputs and should not be used as the sole basis for real credit decisions without validation, monitoring, and fairness review.

## 📜 License

This project is licensed under the terms included in `LICENSE`.

## 👨‍💻 Author

**Yuvaraj Dey** | [Yuvaraj-Dey-2006](https://github.com/Yuvaraj-Dey-2006)

Built as a complete machine learning project for Home Credit default risk prediction, including data preparation, model training, tuning, deployment artifacts, and an interactive Streamlit dashboard.

## 🤝 How to Contribute

Contributions are welcome. To contribute:

1. Fork this repository.
2. Create a new feature branch.
3. Make your changes with clear, readable code.
4. Test the pipeline or app section affected by your change.
5. Open a pull request with a short explanation of what you improved.

Good contribution ideas include improving model performance, adding validation reports, enhancing the Streamlit UI, improving documentation, or adding deployment support.

## ⭐ Support

If this project helped you or you found it useful, please consider starring the repository. It helps the project reach more learners and developers.
