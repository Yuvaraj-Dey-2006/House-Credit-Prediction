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
- ☁️ Runtime download of the processed dataset and model artifacts from a private Hugging Face dataset repo, so no dataset or model files are committed to GitHub
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
|   |-- secrets.toml        # gitignored 
|   `-- assets/
|-- Artifacts/               # gitignored — downloaded at runtime from Hugging Face
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
|-- HCP kaggle Datasets/     # datasets inside it are gitignored
|   `-- final_dataset_maker.py
|-- Model Training/
|   |-- baseline_training.py
|   |-- config.py
|   |-- data_cleaning.py
|   |-- final_training.py
|   `-- tune_models.py
|-- Processed Datasets/      # gitignored — downloaded at runtime from Hugging Face
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
- 🤗 A Hugging Face account with a private dataset repo hosting the processed CSV and model artifacts (see below) — only needed if you want the app to auto-download data/artifacts instead of placing them locally yourself

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are Streamlit, pandas, NumPy, scikit-learn, joblib, XGBoost, LightGBM, CatBoost, and huggingface_hub.

## 📁 Data and Artifacts

The raw Home Credit datasets are not included in this repository because of size and competition data restrictions. Download the raw dataset directly from the official Kaggle competition page:

[Kaggle Datasets | Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview)

After downloading, place the Kaggle CSV files inside:

```text
HCP kaggle Datasets/
```

The **processed/cleaned dataset and all trained model artifacts are also excluded from version control**, since they're either derived from the restricted Kaggle competition data or too large for a plain GitHub commit. Instead of committing them, they're hosted in a **private Hugging Face dataset repo** and downloaded automatically at runtime the first time the app needs them, then cached locally:

- `Processed Datasets/final_train_cleaned.csv`
- `Artifacts/baseline_models.joblib`
- `Artifacts/preprocessors.joblib`
- `Artifacts/base_result.joblib`
- `Artifacts/best_params.joblib` (optional)
- `Artifacts/final_model.joblib` (optional, but required for the app to score with the tuned final model rather than a baseline model)

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

10. ☁️ Upload the resulting files in `Processed Datasets/` and `Artifacts/` to your private Hugging Face dataset repo if you want other deployments (or a fresh clone) to auto-download them instead of running the full pipeline again.

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

> ⚠️NOTE: You can also run the `final_test_cleaned.csv` but it will take a lot of your time due to its **300K+ data** and **230+ features** so if you want to use the `final_test_cleaned.csv`, **divide it into small proportions** and upload it on `Batch Score` to see the results.

## 🧠 Model Details

The baseline training stage compares:

- Elastic Net logistic regression through `SGDClassifier`
- XGBoost classifier
- LightGBM classifier
- CatBoost classifier

The pipeline handles class imbalance with class weights or model-specific balancing options. Evaluation is performed on a held-out evaluation split and ranked primarily by ROC-AUC.

The tuning stage uses Optuna and only searches the top two baseline models by ROC-AUC to keep the search focused. Tuning results are saved to `Artifacts/best_params.joblib`.

The final deployment model is a CatBoost classifier refit on the entire cleaned training dataset (see `Model Training/final_training.py`), using any tuned hyperparameters found in `Artifacts/best_params.joblib`.

## 📝 Notes

- ✅ The dashboard scores with the final tuned CatBoost model (`Artifacts/final_model.joblib`) when it's available, and falls back to the best-performing saved baseline model artifact otherwise.
- 🧩 Missing feature columns in uploaded batch files are filled with blank values before preprocessing.
- 🔐 No dataset or model artifact content is stored in this GitHub repository — everything under `Processed Datasets/` and `Artifacts/` is gitignored and fetched at runtime only after you ran each file in correct order.
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