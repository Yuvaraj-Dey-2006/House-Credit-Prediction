from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
CLEANED_TRAIN_PATH = ROOT / "Processed Datasets" / "final_train_cleaned.csv"
BASELINE_RESULTS_PATH = ROOT / "Artifacts" / "base_result.joblib"
BEST_PARAMS_PATH = ROOT / "Artifacts" / "best_params.joblib"
FINAL_MODEL_PATH = ROOT / "Artifacts" / "final_model.joblib"


def build_catboost_preprocessor(numeric_columns, categorical_columns):
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_scale", numeric_pipeline, numeric_columns),
            ("category_scale", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def main():
    print("Loading cleaned training data...")
    train_df = pd.read_csv(CLEANED_TRAIN_PATH)

    X_full = train_df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    y_full = train_df["TARGET"]

    numeric_columns = X_full.select_dtypes(include=np.number).columns
    categorical_columns = X_full.select_dtypes(include=["object", "category", "string"]).columns

    print(f"Rows: {len(X_full):,}")
    print(f"Features: {X_full.shape[1]:,}")
    print(f"Categorical features: {len(categorical_columns):,}")

    preprocessor = build_catboost_preprocessor(numeric_columns, categorical_columns)
    X_prepared = preprocessor.fit_transform(X_full)

    for column in categorical_columns:
        X_prepared[column] = X_prepared[column].astype(str)

    best_params = joblib.load(BEST_PARAMS_PATH) if BEST_PARAMS_PATH.exists() else {}
    catboost_params = best_params.get("catbc", {}).get("params", {})

    model = CatBoostClassifier(
        iterations=1000,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
        **catboost_params,
    )

    print("Fitting final CatBoost model on the entire cleaned training dataset...")
    model.fit(
        X_prepared,
        y_full,
        cat_features=categorical_columns.tolist(),
    )

    baseline_results = joblib.load(BASELINE_RESULTS_PATH)
    best_baseline = baseline_results.sort_values("ROC-AUC", ascending=False).iloc[0].to_dict()

    FINAL_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model_name": "CatBoost Classifier",
            "model_key": "catbc_full",
            "model_family": "catboost",
            "model": model,
            "preprocessor": preprocessor,
            "feature_names": X_full.columns.tolist(),
            "categorical_features": categorical_columns.tolist(),
            "trained_rows": len(X_full),
            "source_data": str(CLEANED_TRAIN_PATH.relative_to(ROOT)),
            "threshold": 0.48,
            "params": model.get_params(),
            "selection_metric": {
                "baseline_model": best_baseline["Models"],
                "baseline_roc_auc": best_baseline["ROC-AUC"],
            },
        },
        FINAL_MODEL_PATH,
    )
    print(f"Saved final deployment model -> {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
