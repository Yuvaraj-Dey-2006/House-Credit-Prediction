"""
Shared paths for the House Credit Prediction pipeline.
Every stage (data_cleaning.py, training.py, tuning.py, model_saving.py)
imports from here instead of hardcoding paths — one place to change them.
"""

from pathlib import Path

# ── Raw input data ──
RAW_TRAIN_PATH = Path("Processed Datasets/final_train.csv")
RAW_TEST_PATH = Path("Processed Datasets/final_test.csv")

# ── Artifacts directory (created if missing) ──
ARTIFACTS_DIR = Path("Artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ── Stage 1: data_cleaning.py output ──
CLEANED_TRAIN_PATH = Path("Processed Datasets/final_train_cleaned.csv")
CLEANED_TEST_PATH = Path("Processed Datasets/final_test_cleaned.csv")

# ── Stage 2: training.py output ──
SPLIT_DATA_PATH = ARTIFACTS_DIR / "split_data.joblib"          # X_train, X_eval, y_train, y_eval
PREPROCESSORS_PATH = ARTIFACTS_DIR / "preprocessors.joblib"    # pp_elasticnet, pp_xg, pp_lgbm, pp_catbc
BASELINE_MODELS_PATH = ARTIFACTS_DIR / "baseline_models.joblib"
BASELINE_RESULTS_PATH = ARTIFACTS_DIR / "base_result.joblib"   # metrics DataFrame

# ── Stage 3: tuning.py output ──
TUNED_STUDIES_PATH = ARTIFACTS_DIR / "tuned_studies.joblib"

# ── Stage 4: model_saving.py output ──
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
