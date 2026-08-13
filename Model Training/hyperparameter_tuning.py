import warnings
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import (
    SPLIT_DATA_PATH,
    BEST_PARAMS_PATH,
)


warnings.filterwarnings("ignore")


RANDOM_STATE = 42
N_SPLITS = 3


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TRAINING SPLIT
# ══════════════════════════════════════════════════════════════════════════════

split_data = joblib.load(SPLIT_DATA_PATH)

X_train = split_data["X_train"]
X_eval = split_data["X_eval"]

y_train = split_data["y_train"]
y_eval = split_data["y_eval"]


# IMPORTANT:
# Optuna uses ONLY X_train/y_train.
# X_eval/y_eval remains untouched for final evaluation.


numeric_columns = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()


neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()

scale_pos_weight = (
    neg_count / max(pos_count, 1)
)


cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

def create_elasticnet_preprocessor():

    num_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                RobustScaler(
                    quantile_range=(0.25, 0.75)
                ),
            ),
        ]
    )

    cat_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "num",
                num_pipeline,
                numeric_columns,
            ),
            (
                "cat",
                cat_pipeline,
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=False,
    )


def create_xgb_preprocessor():

    num_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    cat_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "num",
                num_pipeline,
                numeric_columns,
            ),
            (
                "cat",
                cat_pipeline,
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=False,
    )