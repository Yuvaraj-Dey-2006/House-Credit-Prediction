"""
██╗███╗   ███╗██████╗  ██████╗ ██████╗ ████████╗██╗███╗   ██╗ ██████╗    ██╗     ██╗██████╗ ██████╗  █████╗ ██████╗ ██╗███████╗███████╗
██║████╗ ████║██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██║████╗  ██║██╔════╝    ██║     ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝
██║██╔████╔██║██████╔╝██║   ██║██████╔╝   ██║   ██║██╔██╗ ██║██║  ███╗   ██║     ██║██████╔╝██████╔╝███████║██████╔╝██║█████╗  ███████╗
██║██║╚██╔╝██║██╔═══╝ ██║   ██║██╔══██╗   ██║   ██║██║╚██╗██║██║   ██║   ██║     ██║██╔══██╗██╔══██╗██╔══██║██╔══██╗██║██╔══╝  ╚════██║
██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║   ██║   ██║██║ ╚████║╚██████╔╝   ███████╗██║██████╔╝██║  ██║██║  ██║██║  ██║██║███████╗███████║
╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝    ╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝
"""

# Accessing data
import numpy as np
import pandas as pd

# Paths and dirs
from pathlib import Path
import os
import warnings

warnings.filterwarnings("ignore")
# Train test split
from sklearn.model_selection import train_test_split

# Pipeline
from sklearn.pipeline import Pipeline

# scaling, encoding & imputing features
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# ╔══╗   ╔══╗ ╔═══════╗ ╔═══════╗ ╔════════╗╔══╗     ╔═══════╗
# ║  ╚╗ ╔╝  ║╔╝ ╔═══╗ ╚╗║  ╔══╗ ╚╗║ ╔══════╝║  ║     ║ ╔═════╝
# ║ ╔╗╚═╝╔╗ ║║  ║   ║  ║║  ║  ║  ║║ ╚═══╗   ║  ║     ║ ╚═════╗
# ║ ║╚╗ ╔╝║ ║║  ║   ║  ║║  ║  ║  ║║ ╔═══╝   ║  ║     ╚═════╗ ║
# ║ ║ ╚═╝ ║ ║╚╗ ╚═══╝ ╔╝║  ╚══╝ ╔╝║ ╚══════╗║  ╚════╗╔═════╝ ║
# ╚═╝     ╚═╝ ╚═══════╝ ╚═══════╝ ╚════════╝╚═══════╝╚═══════╝

from sklearn.linear_model import (
    LogisticRegression,
)  
from xgboost import (
    XGBClassifier,
)  
from lightgbm import (
    LGBMClassifier,
)  

from catboost import (          
    CatBoostClassifier,
)  


# hyperparametr tuner
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
# validation
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Performance metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

# for attractive terminal outputs
from rich.console import Console

console = Console()
# For progress bar
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

"""
██████╗  █████╗ ████████╗ █████╗     ██╗      ██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║     ██╔═══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝
██║  ██║███████║   ██║   ███████║    ██║     ██║   ██║███████║██║  ██║██║██╔██╗ ██║██║  ███╗
██║  ██║██╔══██║   ██║   ██╔══██║    ██║     ██║   ██║██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
██████╔╝██║  ██║   ██║   ██║  ██║    ███████╗╚██████╔╝██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝
"""

# Training dataset
train_df = pd.read_csv(r"Processed Datasets/final_train.csv")
# Testing dataset
test_df = pd.read_csv(r"Processed Datasets/final_test.csv")

"""
 ██████╗ ██╗   ██╗████████╗██╗     ██╗███████╗██████╗      █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
██╔═══██╗██║   ██║╚══██╔══╝██║     ██║██╔════╝██╔══██╗    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
██║   ██║██║   ██║   ██║   ██║     ██║█████╗  ██████╔╝    ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
██║   ██║██║   ██║   ██║   ██║     ██║██╔══╝  ██╔══██╗    ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
╚██████╔╝╚██████╔╝   ██║   ███████╗██║███████╗██║  ██║    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
 ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
"""

numeric_cols = train_df.select_dtypes(include="number")

summary = []

for col in numeric_cols.columns:
    # IQR calculation
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # outlier detection
    outlier_mask = (train_df[col] < lower) | (train_df[col] > upper)
    # outlier count
    outlier_count = outlier_mask.sum()
    # outlier percentage
    outlier_pct = round(outlier_count / len(train_df) * 100, 2)
    # Skewed
    skew = round(train_df[col].skew(), 2)
    # missing value percentage
    missing_pct = round(train_df[col].isna().mean() * 100, 2)
    # min value
    minimum = train_df[col].min()
    # max value
    maximum = train_df[col].max()
    # No. of unique values
    unique = train_df[col].nunique()

    # ---------- Decision ----------
    if col == "TARGET":
        decision = "Target"
    elif col.startswith("SK_ID"):
        decision = "Ignore (ID)"
    elif train_df[col].nunique() <= 2:
        decision = "Binary"
    elif col == "DAYS_EMPLOYED":
        decision = "Replace sentinel"
    elif outlier_pct < 5:
        decision = "Keep"
    elif outlier_pct < 20:
        decision = "Review"
    else:
        decision = "Investigate"
    # the outler summary
    summary.append(
        {
            "Feature": col,
            "Outlier %": outlier_pct,
            "Skew": skew,
            "Missing %": missing_pct,
            "Min": minimum,
            "Max": maximum,
            "Unique": unique,
            "Decision": decision,
        }
    )
# converting the outlier summary to dataframe
outlier_summary = (
    pd.DataFrame(summary)
    .sort_values("Outlier %", ascending=False)
    .reset_index(drop=True)
)

console.print(
f"""\n\n
[bold green]______________________________________ [bold #C7009D]OUTLIER ANALYSIS[/][bold green] ______________________________________[/]\n"
  {outlier_summary}\n
[bold green]___________________________________________________________________________________________________________________________[/]
"""
)

"""
██████╗  █████╗ ██████╗     ██████╗ ██╗   ██╗████████╗██╗     ██╗███████╗██████╗     ██████╗ ███████╗███╗   ███╗ ██████╗ ██╗   ██╗ █████╗ ██╗     
██╔══██╗██╔══██╗██╔══██╗   ██╔═══██╗██║   ██║╚══██╔══╝██║     ██║██╔════╝██╔══██╗    ██╔══██╗██╔════╝████╗ ████║██╔═══██╗██║   ██║██╔══██╗██║     
██████╔╝███████║██║  ██║   ██║   ██║██║   ██║   ██║   ██║     ██║█████╗  ██████╔╝    ██████╔╝█████╗  ██╔████╔██║██║   ██║██║   ██║███████║██║     
██╔══██╗██╔══██║██║  ██║   ██║   ██║██║   ██║   ██║   ██║     ██║██╔══╝  ██╔══██╗    ██╔══██╗██╔══╝  ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██╔══██║██║     
██████╔╝██║  ██║██████╔╝   ╚██████╔╝╚██████╔╝   ██║   ███████╗██║███████╗██║  ██║    ██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ██║  ██║███████╗
╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚══════╝
"""

# Replace known sentinel values
train_df.loc[train_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
test_df.loc[test_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

# Replace infinities
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove impossible values
money_cols = [
    col for col in train_df.columns if col.startswith("AMT_")
]  # Negative monetary values

for col in money_cols:
    train_df.loc[train_df[col] < 0, col] = np.nan
    test_df.loc[test_df[col] < 0, col] = np.nan

# Remove impossible counts
count_cols = [
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "bureau_loan_count",
    "bureau_active_loans",
    "bureau_closed_loans",
    "bureau_sold_loans",
    "bureau_bad_debt_loans",
    "bureau_credit_prolong_count",
    "prev_application_count",
    "prev_approved_count",
    "prev_refused_count",
    "prev_canceled_count",
    "prev_unused_offer_count",
    "inst_late_payment_count",
    "inst_underpaid_count",
    "inst_count",
    "cc_card_count",
    "cc_late_payment_count",
    "cc_serious_dpd_count",
    "pos_loan_count",
    "pos_late_payment_count",
    "pos_serious_dpd_count",
    "pos_active_contracts",
    "pos_completed_contracts",
    "pos_signed_contracts",
    "pos_demand_contracts",
]

for col in count_cols:
    if (train_df[col] < 0).any():
        print(col, (train_df[col] < 0).sum())

"""
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
███    ⚠️ AS I FOUND THAT THERE ARE LOT OF FEATURES CONTAINING OUTLIERS BUT REMOVING THEM AFFECTS THE MODEL PERFORMANCE    ███
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
"""
"""
████████╗██████╗  █████╗ ██╗███╗   ██╗    ████████╗███████╗███████╗████████╗    ███████╗██████╗ ██╗     ██╗████████╗
╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ██╔════╝██╔══██╗██║     ██║╚══██╔══╝
   ██║   ██████╔╝███████║██║██╔██╗ ██║       ██║   █████╗  ███████╗   ██║       ███████╗██████╔╝██║     ██║   ██║
   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║       ██║   ██╔══╝  ╚════██║   ██║       ╚════██║██╔═══╝ ██║     ██║   ██║
   ██║   ██║  ██║██║  ██║██║██║ ╚████║       ██║   ███████╗███████║   ██║       ███████║██║     ███████╗██║   ██║
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝       ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚══════╝╚═╝     ╚══════╝╚═╝   ╚═╝
"""

# Input features for training
training_features = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
# Input features for testing
testing_features = test_df.drop(columns="SK_ID_CURR")
# Output feature for training
target_set = train_df["TARGET"]

# Splitting is done on training dataset to get model performance
X_train, X_eval, y_train, y_eval = train_test_split(
    training_features,
    target_set,
    test_size=0.2,
    shuffle=True,
    stratify=target_set,
    random_state=42,
)
# storinf datasets using their name in a dict
datasets = {
    "train_df": train_df,
    "test_df": test_df,
    "training_features": training_features,
    "testing_features": testing_features,
    "target_set": target_set,
    "X_train": X_train,
    "X_eval": X_eval,
    "y_train": y_train,
    "y_eval": y_eval,
}
# This will return a the summar of the collection of the datasets
summary = []

for name, dataset in datasets.items():

    rows = dataset.shape[0]
    cols = dataset.shape[1] if isinstance(dataset, pd.DataFrame) else 1

    summary.append(
        {
            # name of the datasets
            "DATASET": name,
            # No. of rows
            "ROWS": rows,
            # No. of columns
            "COLUMNS": cols,
            # missing values
            "MISSING": dataset.isnull().sum().sum(),
            # numeric columns
            "NUMERIC": (
                len(dataset.select_dtypes(include="number").columns)
                if isinstance(dataset, pd.DataFrame)
                else "-"
            ),
            # categorial of string type or string columns
            "STRING": (
                len(dataset.select_dtypes(include=["object", "string"]).columns)
                if isinstance(dataset, pd.DataFrame)
                else "-"
            ),
            # categorial of boolean type or boolean columns
            "BOOLEAN": (
                len(dataset.select_dtypes(include="bool").columns)
                if isinstance(dataset, pd.DataFrame)
                else "-"
            ),
        }
    )

summary_df = pd.DataFrame(summary)

console.print(
f"""\n\n
[bold green]______________________________________ [bold #C7009D]SUMMARY OF DATAS[/][bold green] ______________________________________[/]\n
  {summary_df}\n
[bold green]___________________________________________________________________________________________________________________________[/]
"""
)

"""
██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗    ██████╗ ██████╗  ██████╗  ██████╗███████╗███████╗███████╗██╗███╗   ██╗ ██████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝    ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔════╝
██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║       ██████╔╝██████╔╝██║   ██║██║     █████╗  ███████╗███████╗██║██╔██╗ ██║██║  ███╗
██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║       ██╔═══╝ ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║╚════██║██║██║╚██╗██║██║   ██║
██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║       ██║     ██║  ██║╚██████╔╝╚██████╗███████╗███████║███████║██║██║ ╚████║╚██████╔╝
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝       ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝
"""

num_cols_X_train = X_train.select_dtypes(include=np.number).columns
category_cols_X_train = X_train.select_dtypes(include=["object", "category"]).columns

num_col_pipeline_en = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(quantile_range=(0.25, 0.75))),
    ]
)

category_col_pipeline_en = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
        ),
    ]
)

pp_elasticnet = ColumnTransformer(
    transformers=[
        ("num_scale", num_col_pipeline_en, num_cols_X_train),
        ("category_scale", category_col_pipeline_en, category_cols_X_train),
    ],
    verbose_feature_names_out=False
)

pipeline_elastic_net = Pipeline(
    [
        ("pp_elasticnet", pp_elasticnet),
        (
            "logreg_en__base",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",  # Required for elasticnet
                l1_ratio=0.5,  # 0 = L2, 1 = L1, 0.5 = mix
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                tol=1e-4,
            ),
        ),
    ]
)

num_col_pipeline_lgbm_xg = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

category_col_pipeline_lgbm_xg = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

pp_lgbm_xg = ColumnTransformer(
    transformers=[
        ("num_scale", num_col_pipeline_lgbm_xg, num_cols_X_train),
        ("category_scale", category_col_pipeline_lgbm_xg, category_cols_X_train),
    ],
    verbose_feature_names_out=False
)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()

pipeline_xgb = Pipeline(
    [
        ("pp_lgbm_xg", pp_lgbm_xg),
        (
            "xgbc",
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                scale_pos_weight = neg_count / max(pos_count, 1),
                n_estimators=1000,
                tree_method="hist",
                n_jobs=-1,
            ),
        ),
    ]
)

pipeline_lgbmc = Pipeline(
    [
        ("pp_lgbm_xg", pp_lgbm_xg),
        (
            "lgbmc_base",
            LGBMClassifier(
                n_estimators=1000,
                random_state=42, 
                scale_pos_weight = neg_count / max(pos_count, 1), 
                verbose=-1
            ),
        ),
    ]
)

num_col_pipeline_catbc = Pipeline([("imputer", SimpleImputer(strategy="median"))])

categorty_col_pipeline_catbc = Pipeline(
    [("imputer", SimpleImputer(strategy="most_frequent"))]
)

pp_catbc = ColumnTransformer(
    transformers=[
        ("num_scale", num_col_pipeline_catbc, num_cols_X_train),
        ("category_scale", categorty_col_pipeline_catbc, category_cols_X_train),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

pp_catbc.set_output(transform="pandas")

pipeline_catbc = Pipeline(
    [
        ("pp_catbc", pp_catbc),
        (
            "catbc_base",
            CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                loss_function="logloss",
                eval_metric="AUC",
                auto_class_weights="balanced",
                random_seed=42,
                verbose=100,
                cat_features=category_cols_X_train.tolist(),
                allow_writing_files=False,
            ),
        ),
    ]
)
