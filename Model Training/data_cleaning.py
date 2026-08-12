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

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Train test split
from sklearn.model_selection import train_test_split

# Pipeline
from sklearn.pipeline import Pipeline

# scaling, encoding & imputing features
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from xgboost.callback import TrainingCallback
from lightgbm import early_stopping

# hyperparametr tuner
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
# validation
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Performance metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    ConfusionMatrixDisplay
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

progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console,
)

# Helper: print a static "complete" line for a finished sub-model, then remove
# its task from the live-rendered area. progress/parent_task's Live session is
# never stopped or restarted by this, so the parent bar stays continuously
# live throughout instead of freezing/reprinting itself at a stop/start boundary.
def finish_sub_task(progress_obj, task_id, label):
    progress_obj.update(task_id, completed=100)
    console.print(f"[#BDFF08]{label} • Complete[/]  " + "━" * 40 + "  100%")
    progress_obj.remove_task(task_id)


with progress:

    parent_task = progress.add_task("[#C7009D]MODEL TRAINING PIPELINE[/]", total=100)

    """
    ██████╗  █████╗ ████████╗ █████╗     ██╗      ██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗
    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║     ██╔═══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝
    ██║  ██║███████║   ██║   ███████║    ██║     ██║   ██║███████║██║  ██║██║██╔██╗ ██║██║  ███╗
    ██║  ██║██╔══██║   ██║   ██╔══██║    ██║     ██║   ██║██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
    ██████╔╝██║  ██║   ██║   ██║  ██║    ███████╗╚██████╔╝██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝
    """

    progress.update(parent_task, description="[#00FFFF]DATA LOADING[/]")

    # Training dataset
    train_df = pd.read_csv(r"Processed Datasets/final_train.csv")
    # Testing dataset
    test_df = pd.read_csv(r"Processed Datasets/final_test.csv")

    # Replace known sentinel values
    train_df.loc[train_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    test_df.loc[test_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # Replace infinities
    train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    progress.update(parent_task, completed=10)

    """
     ██████╗ ██╗   ██╗████████╗██╗     ██╗███████╗██████╗      █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
    ██╔═══██╗██║   ██║╚══██╔══╝██║     ██║██╔════╝██╔══██╗    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
    ██║   ██║██║   ██║   ██║   ██║     ██║█████╗  ██████╔╝    ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
    ██║   ██║██║   ██║   ██║   ██║     ██║██╔══╝  ██╔══██╗    ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
    ╚██████╔╝╚██████╔╝   ██║   ███████╗██║███████╗██║  ██║    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
     ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
    """

    progress.update(parent_task, description="[#00FFFF]DATA CLEANING[/]")

    numeric_cols = train_df.select_dtypes(include="number")

    summary = []

    for col in numeric_cols.columns:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (train_df[col] < lower) | (train_df[col] > upper)
        outlier_count = outlier_mask.sum()
        outlier_pct = round(outlier_count / len(train_df) * 100, 2)
        skew = round(train_df[col].skew(), 2)
        missing_pct = round(train_df[col].isna().mean() * 100, 2)
        minimum = train_df[col].min()
        maximum = train_df[col].max()
        unique = train_df[col].nunique()

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

    outlier_summary = (
        pd.DataFrame(summary)
        .sort_values("Outlier %", ascending=False)
        .reset_index(drop=True)
    )

    console.print(
        "[bold green]___________________________________________________[/] "
        "[bold #C7009D]SUMMARY OF OUTLIERS[/] "
        "[bold green]___________________________________________________[/]"
    )
    console.print(outlier_summary)
    console.print(
        "[bold green]___________________________________________________________________________________________________________________________\n\n[/]"
    )

    progress.update(parent_task, completed=25)

    """
    ██████╗  █████╗ ██████╗     ██████╗ ██╗   ██╗████████╗██╗     ██╗███████╗██████╗     ██████╗ ███████╗███╗   ███╗ ██████╗ ██╗   ██╗ █████╗ ██╗     
    ██╔══██╗██╔══██╗██╔══██╗   ██╔═══██╗██║   ██║╚══██╔══╝██║     ██║██╔════╝██╔══██╗    ██╔══██╗██╔════╝████╗ ████║██╔═══██╗██║   ██║██╔══██╗██║     
    ██████╔╝███████║██║  ██║   ██║   ██║██║   ██║   ██║   ██║     ██║█████╗  ██████╔╝    ██████╔╝█████╗  ██╔████╔██║██║   ██║██║   ██║███████║██║     
    ██╔══██╗██╔══██║██║  ██║   ██║   ██║██║   ██║   ██║   ██║     ██║██╔══╝  ██╔══██╗    ██╔══██╗██╔══╝  ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██╔══██║██║     
    ██████╔╝██║  ██║██████╔╝   ╚██████╔╝╚██████╔╝   ██║   ███████╗██║███████╗██║  ██║    ██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ██║  ██║███████╗
    ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚══════╝
    """



    progress.update(parent_task, completed=25)

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

    progress.update(parent_task, description="[#00FFFF]DATA SPLITTING[/]")

    training_features = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    testing_features = test_df.drop(columns="SK_ID_CURR")
    target_set = train_df["TARGET"]

    progress.update(parent_task, completed=28)

    progress.update(parent_task, description="[blue]DATA SPLITTING[/]")

    X_train, X_eval, y_train, y_eval = train_test_split(
        training_features,
        target_set,
        test_size=0.2,
        shuffle=True,
        stratify=target_set,
        random_state=42,
    )

    progress.update(parent_task, completed=35)

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

    summary = []

    for name, dataset in datasets.items():
        rows = dataset.shape[0]
        cols = dataset.shape[1] if isinstance(dataset, pd.DataFrame) else 1

        summary.append(
            {
                "DATASET": name,
                "ROWS": rows,
                "COLUMNS": cols,
                "MISSING": dataset.isnull().sum().sum(),
                "NUMERIC": (
                    len(dataset.select_dtypes(include="number").columns)
                    if isinstance(dataset, pd.DataFrame)
                    else "-"
                ),
                "STRING": (
                    len(dataset.select_dtypes(include=["object", "string"]).columns)
                    if isinstance(dataset, pd.DataFrame)
                    else "-"
                ),
                "BOOLEAN": (
                    len(dataset.select_dtypes(include="bool").columns)
                    if isinstance(dataset, pd.DataFrame)
                    else "-"
                ),
            }
        )

    summary_df = pd.DataFrame(summary)

    progress.update(parent_task, completed=39)

    console.print(
        "[bold green]_____________________________________________________[/] "
        "[bold #C7009D]SUMMARY OF DATAS[/] "
        "[bold green]_____________________________________________________[/]"
    )
    console.print(summary_df)
    console.print(
        "[bold green]____________________________________________________________________________________________________________________________\n\n[/]"
    )

    progress.update(parent_task, completed=40)

    """
    ██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗    ██████╗ ██████╗  ██████╗  ██████╗███████╗███████╗███████╗██╗███╗   ██╗ ██████╗
    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝    ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔════╝
    ██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║       ██████╔╝██████╔╝██║   ██║██║     █████╗  ███████╗███████╗██║██╔██╗ ██║██║  ███╗
    ██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║       ██╔═══╝ ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║╚════██║██║██║╚██╗██║██║   ██║
    ██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║       ██║     ██║  ██║╚██████╔╝╚██████╗███████╗███████║███████║██║██║ ╚████║╚██████╔╝
    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝       ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝
    """

    progress.update(parent_task, description="[magenta]PREPROCESSING PIPELINES[/]")

    num_cols_X_train = X_train.select_dtypes(include=np.number).columns
    category_cols_X_train = X_train.select_dtypes(
        include=["object", "category", "str"]
    ).columns

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
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
            ),
        ]
    )
    pp_elasticnet = ColumnTransformer(
        transformers=[
            ("num_scale", num_col_pipeline_en, num_cols_X_train),
            ("category_scale", category_col_pipeline_en, category_cols_X_train),
        ],
        verbose_feature_names_out=False,
    )

    num_col_pipeline_xg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    category_col_pipeline_xg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    pp_xg = ColumnTransformer(
        transformers=[
            ("num_scale", num_col_pipeline_xg, num_cols_X_train),
            ("category_scale", category_col_pipeline_xg, category_cols_X_train),
        ],
        verbose_feature_names_out=False,
    )
    pp_lgbm = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols_X_train),
            ("cat", SimpleImputer(strategy="most_frequent"), category_cols_X_train),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pp_lgbm.set_output(transform="pandas")
    X_train_lgbm = pp_lgbm.fit_transform(X_train)
    X_eval_lgbm = pp_lgbm.transform(X_eval)
    for col in category_cols_X_train:
        X_train_lgbm[col] = X_train_lgbm[col].astype("category")
        X_eval_lgbm[col] = pd.Categorical(
            X_eval_lgbm[col],
            categories=X_train_lgbm[col].cat.categories
        )

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

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

    X_train_catbc = pp_catbc.fit_transform(X_train)
    X_eval_catbc = pp_catbc.transform(X_eval)

    for col in category_cols_X_train:
        X_train_catbc[col] = X_train_catbc[col].astype(str)
        X_eval_catbc[col] = X_eval_catbc[col].astype(str)

    console.print("[bold green]✅ PREPROCESSING COMPLETED[/]\n\n")
    progress.update(parent_task, completed=42)\

    """
    ██████╗  █████╗ ███████╗███████╗██╗     ██╗███╗   ██╗███████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     ███████╗
    ██╔══██╗██╔══██╗██╔════╝██╔════╝██║     ██║████╗  ██║██╔════╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔════╝
    ██████╔╝███████║███████╗█████╗  ██║     ██║██╔██╗ ██║█████╗      ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ███████╗
    ██╔══██╗██╔══██║╚════██║██╔══╝  ██║     ██║██║╚██╗██║██╔══╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ╚════██║
    ██████╔╝██║  ██║███████║███████╗███████╗██║██║ ╚████║███████╗    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗███████║
    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝
    """

    progress.update(parent_task, description="[magenta]PREPARING BASELINE MODELS[/]")

    # ──────────────────────────────────────────────────────────────────────────────
    # ElasticNet Baseline Parameters
    # ──────────────────────────────────────────────────────────────────────────────

    pipeline_elastic_net = Pipeline(
        [
            ("pp_elasticnet", pp_elasticnet),
            (
                "en_base",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
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

    # ──────────────────────────────────────────────────────────────────────────────
    # XGBoost Baseline Parameters
    # ──────────────────────────────────────────────────────────────────────────────

    class XGBRichProgress(TrainingCallback):
        def __init__(self, progress, task, total):
            self.progress = progress
            self.task = task
            self.total = total  # max possible rounds — used only for the % calc

        def after_iteration(self, model, epoch, evals_log):
            pct = min((epoch + 1) / self.total * 100, 100)
            self.progress.update(self.task, completed=pct)  # total is never touched
            return False

    X_train_xgb = pp_xg.fit_transform(X_train)
    X_eval_xgb = pp_xg.transform(X_eval)

    xgbc_base = XGBClassifier(
        random_state=42,
        eval_metric="auc",
        scale_pos_weight=neg_count / max(pos_count, 1),
        n_estimators=1000,
        tree_method="hist",
        early_stopping_rounds=250,
        n_jobs=-1,
    )
    # callback is attached right before .fit(), once the real task exists — see below

    # ──────────────────────────────────────────────────────────────────────────────
    # LightGBM Baseline parameters
    # ──────────────────────────────────────────────────────────────────────────────

    lgbmc_base = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        metric="auc",
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_count / pos_count,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    # ──────────────────────────────────────────────────────────────────────────────
    # CatBoost Baseline parameters
    # ──────────────────────────────────────────────────────────────────────────────

    class CatBoostRichProgress:
        def __init__(self, progress, task, total):
            self.progress = progress
            self.task = task
            self.total = total

        def after_iteration(self, info):
            pct = min((info.iteration + 1) / self.total * 100, 100)
            self.progress.update(self.task, completed=pct)
            return False

    catbc_base = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=250,
        allow_writing_files=False,
    )

    console.print(
        "[bold green]✅ BASELINE MODELS PREPARED AND READY TO TRAIN[/]\n\n"
    )
    progress.update(parent_task, completed=45)

    """
    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║            ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║            ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗       ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝
    """

    # NOTE: no progress.stop()/start() and no nested `with Progress(...)` here.
    # Every sub-task below is added to and removed from the SAME `progress`
    # instance that owns parent_task — so parent_task's Live session is never
    # interrupted, and never gets frozen/reprinted as a static frame mid-run.

    # ──────────────────────────────────────────────────────────────────────────────
    # Elastic Net
    # ──────────────────────────────────────────────────────────────────────────────
    elastic_task = progress.add_task("[#BDFF08]Elastic Net • Fitting...[/]", total=100)

    pipeline_elastic_net.fit(X_train, y_train)

    finish_sub_task(progress, elastic_task, "Elastic Net")
    progress.update(parent_task, completed=57)

    # ──────────────────────────────────────────────────────────────────────────────
    # XGBoost
    # ──────────────────────────────────────────────────────────────────────────────
    xgb_task = progress.add_task("[#BDFF08]XGBoost • Fitting...[/]", total=100)

    xgbc_base.callbacks = [XGBRichProgress(progress, xgb_task, 1000)]
    xgbc_base.fit(
        X_train_xgb, y_train,
        eval_set=[(X_eval_xgb, y_eval)],
        verbose=False,
    )

    finish_sub_task(progress, xgb_task, "XGBoost")
    progress.update(parent_task, completed=64)

    # ──────────────────────────────────────────────────────────────────────────────
    # LightGBM
    # ──────────────────────────────────────────────────────────────────────────────
    lgbm_task = progress.add_task("[#BDFF08]LightGBM • Fitting...[/]", total=100)

    def lgbm_rich_progress(env):
        if env.evaluation_result_list:
            pct = min((env.iteration + 1) / env.end_iteration * 100, 100)
            progress.update(lgbm_task, completed=pct)

    lgbmc_base.fit(
        X_train_lgbm,
        y_train,
        categorical_feature=category_cols_X_train.tolist(),
        eval_set=[(X_eval_lgbm, y_eval)],
        callbacks=[lgbm_rich_progress, early_stopping(250, verbose=False)],
    )

    finish_sub_task(progress, lgbm_task, "LightGBM")
    progress.update(parent_task, completed=71)

    # ──────────────────────────────────────────────────────────────────────────────
    # CatBoost
    # ──────────────────────────────────────────────────────────────────────────────
    catboost_task = progress.add_task("[#BDFF08]CatBoost • Fitting...[/]", total=100)

    catbc_base.fit(
        X_train_catbc,
        y_train,
        cat_features=category_cols_X_train.tolist(),
        eval_set=(X_eval_catbc, y_eval),
        callbacks=[CatBoostRichProgress(progress, catboost_task, 1000)],
    )

    finish_sub_task(progress, catboost_task, "CatBoost")
    progress.update(parent_task, completed=74)

    """
    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║         █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║         ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗    ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    """

    progress.update(
        parent_task, description="[yellow]MODEL EVALUATION[/]", completed=75
    )

    y_pred_en = pipeline_elastic_net.predict(X_eval)
    y_prob_en = pipeline_elastic_net.predict_proba(X_eval)[:, 1]

    y_pred_xgbc = xgbc_base.predict(X_eval_xgb)
    y_prob_xgbc = xgbc_base.predict_proba(X_eval_xgb)[:, 1]

    y_pred_lgbmc = lgbmc_base.predict(X_eval_lgbm)
    y_prob_lgbmc = lgbmc_base.predict_proba(X_eval_lgbm)[:, 1]

    y_pred_catbc = catbc_base.predict(X_eval_catbc)
    y_prob_catbc = catbc_base.predict_proba(X_eval_catbc)[:, 1]

    progress.update(
        parent_task, description="[yellow]METRIC CALCULATION[/]", completed=78
    )

    base_result = pd.DataFrame(
        {
            "Models": [
                "Elastic Net log reg.",
                "XG Boost Classifier",
                "LightGBM Classifier",
                "CatBoost Classifier",
            ],
            "ACCURACY": [
                accuracy_score(y_eval, y_pred_en),
                accuracy_score(y_eval, y_pred_xgbc),
                accuracy_score(y_eval, y_pred_lgbmc),
                accuracy_score(y_eval, y_pred_catbc),
            ],
            "PRECISION": [
                precision_score(y_eval, y_pred_en),
                precision_score(y_eval, y_pred_xgbc),
                precision_score(y_eval, y_pred_lgbmc),
                precision_score(y_eval, y_pred_catbc),
            ],
            "RECALL": [
                recall_score(y_eval, y_pred_en),
                recall_score(y_eval, y_pred_xgbc),
                recall_score(y_eval, y_pred_lgbmc),
                recall_score(y_eval, y_pred_catbc),
            ],
            "F1": [
                f1_score(y_eval, y_pred_en),
                f1_score(y_eval, y_pred_xgbc),
                f1_score(y_eval, y_pred_lgbmc),
                f1_score(y_eval, y_pred_catbc),
            ],
            "ROC-AUC": [
                roc_auc_score(y_eval, y_prob_en),
                roc_auc_score(y_eval, y_prob_xgbc),
                roc_auc_score(y_eval, y_prob_lgbmc),
                roc_auc_score(y_eval, y_prob_catbc),
            ],
        }
    )

    console.print(
        "\n______________________________________ BASE MODELS PERFORMANCE (IMBALANCED CLASS) ______________________________________\n"
    )
    console.print(base_result.round(5))
    console.print(
        "________________________________________________________________________________________________________________________\n"
    )

    progress.update(
        parent_task, description="[green]BASELINE EVALUATION COMPLETE[/]", completed=80
    )

    # ─── Optuna tuning picks up parent_task from completed=90 onward ───