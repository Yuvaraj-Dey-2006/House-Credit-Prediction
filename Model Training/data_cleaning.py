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
    ConfusionMatrixDisplay,
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
    ██████╗  █████╗ ██████╗     ██████╗ ██╗   ██╗████████╗██╗     ██╗███████╗██████╗          
    ██╔══██╗██╔══██╗██╔══██╗   ██╔═══██╗██║   ██║╚══██╔══╝██║     ██║██╔════╝██╔══██╗         
    ██████╔╝███████║██║  ██║   ██║   ██║██║   ██║   ██║   ██║     ██║█████╗  ██████╔╝        
    ██╔══██╗██╔══██║██║  ██║   ██║   ██║██║   ██║   ██║   ██║     ██║██╔══╝  ██╔══██╗        
    ██████╔╝██║  ██║██████╔╝   ╚██████╔╝╚██████╔╝   ██║   ███████╗██║███████╗██║  ██║    
    ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝  
    
    ██████╗ ███████╗███╗   ███╗ ██████╗ ██╗   ██╗ █████╗ ██╗
    ██╔══██╗██╔════╝████╗ ████║██╔═══██╗██║   ██║██╔══██╗██║
    ██████╔╝█████╗  ██╔████╔██║██║   ██║██║   ██║███████║██║ 
    ██╔══██╗██╔══╝  ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██╔══██║██║ 
    ██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ██║  ██║███████╗
    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚══════╝
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
    ██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗    
    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝    
    ██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║       
    ██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║      
    ██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║       
    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝       
    
    ██████╗ ██████╗  ██████╗  ██████╗███████╗███████╗███████╗██╗███╗   ██╗ ██████╗
    ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔════╝
    ██████╔╝██████╔╝██║   ██║██║     █████╗  ███████╗███████╗██║██╔██╗ ██║██║  ███╗
    ██╔═══╝ ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║╚════██║██║██║╚██╗██║██║   ██║
    ██║     ██║  ██║╚██████╔╝╚██████╗███████╗███████║███████║██║██║ ╚████║╚██████╔╝
    ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝
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
            X_eval_lgbm[col], categories=X_train_lgbm[col].cat.categories
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
    progress.update(parent_task, completed=42)
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
            return True   

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

    console.print("[bold green]✅ BASELINE MODELS PREPARED AND READY TO TRAIN[/]\n\n")
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
        X_train_xgb,
        y_train,
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
    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         
    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         
    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║         
    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║         
    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗   
    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝    
    
    ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
    ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
    █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
    ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
    ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
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
                "Elastic Net log reg",
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

    best_model_name = base_result.loc[base_result["ROC-AUC"].idxmax(), "Models"]

    console.print(
        f"\n[bold #C7009D]➤  BEST BASELINE MODEL (by ROC-AUC): {best_model_name}[/]\n"
    )

    model_lookup = {
        "Elastic Net log reg": pipeline_elastic_net,
        "XG Boost Classifier": xgbc_base,
        "LightGBM Classifier": lgbmc_base,
        "CatBoost Classifier": catbc_base,
    }
    best_model = model_lookup[best_model_name]

    progress.update(
        parent_task, description="[green]BASELINE EVALUATION COMPLETE[/]", completed=80
    )

"""
██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗████████╗███████╗██████╗ 
██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔════╝██╔══██╗
███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗     ██║   █████╗  ██████╔╝
██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══╝  ██╔══██╗
██║  ██║   ██║   ██║     ███████╗██║  ██║██║     ██║  ██║██████╔╝██║  ██║██║ ╚═╝ ██║███████╗   ██║   ███████╗██║  ██║
╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

████████╗██╗   ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ 
╚══██╔══╝██║   ██║████╗  ██║██║████╗  ██║██╔════╝ 
   ██║   ██║   ██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
   ██║   ██║   ██║██║╚██╗██║██║██║╚██╗██║██║   ██║
   ██║   ╚██████╔╝██║ ╚████║██║██║ ╚████║╚██████╔╝
   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 



def objective_elasticnet(trial):

    C = trial.suggest_float("C", 1e-4, 10.0, log=True)

    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    pipeline_en = Pipeline(
        [
            ("pp_elasticnet", pp_elasticnet),
            (
                "en_tuned",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=C,
                    l1_ratio=l1_ratio,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    tol=1e-4,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline_en, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    # store the fold spread on the trial itself so we can inspect stability later
    trial.set_user_attr("cv_std", scores.std())
    trial.set_user_attr("cv_scores", scores.tolist())

    return scores.mean()


study_elasticnet = optuna.create_study(direction="maximize")
study_elasticnet.optimize(objective_elasticnet, n_trials=50, show_progress_bar=False)

best_trial = study_elasticnet.best_trial

console.print(
    f"\n[bold #C7009D]➤ Elastic Net — best CV ROC-AUC: "
    f"{best_trial.value:.5f} ± {best_trial.user_attrs['cv_std']:.5f}[/]"
)
console.print(f"  Best params: {best_trial.params}\n")

# ── Range of good values, not just the single best point ──
# Looking only at the #1 trial can be misleading if the surface is flat/noisy.
# This pulls the top 10% of trials and shows the range C/l1_ratio took in that
# band, so you can see whether the optimum is a narrow spike or a broad region.
trials_df = study_elasticnet.trials_dataframe()
trials_df = trials_df[trials_df["state"] == "COMPLETE"].sort_values(
    "value", ascending=False
)

top_n = max(1, int(len(trials_df) * 0.1))  # top 10% of trials
top_trials = trials_df.head(top_n)

console.print(
    f"[bold #C7009D]➤ Elastic Net — top {top_n} trials (of {len(trials_df)}) parameter ranges:[/]"
)
console.print(
    f"  C:        {top_trials['params_C'].min():.5g} – {top_trials['params_C'].max():.5g}"
)
console.print(
    f"  l1_ratio: {top_trials['params_l1_ratio'].min():.3f} – {top_trials['params_l1_ratio'].max():.3f}"
)
console.print(
    f"  ROC-AUC:  {top_trials['value'].min():.5f} – {top_trials['value'].max():.5f}\n"
)

# ──────────────────────────────────────────────────────────────────────────────
# XGBoost — Optuna objective
#
# Unlike Elastic Net, XGBoost's baseline used early_stopping_rounds, which
# sklearn's cross_val_score can't handle cleanly (it needs an eval_set per
# fold). So this loops the folds manually, refitting pp_xg per fold to avoid
# leakage, and early-stops against that fold's own held-out slice.
#
# 3 folds instead of 5 — with 50 trials × early-stopped boosting, 5-fold CV
# here would be considerably slower for a similar signal; 3 folds is a
# reasonable speed/reliability trade-off during search. You can raise this
# once you've narrowed the search space if you want tighter estimates.
# ──────────────────────────────────────────────────────────────────────────────


def objective_xgboost(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        # refit preprocessing per fold — avoids leaking val-fold stats into
        # imputation, unlike the single fit_transform used for the baseline
        X_fold_train_xgb = pp_xg.fit_transform(X_fold_train)
        X_fold_val_xgb = pp_xg.transform(X_fold_val)
        fold_neg = (y_fold_train == 0).sum()
        fold_pos = (y_fold_train == 1).sum()
        model = XGBClassifier(
            **params,
            n_estimators=1000,
            random_state=42,
            eval_metric="auc",
            scale_pos_weight=fold_neg / max(fold_pos, 1),
            tree_method="hist",
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        model.fit(
            X_fold_train_xgb,
            y_fold_train,
            eval_set=[(X_fold_val_xgb, y_fold_val)],
            verbose=False,
        )
        fold_prob = model.predict_proba(X_fold_val_xgb)[:, 1]
        fold_scores.append(roc_auc_score(y_fold_val, fold_prob))
    fold_scores = np.array(fold_scores)
    trial.set_user_attr("cv_std", fold_scores.std())
    trial.set_user_attr("cv_scores", fold_scores.tolist())
    return fold_scores.mean()


study_xgboost = optuna.create_study(direction="maximize")
study_xgboost.optimize(objective_xgboost, n_trials=50, show_progress_bar=False)
best_trial = study_xgboost.best_trial
console.print(
    f"\n[bold #C7009D]➤ XGBoost — best CV ROC-AUC: "
    f"{best_trial.value:.5f} ± {best_trial.user_attrs['cv_std']:.5f}[/]"
)
console.print(f"  Best params: {best_trial.params}\n")
# ── Range of good values across the top 10% of trials ──
trials_df = study_xgboost.trials_dataframe()
trials_df = trials_df[trials_df["state"] == "COMPLETE"].sort_values(
    "value", ascending=False
)
top_n = max(1, int(len(trials_df) * 0.1))
top_trials = trials_df.head(top_n)
console.print(
    f"[bold #C7009D]➤ XGBoost — top {top_n} trials (of {len(trials_df)}) parameter ranges:[/]"
)
for param in [
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_alpha",
    "reg_lambda",
]:
    col = f"params_{param}"
    console.print(
        f"  {param:<17} {top_trials[col].min():.5g} – {top_trials[col].max():.5g}"
    )
console.print(
    f"  {'ROC-AUC':<17} {top_trials['value'].min():.5f} – {top_trials['value'].max():.5f}\n"
)


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM — Optuna objective
#
# Same manual-fold approach as XGBoost (early stopping needs its own eval_set
# per fold). Also needs pp_lgbm refit per fold + category dtype re-applied,
# matching how X_train_lgbm/X_eval_lgbm were built for the baseline.
# ──────────────────────────────────────────────────────────────────────────────


def objective_lightgbm(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        # refit pp_lgbm per fold — same leakage concern as XGBoost's objective
        X_fold_train_lgbm = pp_lgbm.fit_transform(X_fold_train)
        X_fold_val_lgbm = pp_lgbm.transform(X_fold_val)
        # re-apply category dtype using this fold's own categories,
        # matching how X_train_lgbm/X_eval_lgbm were set up originally
        for col in category_cols_X_train:
            X_fold_train_lgbm[col] = X_fold_train_lgbm[col].astype("category")
            X_fold_val_lgbm[col] = pd.Categorical(
                X_fold_val_lgbm[col],
                categories=X_fold_train_lgbm[col].cat.categories,
            )
        fold_neg = (y_fold_train == 0).sum()
        fold_pos = (y_fold_train == 1).sum()
        model = LGBMClassifier(
            **params,
            n_estimators=1000,
            metric="auc",
            scale_pos_weight=fold_neg / max(fold_pos, 1),
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(
            X_fold_train_lgbm,
            y_fold_train,
            categorical_feature=category_cols_X_train.tolist(),
            eval_set=[(X_fold_val_lgbm, y_fold_val)],
            callbacks=[early_stopping(50, verbose=False)],
        )
        fold_prob = model.predict_proba(X_fold_val_lgbm)[:, 1]
        fold_scores.append(roc_auc_score(y_fold_val, fold_prob))
    fold_scores = np.array(fold_scores)
    trial.set_user_attr("cv_std", fold_scores.std())
    trial.set_user_attr("cv_scores", fold_scores.tolist())
    return fold_scores.mean()


study_lightgbm = optuna.create_study(direction="maximize")
study_lightgbm.optimize(objective_lightgbm, n_trials=50, show_progress_bar=False)
best_trial = study_lightgbm.best_trial
console.print(
    f"\n[bold #C7009D]➤ LightGBM — best CV ROC-AUC: "
    f"{best_trial.value:.5f} ± {best_trial.user_attrs['cv_std']:.5f}[/]"
)
console.print(f"  Best params: {best_trial.params}\n")
# ── Range of good values across the top 10% of trials ──
trials_df = study_lightgbm.trials_dataframe()
trials_df = trials_df[trials_df["state"] == "COMPLETE"].sort_values(
    "value", ascending=False
)
top_n = max(1, int(len(trials_df) * 0.1))
top_trials = trials_df.head(top_n)
console.print(
    f"[bold #C7009D]➤ LightGBM — top {top_n} trials (of {len(trials_df)}) parameter ranges:[/]"
)
for param in [
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
]:
    col = f"params_{param}"
    console.print(
        f"  {param:<17} {top_trials[col].min():.5g} – {top_trials[col].max():.5g}"
    )
console.print(
    f"  {'ROC-AUC':<17} {top_trials['value'].min():.5f} – {top_trials['value'].max():.5f}\n"
)


# ──────────────────────────────────────────────────────────────────────────────
# CatBoost — Optuna objective
#
# Same manual-fold pattern. CatBoost's early stopping is a constructor arg
# (early_stopping_rounds) rather than a fit-time callback, and categorical
# columns just need to be cast to str per fold — no dtype/category-list
# bookkeeping like LightGBM needed.
# ──────────────────────────────────────────────────────────────────────────────


def objective_catboost(trial):
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        # refit pp_catbc per fold — same leakage concern as the other objectives
        X_fold_train_catbc = pp_catbc.fit_transform(X_fold_train)
        X_fold_val_catbc = pp_catbc.transform(X_fold_val)
        for col in category_cols_X_train:
            X_fold_train_catbc[col] = X_fold_train_catbc[col].astype(str)
            X_fold_val_catbc[col] = X_fold_val_catbc[col].astype(str)
        model = CatBoostClassifier(
            **params,
            iterations=1000,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
            early_stopping_rounds=50,
            allow_writing_files=False,
        )
        model.fit(
            X_fold_train_catbc,
            y_fold_train,
            cat_features=category_cols_X_train.tolist(),
            eval_set=(X_fold_val_catbc, y_fold_val),
        )
        fold_prob = model.predict_proba(X_fold_val_catbc)[:, 1]
        fold_scores.append(roc_auc_score(y_fold_val, fold_prob))
    fold_scores = np.array(fold_scores)
    trial.set_user_attr("cv_std", fold_scores.std())
    trial.set_user_attr("cv_scores", fold_scores.tolist())
    return fold_scores.mean()


study_catboost = optuna.create_study(direction="maximize")
study_catboost.optimize(objective_catboost, n_trials=50, show_progress_bar=False)
best_trial = study_catboost.best_trial
console.print(
    f"\n[bold #C7009D]➤ CatBoost — best CV ROC-AUC: "
    f"{best_trial.value:.5f} ± {best_trial.user_attrs['cv_std']:.5f}[/]"
)
console.print(f"  Best params: {best_trial.params}\n")
# ── Range of good values across the top 10% of trials ──
trials_df = study_catboost.trials_dataframe()
trials_df = trials_df[trials_df["state"] == "COMPLETE"].sort_values(
    "value", ascending=False
)
top_n = max(1, int(len(trials_df) * 0.1))
top_trials = trials_df.head(top_n)
console.print(
    f"[bold #C7009D]➤ CatBoost — top {top_n} trials (of {len(trials_df)}) parameter ranges:[/]"
)
for param in [
    "depth",
    "learning_rate",
    "l2_leaf_reg",
    "bagging_temperature",
    "random_strength",
    "border_count",
]:
    col = f"params_{param}"
    console.print(
        f"  {param:<19} {top_trials[col].min():.5g} – {top_trials[col].max():.5g}"
    )
console.print(
    f"  {'ROC-AUC':<19} {top_trials['value'].min():.5f} – {top_trials['value'].max():.5f}\n"
)
"""