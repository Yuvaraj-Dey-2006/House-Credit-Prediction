'''
██╗███╗   ███╗██████╗  ██████╗ ██████╗ ████████╗██╗███╗   ██╗ ██████╗    ██╗     ██╗██████╗ ██████╗  █████╗ ██████╗ ██╗███████╗███████╗ 
██║████╗ ████║██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██║████╗  ██║██╔════╝    ██║     ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝
██║██╔████╔██║██████╔╝██║   ██║██████╔╝   ██║   ██║██╔██╗ ██║██║  ███╗   ██║     ██║██████╔╝██████╔╝███████║██████╔╝██║█████╗  ███████╗
██║██║╚██╔╝██║██╔═══╝ ██║   ██║██╔══██╗   ██║   ██║██║╚██╗██║██║   ██║   ██║     ██║██╔══██╗██╔══██╗██╔══██║██╔══██╗██║██╔══╝  ╚════██║
██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║   ██║   ██║██║ ╚████║╚██████╔╝   ███████╗██║██████╔╝██║  ██║██║  ██║██║  ██║██║███████╗███████║
╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝    ╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝ 
'''

# Accessing data
import numpy as np
import pandas as pd
# Paths and dirs
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')
# Train test split
from sklearn.model_selection import train_test_split
# Pipeline
from sklearn.pipeline import Pipeline
                                                        # ╔══╗   ╔══╗ ╔═══════╗ ╔═══════╗ ╔════════╗╔══╗     ╔═══════╗
from sklearn.linear_model import ElasticNet             # ║  ╚╗ ╔╝  ║╔╝ ╔═══╗ ╚╗║  ╔══╗ ╚╗║ ╔══════╝║  ║     ║ ╔═════╝
from xgboost import XGBRegressor                        # ║ ╔╗╚═╝╔╗ ║║  ║   ║  ║║  ║  ║  ║║ ╚═══╗   ║  ║     ║ ╚═════╗
from lightgbm import LGBMRegressor                      # ║ ║╚╗ ╔╝║ ║║  ║   ║  ║║  ║  ║  ║║ ╔═══╝   ║  ║     ╚═════╗ ║
from catboost import CatBoostRegressor                  # ║ ║ ╚═╝ ║ ║╚╗ ╚═══╝ ╔╝║  ╚══╝ ╔╝║ ╚══════╗║  ╚════╗╔═════╝ ║
                                                        # ╚═╝     ╚═╝ ╚═══════╝ ╚═══════╝ ╚════════╝╚═══════╝╚═══════╝
# hyperparametr tuner                                                        
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
# validation
from sklearn.model_selection import StratifiedKFold, cross_val_score
# Performance metrics
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, precision_recall_curve
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

'''
██████╗  █████╗ ████████╗ █████╗     ██╗      ██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║     ██╔═══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝
██║  ██║███████║   ██║   ███████║    ██║     ██║   ██║███████║██║  ██║██║██╔██╗ ██║██║  ███╗
██║  ██║██╔══██║   ██║   ██╔══██║    ██║     ██║   ██║██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
██████╔╝██║  ██║   ██║   ██║  ██║    ███████╗╚██████╔╝██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝
'''

# Training dataset
train_df = pd.read_csv(r"Processed Datasets/final_train.csv")
# Testing dataset
test_df = pd.read_csv(r"Processed Datasets/final_test.csv")
# Input features for training
training_features = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
# Input features for testing
testing_features = test_df.drop(columns="SK_ID_CURR")
# Output feature for training
target_set = train_df['TARGET']

'''
████████╗██████╗  █████╗ ██╗███╗   ██╗    ████████╗███████╗███████╗████████╗    ███████╗██████╗ ██╗     ██╗████████╗
╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ██╔════╝██╔══██╗██║     ██║╚══██╔══╝
   ██║   ██████╔╝███████║██║██╔██╗ ██║       ██║   █████╗  ███████╗   ██║       ███████╗██████╔╝██║     ██║   ██║
   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║       ██║   ██╔══╝  ╚════██║   ██║       ╚════██║██╔═══╝ ██║     ██║   ██║
   ██║   ██║  ██║██║  ██║██║██║ ╚████║       ██║   ███████╗███████║   ██║       ███████║██║     ███████╗██║   ██║
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝       ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚══════╝╚═╝     ╚══════╝╚═╝   ╚═╝
'''
   
# Splitting is done on training dataset to get model performance
X_train, X_eval, y_train, y_eval = train_test_split(
                                        training_features, target_set,
                                        test_size=0.2, shuffle=True)

datasets = {"train_df": train_df, "test_df": test_df, 
            "training_features": training_features, "testing_features": testing_features, "target_set": target_set,
            "X_train": X_train, "X_eval": X_eval, "y_train": y_train, "y_eval": y_eval}

summary = []

for name, dataset in datasets.items():

    rows = dataset.shape[0]
    cols = dataset.shape[1] if isinstance(dataset, pd.DataFrame) else 1

    summary.append({
        # name of the datasets
        "DATASET": name,
        # 
        "ROWS": rows,
        "COLUMNS": cols,
        "MISSING": dataset.isnull().sum().sum(),
        "NUMERIC": len(dataset.select_dtypes(include="number").columns)
                    if isinstance(dataset, pd.DataFrame) else "-",
        "STRING": len(dataset.select_dtypes(include=["object", "string"]).columns)
                    if isinstance(dataset, pd.DataFrame) else "-",
        "BOOLEAN": len(dataset.select_dtypes(include="bool").columns)
                    if isinstance(dataset, pd.DataFrame) else "-"
    })

summary_df = pd.DataFrame(summary)

console.print("\n\n[bold green]______________________________________ [bold #C7009D]SUMMARY OF DATAS[/][bold green] ______________________________________[/]\n")
console.print(summary_df)
console.print("\n[bold green]______________________________________________________________________________________________")

'''
 ██████╗ ██╗   ██╗████████╗██╗     ██╗███████╗██████╗      █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
██╔═══██╗██║   ██║╚══██╔══╝██║     ██║██╔════╝██╔══██╗    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
██║   ██║██║   ██║   ██║   ██║     ██║█████╗  ██████╔╝    ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
██║   ██║██║   ██║   ██║   ██║     ██║██╔══╝  ██╔══██╗    ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
╚██████╔╝╚██████╔╝   ██║   ███████╗██║███████╗██║  ██║    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
 ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
'''

numeric_cols = train_df.select_dtypes(include='number')

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

    summary.append({
        "Feature": col,
        "Outlier %": outlier_pct,
        "Skew": skew,
        "Missing %": missing_pct,
        "Min": minimum,
        "Max": maximum,
        "Unique": unique,
        "Decision": decision
    })

outlier_summary = (
    pd.DataFrame(summary)
      .sort_values("Outlier %", ascending=False)
      .reset_index(drop=True)
)

console.print("[bold green]______________________________________ [bold #C7009D]OUTLIER ANALYSIS[/][bold green] ______________________________________[/]\n")
console.print(outlier_summary)
console.print("\n[bold green]______________________________________________________________________________________________")

'''
██████╗  █████╗ ██████╗     ██████╗ ██╗   ██╗████████╗██╗     ██╗███████╗██████╗     ██████╗ ███████╗███╗   ███╗ ██████╗ ██╗   ██╗ █████╗ ██╗     
██╔══██╗██╔══██╗██╔══██╗   ██╔═══██╗██║   ██║╚══██╔══╝██║     ██║██╔════╝██╔══██╗    ██╔══██╗██╔════╝████╗ ████║██╔═══██╗██║   ██║██╔══██╗██║     
██████╔╝███████║██║  ██║   ██║   ██║██║   ██║   ██║   ██║     ██║█████╗  ██████╔╝    ██████╔╝█████╗  ██╔████╔██║██║   ██║██║   ██║███████║██║     
██╔══██╗██╔══██║██║  ██║   ██║   ██║██║   ██║   ██║   ██║     ██║██╔══╝  ██╔══██╗    ██╔══██╗██╔══╝  ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██╔══██║██║     
██████╔╝██║  ██║██████╔╝   ╚██████╔╝╚██████╔╝   ██║   ███████╗██║███████╗██║  ██║    ██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ██║  ██║███████╗
╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚══════╝
'''

# Replace known sentinel values
train_df.loc[train_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
test_df.loc[test_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

# Replace infinities
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove impossible values
money_cols = [col for col in train_df.columns if col.startswith("AMT_")] # Negative monetary values

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
    "pos_demand_contracts"
]

for col in count_cols:
    if (train_df[col] < 0).any():
        print(col, (train_df[col] < 0).sum())

'''
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
███    ⚠️ AS I FOUND THAT THERE ARE LOT OF FEATURES CONTAINING OUTLIERS BUT REMOVING THEM AFFECTS THE MODEL PERFORMANCE    ███
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
'''

'''
██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗    ██████╗ ██████╗  ██████╗  ██████╗███████╗███████╗███████╗██╗███╗   ██╗ ██████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝    ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔════╝
██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║       ██████╔╝██████╔╝██║   ██║██║     █████╗  ███████╗███████╗██║██╔██╗ ██║██║  ███╗
██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║       ██╔═══╝ ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║╚════██║██║██║╚██╗██║██║   ██║
██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║       ██║     ██║  ██║╚██████╔╝╚██████╗███████╗███████║███████║██║██║ ╚████║╚██████╔╝
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝       ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝
'''

    