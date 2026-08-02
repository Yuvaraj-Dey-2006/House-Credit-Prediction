 
# ██╗███╗   ███╗██████╗  ██████╗ ██████╗ ████████╗██╗███╗   ██╗ ██████╗    ██╗     ██╗██████╗ ██████╗  █████╗ ██████╗ ██╗███████╗███████╗ 
# ██║████╗ ████║██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██║████╗  ██║██╔════╝    ██║     ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝
# ██║██╔████╔██║██████╔╝██║   ██║██████╔╝   ██║   ██║██╔██╗ ██║██║  ███╗   ██║     ██║██████╔╝██████╔╝███████║██████╔╝██║█████╗  ███████╗
# ██║██║╚██╔╝██║██╔═══╝ ██║   ██║██╔══██╗   ██║   ██║██║╚██╗██║██║   ██║   ██║     ██║██╔══██╗██╔══██╗██╔══██║██╔══██╗██║██╔══╝  ╚════██║
# ██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║   ██║   ██║██║ ╚████║╚██████╔╝   ███████╗██║██████╔╝██║  ██║██║  ██║██║  ██║██║███████╗███████║
# ╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝    ╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝ 

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
test_df = pd.read_csv(r"Processed Datasets/final_train.csv")
# Input features for training
training_features = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
# Input features for testing
testing_features = test_df
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
        "DATASET": name,
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

console.print("__________________________ SUMMARY OF DATAS __________________________\n")
console.print(summary_df)
console.print("\n______________________________________________________________________")

