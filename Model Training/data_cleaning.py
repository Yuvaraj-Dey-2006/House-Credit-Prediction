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

warnings.filterwarnings("ignore", category=FutureWarning)
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
    LGBMClassifier, early_stopping
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

progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)

with progress:

    elastic_task = progress.add_task(
        "[#BDFF08]Elastic Net[/]",
        total=100,
        visible=False
    )

    xgb_task = progress.add_task(
        "[#BDFF08]XGBoost[/]",
        total=100,
        visible=False
    )

    lgbm_task = progress.add_task(
        "[#BDFF08]LightGBM[/]",
        total=100,
        visible=False
    )

    catboost_task = progress.add_task(
        "[#BDFF08]CatBoost[/]",
        total=100,
        visible=False
    )

    parent_task = progress.add_task(
        "[#C7009D]MODEL TRAINING PIPELINE[/]",
        total=100
    )

    """
    ██████╗  █████╗ ████████╗ █████╗     ██╗      ██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗
    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║     ██╔═══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝
    ██║  ██║███████║   ██║   ███████║    ██║     ██║   ██║███████║██║  ██║██║██╔██╗ ██║██║  ███╗
    ██║  ██║██╔══██║   ██║   ██╔══██║    ██║     ██║   ██║██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
    ██████╔╝██║  ██║   ██║   ██║  ██║    ███████╗╚██████╔╝██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝
    """

    

    progress.update(
    parent_task,
    description="[#00FFFF]DATA LOADING[/]"
    )

    # Training dataset
    train_df = pd.read_csv(r"Processed Datasets/final_train.csv")
    # Testing dataset
    test_df = pd.read_csv(r"Processed Datasets/final_test.csv")

    progress.update(parent_task, completed=10)

    """
     ██████╗ ██╗   ██╗████████╗██╗     ██╗███████╗██████╗      █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
    ██╔═══██╗██║   ██║╚══██╔══╝██║     ██║██╔════╝██╔══██╗    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
    ██║   ██║██║   ██║   ██║   ██║     ██║█████╗  ██████╔╝    ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
    ██║   ██║██║   ██║   ██║   ██║     ██║██╔══╝  ██╔══██╗    ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
    ╚██████╔╝╚██████╔╝   ██║   ███████╗██║███████╗██║  ██║    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
     ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
    """

    progress.update(
        parent_task,
        description="[#00FFFF]DATA CLEANING[/]"
    )

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
        "[bold green]___________________________________________________[/] "
        "[bold #C7009D]SUMMARY OF OUTLIERS[/] "
        "[bold green]___________________________________________________[/]"
    )
    console.print(outlier_summary)
    console.print(
        "[bold green]___________________________________________________________________________________________________________________________\n\n[/]"
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

    progress.update(
        parent_task,
        description="[#00FFFF]DATA SPLITTING[/]"
        )

    # Input features for training
    training_features = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    # Input features for testing
    testing_features = test_df.drop(columns="SK_ID_CURR")
    # Output feature for training
    target_set = train_df["TARGET"]

    progress.update(parent_task, completed=28)

    progress.update(
        parent_task,
        description="[blue]DATA SPLITTING[/]"
        )
    
    # Splitting is done on training dataset to get model performance
    X_train, X_eval, y_train, y_eval = train_test_split(
        training_features,
        target_set,
        test_size=0.2,
        shuffle=True,
        stratify=target_set,
        random_state=42,
    )

    progress.update(parent_task, completed=35)

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

    progress.update(parent_task, completed=39)

    console.print(
        "[bold green]_____________________________________________________[/] "
        "[bold #C7009D]SUMMARY OF DATAS[/] "
        "[bold green]_____________________________________________________[/]"
    )
    console.print(summary_df)
    console.print(
        "[bold green]____________________________________________________________________________________________________________________________[/]"
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
    # list of numeric columns
    num_cols_X_train = X_train.select_dtypes(include=np.number).columns
    # list of catrgorical columns
    category_cols_X_train = X_train.select_dtypes(include=["object", "category"]).columns

    # pipeline for imputing and scaling numerical values for elastic net
    num_col_pipeline_en = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(0.25, 0.75))),
        ]
    )
    # pipeline for imputing and encoding gategorical values for elastic net
    category_col_pipeline_en = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )
    # preprocessing pipeline for elastic net
    pp_elasticnet = ColumnTransformer(
        transformers=[
            ("num_scale", num_col_pipeline_en, num_cols_X_train),
            ("category_scale", category_col_pipeline_en, category_cols_X_train),
        ],
        verbose_feature_names_out=False
    )

    # pipeline for imputing numerical values for XG boost classifier
    num_col_pipeline_xg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    # pipeline for imputing and encoding categorical values for XG boost classifier 
    category_col_pipeline_xg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    # preprocessing pipeline for XG boost classifier 
    pp_xg = ColumnTransformer(
        transformers=[
            ("num_scale", num_col_pipeline_xg, num_cols_X_train),
            ("category_scale", category_col_pipeline_xg, category_cols_X_train),
        ],
        verbose_feature_names_out=False
    )
    # preprocessing pipeline for LGBM classifier
    pp_lgbm = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            num_cols_X_train
        ),
        (
            "cat",
            SimpleImputer(strategy="most_frequent"),
            category_cols_X_train
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False
    )
    # setting the preprocessor lgbm output as pandas dataframe
    pp_lgbm.set_output(transform="pandas")
    # Storing the preprocessed data earlier for lgbm native category identifing
    X_train_lgbm = pp_lgbm.fit_transform(X_train)
    X_eval_lgbm = pp_lgbm.transform(X_eval)
    # Setting the data as category type for lgbm to distinguish
    for col in category_cols_X_train:
        X_train_lgbm[col] = X_train_lgbm[col].astype("category")
        X_eval_lgbm[col] = X_eval_lgbm[col].astype("category")

    # count of true and false values
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    # pipeline for imputing numerical values for Cat Boosting classifier
    num_col_pipeline_catbc = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    # pipeline for imputing catgorical values for Cat Boosting classifier
    categorty_col_pipeline_catbc = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    # column trasformer pipeline for Cat Boosting classifier
    pp_catbc = ColumnTransformer(
        transformers=[
            ("num_scale", num_col_pipeline_catbc, num_cols_X_train),
            ("category_scale", categorty_col_pipeline_catbc, category_cols_X_train),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # setting the output of ColumnTransformer of Cat boost to pandas format
    pp_catbc.set_output(transform="pandas")

    X_train_catbc = pp_catbc.fit_transform(X_train)
    X_eval_catbc = pp_catbc.transform(X_eval)

    for col in category_cols_X_train:
        X_train_catbc[col] = X_train_catbc[col].astype(str)
        X_eval_catbc[col] = X_eval_catbc[col].astype(str)

    progress.update(
    parent_task,
    description="[magenta]PREPROCESSING PIPELINES[/]"
    )
    progress.update(parent_task, completed=45)

    """
    ██████╗  █████╗ ███████╗███████╗██╗     ██╗███╗   ██╗███████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     ███████╗
    ██╔══██╗██╔══██╗██╔════╝██╔════╝██║     ██║████╗  ██║██╔════╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔════╝
    ██████╔╝███████║███████╗█████╗  ██║     ██║██╔██╗ ██║█████╗      ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ███████╗
    ██╔══██╗██╔══██║╚════██║██╔══╝  ██║     ██║██║╚██╗██║██╔══╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ╚════██║
    ██████╔╝██║  ██║███████║███████╗███████╗██║██║ ╚████║███████╗    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗███████║
    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝
    """
    # pipeline for baseline elastic net model
    pipeline_elastic_net = Pipeline(
        [
            ("pp_elasticnet", pp_elasticnet),
            (
                "en_base",
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

    # pipeline for baseline XG boosting classifier model
    pipeline_xgb = Pipeline(
        [
            ("pp_xg", pp_xg),
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

    # baseline LGBM classifier model using native categorial identifier
    lgbmc_base = LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=neg_count / pos_count,
                    random_state=42,
                    verbosity=-1,
                    n_jobs=-1
                )

    

    # pipeline for baseline Cat boosting classifier model
    catbc_base = CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="logloss",
                    eval_metric="AUC",
                    auto_class_weights="balanced",
                    random_seed=42,
                    verbose=0,
                    cat_features=category_cols_X_train.tolist(),
                    allow_writing_files=False,
                )

    '''
    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║            ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║            ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗       ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝
    '''


    # ──────────────────────────────────────────────────────────────────────────────
    # Elastic Net
    # ──────────────────────────────────────────────────────────────────────────────

    # Elastic Net becomes visible
    progress.update(
        elastic_task,
        visible=True,
        description="[#BDFF08]Elastic Net • Training...[/]"
    )

    pipeline_elastic_net.fit(X_train, y_train)

    # Elastic Net finished
    progress.update(
        elastic_task,
        completed=100,
        description="[#BDFF08]Elastic Net • Complete[/]"
    )


    # ──────────────────────────────────────────────────────────────────────────────
    # XGBoost
    # ──────────────────────────────────────────────────────────────────────────────

    progress.update(
    xgb_task,
    visible=True,
    description="[#BDFF08]XGBoost • Training...[/]"
    )

    pipeline_xgb.fit(X_train, y_train)

    progress.update(
        xgb_task,
        completed=100,
        description="[#BDFF08]XGBoost • Complete[/]"
    )


    # ──────────────────────────────────────────────────────────────────────────────
    # LightGBM
    # ──────────────────────────────────────────────────────────────────────────────

    progress.update(
    lgbm_task,
    visible=True,
    description="[#BDFF08]LightGBM • Training...[/]"
    )

    lgbmc_base.fit(X_train_lgbm, y_train,
                       categorical_feature=category_cols_X_train.tolist(),
                       eval_set=[(X_eval_lgbm, y_eval)],
                       callbacks=[early_stopping(100, verbose=False)])

    progress.update(
        lgbm_task,
        completed=100,
        description="[#BDFF08]LightGBM • Complete[/]"
    )


    # ──────────────────────────────────────────────────────────────────────────────
    # CatBoost
    # ──────────────────────────────────────────────────────────────────────────────

    progress.update(
    catboost_task,
    visible=True,
    description="[#BDFF08]CatBoost • Training...[/]"
)

    catbc_base.fit(
            X_train_catbc,
            y_train,
            cat_features=category_cols_X_train.tolist(),
            eval_set=[(X_train_catbc, y_eval)],
            early_stopping_rounds=100
        )

    progress.update(
        catboost_task,
        completed=100,
        description="[#BDFF08]CatBoost • Complete[/]"
    )

    '''
    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║         █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║         ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗    ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    '''

    progress.update(
    parent_task,
    description="[yellow]MODEL EVALUATION[/]",
    completed=80
    )

    # prediction and prediction probability for elastic net
    y_pred_en = pipeline_elastic_net.predict(X_eval)
    y_prob_en = pipeline_elastic_net.predict_proba(X_eval)[:,1]

    # prediction and prediction probability for XG boosting classifier
    y_pred_xgbc = pipeline_xgb.predict(X_eval)
    y_prob_xgbc = pipeline_xgb.predict_proba(X_eval)[:,1]

    # prediction and prediction probability for lgb classifier
    y_pred_lgbmc = lgbmc_base.predict(X_eval_lgbm)
    y_prob_lgbmc = lgbmc_base.predict_proba(X_eval_lgbm)[:,1]

    # prediction and prediction probability for cat boosting classifier
    y_pred_catbc = catbc_base.predict(X_eval_catbc)
    y_prob_catbc = catbc_base.predict_proba(X_eval_catbc)[:,1]

    progress.update(
    parent_task,
    description="[yellow]METRIC CALCULATION[/]",
    completed=90
    )

    # base model evaluation with accuracy, precision, recall, f1, and roc-auc
    base_result = pd.DataFrame({
                        'Models': ['Elastic Net log reg.', 'XG Boost Classifier', 'LightGBM Classifier', 'CatBoost Classifier'],
                        'ACCURACY': [
                            accuracy_score(y_eval, y_pred_en),
                            accuracy_score(y_eval, y_pred_xgbc),
                            accuracy_score(y_eval, y_pred_lgbmc),
                            accuracy_score(y_eval, y_pred_catbc)
                        ],
                        'PRECISION': [
                            precision_score(y_eval, y_pred_en),
                            precision_score(y_eval, y_pred_xgbc),
                            precision_score(y_eval, y_pred_lgbmc),
                            precision_score(y_eval, y_pred_catbc)
                        ],
                        'RECALL': [
                            recall_score(y_eval, y_pred_en),
                            recall_score(y_eval, y_pred_xgbc),
                            recall_score(y_eval, y_pred_lgbmc),
                            recall_score(y_eval, y_pred_catbc)
                        ],
                        'F1': [
                            f1_score(y_eval, y_pred_en),
                            f1_score(y_eval, y_pred_xgbc),
                            f1_score(y_eval, y_pred_lgbmc),
                            f1_score(y_eval, y_pred_catbc)
                        ],
                        'ROC-AUC': [
                            roc_auc_score(y_eval, y_prob_en),
                            roc_auc_score(y_eval, y_prob_xgbc),
                            roc_auc_score(y_eval, y_prob_lgbmc),
                            roc_auc_score(y_eval, y_prob_catbc)
                        ]
                                      })

    console.print("\n______________________________________ BASE MODELS PERFORMANCE (IMBALANCED CLASS) ______________________________________\n")
    console.print(base_result.round(5))
    console.print("________________________________________________________________________________________________________________________\n")

    progress.update(
    parent_task,
    description="[green]MODEL EVALUATION COMPLETE[/]",
    completed=100
    )