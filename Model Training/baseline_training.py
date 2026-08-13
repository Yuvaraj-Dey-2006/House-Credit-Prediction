"""
██████╗  █████╗ ████████╗ █████╗     ██╗      ██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║     ██╔═══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝
██║  ██║███████║   ██║   ███████║    ██║     ██║   ██║███████║██║  ██║██║██╔██╗ ██║██║  ███╗
██║  ██║██╔══██║   ██║   ██╔══██║    ██║     ██║   ██║██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
██████╔╝██║  ██║   ██║   ██║  ██║    ███████╗╚██████╔╝██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝
"""

import numpy as np
import pandas as pd
import warnings
import joblib

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config import (
    CLEANED_TRAIN_PATH,
    SPLIT_DATA_PATH,
    PREPROCESSORS_PATH,
    BASELINE_MODELS_PATH,
    BASELINE_RESULTS_PATH,
)

# Initialize console for rich formatting and progress display
console = Console()

progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console,
)


# Helper function to mark a training subtask as complete and remove it from progress display
def finish_sub_task(progress_obj, task_id, label):
    progress_obj.update(task_id, completed=100)
    console.print(f"[#BDFF08]{label} • Complete[/]  " + "━" * 40 + "  100%")
    progress_obj.remove_task(task_id)


# Custom callback to update progress bar for XGBoost training iterations
class XGBRichProgress(TrainingCallback):
    def __init__(self, progress, task, total):
        self.progress = progress
        self.task = task
        self.total = total  # max possible rounds — used only for the % calc

    def after_iteration(self, model, epoch, evals_log):
        pct = min((epoch + 1) / self.total * 100, 100)
        self.progress.update(self.task, completed=pct)  # total is never touched
        return False  # XGBoost: False = keep training


# Custom callback to update progress bar for CatBoost training iterations
class CatBoostRichProgress:
    def __init__(self, progress, task, total):
        self.progress = progress
        self.task = task
        self.total = total

    def after_iteration(self, info):
        pct = min((info.iteration + 1) / self.total * 100, 100)
        self.progress.update(self.task, completed=pct)
        return True  # CatBoost: True = keep training (opposite convention to XGBoost)


with progress:

    parent_task = progress.add_task("[#C7009D]BASELINE TRAINING PIPELINE[/]", total=100)

    """
    ██████╗  █████╗ ████████╗ █████╗     ██╗      ██████╗  █████╗ ██████╗
    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║     ██╔═══██╗██╔══██╗██╔══██╗
    ██║  ██║███████║   ██║   ███████║    ██║     ██║   ██║███████║██║  ██║
    ██║  ██║██╔══██║   ██║   ██╔══██║    ██║     ██║   ██║██╔══██║██║  ██║
    ██████╔╝██║  ██║   ██║   ██║  ██║    ███████╗╚██████╔╝██║  ██║██████╔╝
    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝
    """

    progress.update(parent_task, description="[#00FFFF]LOADING CLEANED DATA[/]")

    # Load preprocessed training data from cleaned CSV file
    train_df = pd.read_csv(CLEANED_TRAIN_PATH)

    progress.update(parent_task, completed=5)

    """
    ████████╗██████╗  █████╗ ██╗███╗   ██╗    ████████╗███████╗███████╗████████╗    ███████╗██████╗ ██╗     ██╗████████╗
    ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ██╔════╝██╔══██╗██║     ██║╚══██╔══╝
       ██║   ██████╔╝███████║██║██╔██╗ ██║       ██║   █████╗  ███████╗   ██║       ███████╗██████╔╝██║     ██║   ██║
       ██║   ██╔══██╗██╔══██║██║██║╚██╗██║       ██║   ██╔══╝  ╚════██║   ██║       ╚════██║██╔═══╝ ██║     ██║   ██║
       ██║   ██║  ██║██║  ██║██║██║ ╚████║       ██║   ███████╗███████║   ██║       ███████║██║     ███████╗██║   ██║
       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝       ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚══════╝╚═╝     ╚══════╝╚═╝   ╚═╝
    """

    progress.update(parent_task, description="[blue]DATA SPLITTING[/]")

    # Separate features from target variable, removing ID column
    training_features = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    target_set = train_df["TARGET"]

    # Split data into train (70%), validation (15%), and evaluation (15%) sets while maintaining class distribution
    X_train, X_temp, y_train, y_temp = train_test_split(
        training_features,
        target_set,
        test_size=0.30,
        shuffle=True,
        stratify=target_set,
        random_state=42,
    )

    # Further split temporary set into validation and evaluation sets
    X_val, X_eval, y_val, y_eval = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        shuffle=True,
        stratify=y_temp,
        random_state=42,
    )

    progress.update(parent_task, completed=15)

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

    # Identify numeric and categorical feature columns for selective preprocessing
    num_cols_X_train = X_train.select_dtypes(include=np.number).columns
    category_cols_X_train = X_train.select_dtypes(
        include=["object", "category", "str"]
    ).columns

    # Create preprocessing pipeline for Elastic Net: impute numeric features with median and scale using RobustScaler
    # For categorical features: impute with most frequent value and apply one-hot encoding
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

    # Create preprocessing pipeline for XGBoost: impute numeric features with median, categorical with one-hot encoding (sparse format)
    num_col_pipeline_xg = Pipeline([("imputer", SimpleImputer(strategy="median"))])
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

    # Create preprocessing pipeline for LightGBM: impute features but keep categorical data as is (LightGBM handles categories natively)
    pp_lgbm = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols_X_train),
            ("cat", SimpleImputer(strategy="most_frequent"), category_cols_X_train),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pp_lgbm.set_output(transform="pandas")

    # Apply preprocessing to all data splits
    X_train_lgbm = pp_lgbm.fit_transform(X_train)
    X_val_lgbm = pp_lgbm.transform(X_val)
    X_eval_lgbm = pp_lgbm.transform(X_eval)

    # Convert categorical columns to pandas Categorical type for LightGBM's categorical feature handling
    for col in category_cols_X_train:
        categories = X_train_lgbm[col].astype("category").cat.categories

        X_train_lgbm[col] = pd.Categorical(
            X_train_lgbm[col],
            categories=categories,
        )

        X_val_lgbm[col] = pd.Categorical(
            X_val_lgbm[col],
            categories=categories,
        )

        X_eval_lgbm[col] = pd.Categorical(
            X_eval_lgbm[col],
            categories=categories,
        )

    # Count class occurrences to handle class imbalance in model training
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    # Create preprocessing pipeline for CatBoost: impute features without encoding (CatBoost handles categorical features natively)
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

    # Apply preprocessing to all data splits for CatBoost
    X_train_catbc = pp_catbc.fit_transform(X_train)
    X_val_catbc = pp_catbc.transform(X_val)
    X_eval_catbc = pp_catbc.transform(X_eval)

    # Convert categorical columns to string type for CatBoost's categorical feature handling
    for col in category_cols_X_train:
        X_train_catbc[col] = X_train_catbc[col].astype(str)
        X_val_catbc[col] = X_val_catbc[col].astype(str)
        X_eval_catbc[col] = X_eval_catbc[col].astype(str)

    console.print("[bold green]✅ PREPROCESSING COMPLETED[/]\n\n")
    progress.update(parent_task, completed=30)

    """
    ██████╗  █████╗ ███████╗███████╗██╗     ██╗███╗   ██╗███████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     ███████╗
    ██╔══██╗██╔══██╗██╔════╝██╔════╝██║     ██║████╗  ██║██╔════╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔════╝
    ██████╔╝███████║███████╗█████╗  ██║     ██║██╔██╗ ██║█████╗      ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ███████╗
    ██╔══██╗██╔══██║╚════██║██╔══╝  ██║     ██║██║╚██╗██║██╔══╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ╚════██║
    ██████╔╝██║  ██║███████║███████╗███████╗██║██║ ╚████║███████╗    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗███████║
    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝
    """

    progress.update(parent_task, description="[magenta]PREPARING BASELINE MODELS[/]")

    # Apply Elastic Net preprocessing to all data splits
    X_train_en = pp_elasticnet.fit_transform(X_train)
    X_val_en = pp_elasticnet.transform(X_val)
    X_eval_en = pp_elasticnet.transform(X_eval)

    # Calculate class weights to address class imbalance in the training data
    classes = np.array([0, 1])
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))

    # Initialize Elastic Net (SGD) classifier with log loss and elasticnet regularization
    sgd_base = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        l1_ratio=0.5,
        alpha=1e-4,
        class_weight=class_weight_dict,
        random_state=42,
        learning_rate="optimal",
        n_jobs=-1,
    )

    # Apply XGBoost preprocessing to all data splits
    X_train_xgb = pp_xg.fit_transform(X_train)
    X_val_xgb = pp_xg.transform(X_val)
    X_eval_xgb = pp_xg.transform(X_eval)

    # Initialize XGBoost classifier with early stopping and class weight balancing
    xgbc_base = XGBClassifier(
        random_state=42,
        eval_metric="auc",
        scale_pos_weight=neg_count / max(pos_count, 1),
        n_estimators=1000,
        tree_method="hist",
        early_stopping_rounds=250,
        n_jobs=-1,
    )

    # Initialize LightGBM classifier with AUC metric and scale_pos_weight for class imbalance
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

    # Initialize CatBoost classifier with auto class weights and early stopping
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
    progress.update(parent_task, completed=35)

    """
    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║            ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║            ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗       ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝
    """

    # Train Elastic Net using incremental learning with early stopping based on validation ROC-AUC
    sgd_task = progress.add_task(
        "[#BDFF08]Elastic Net (SGD) • Fitting...[/]", total=100
    )

    max_epochs = 200
    patience = 10
    best_score = -np.inf
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # Incrementally fit SGD on mini-batches
        sgd_base.partial_fit(
            X_train_en,
            y_train,
            classes=classes,
        )

        # Evaluate on validation set and check for improvement
        val_prob = sgd_base.predict_proba(X_val_en)[:, 1]
        val_score = roc_auc_score(y_val, val_prob)

        pct = min((epoch + 1) / max_epochs * 100, 100)
        progress.update(sgd_task, completed=pct)

        # Implement early stopping if validation score doesn't improve
        if val_score > best_score:
            best_score = val_score
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    finish_sub_task(progress, sgd_task, "Elastic Net (SGD)")

    # Train XGBoost classifier with custom progress callback for real-time visualization
    xgb_task = progress.add_task("[#BDFF08]XGBoost • Fitting...[/]", total=100)

    xgbc_base.callbacks = [XGBRichProgress(progress, xgb_task, 1000)]

    xgbc_base.fit(
        X_train_xgb,
        y_train,
        eval_set=[(X_val_xgb, y_val)],
        verbose=False,
    )
    finish_sub_task(progress, xgb_task, "XGBoost")

    # Clear callback reference to avoid serialization issues during model saving
    xgbc_base.callbacks = None

    progress.update(parent_task, completed=65)

    # Train LightGBM classifier with categorical features and early stopping
    lgbm_task = progress.add_task("[#BDFF08]LightGBM • Fitting...[/]", total=100)

    # Define callback to update progress bar during LightGBM training
    def lgbm_rich_progress(env):
        if env.evaluation_result_list:
            pct = min((env.iteration + 1) / env.end_iteration * 100, 100)
            progress.update(lgbm_task, completed=pct)

    lgbmc_base.fit(
        X_train_lgbm,
        y_train,
        categorical_feature=category_cols_X_train.tolist(),
        eval_set=[(X_val_lgbm, y_val)],
        callbacks=[
            lgbm_rich_progress,
            early_stopping(250, verbose=False),
        ],
    )

    finish_sub_task(progress, lgbm_task, "LightGBM")

    progress.update(parent_task, completed=78)

    # Train CatBoost classifier with native categorical feature handling and custom progress tracking
    catboost_task = progress.add_task("[#BDFF08]CatBoost • Fitting...[/]", total=100)

    catbc_base.fit(
        X_train_catbc,
        y_train,
        cat_features=category_cols_X_train.tolist(),
        eval_set=(X_val_catbc, y_val),
        callbacks=[CatBoostRichProgress(progress, catboost_task, 1000)],
    )

    finish_sub_task(progress, catboost_task, "CatBoost")

    progress.update(parent_task, completed=85)

    """
    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗   ██║   ██║██║   ██║████╗  ██║
    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║         █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║         ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗    ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    """

    progress.update(
        parent_task, description="[yellow]MODEL EVALUATION[/]", completed=88
    )

    # Generate predictions and probabilities for all models on evaluation set
    y_pred_en = sgd_base.predict(X_eval_en)
    y_prob_en = sgd_base.predict_proba(X_eval_en)[:, 1]

    y_pred_xgbc = xgbc_base.predict(X_eval_xgb)
    y_prob_xgbc = xgbc_base.predict_proba(X_eval_xgb)[:, 1]

    y_pred_lgbmc = lgbmc_base.predict(X_eval_lgbm)
    y_prob_lgbmc = lgbmc_base.predict_proba(X_eval_lgbm)[:, 1]

    y_pred_catbc = catbc_base.predict(X_eval_catbc)
    y_prob_catbc = catbc_base.predict_proba(X_eval_catbc)[:, 1]

    progress.update(
        parent_task, description="[yellow]METRIC CALCULATION[/]", completed=92
    )

    # Calculate comprehensive metrics for each model: Accuracy, Precision, Recall, F1, and ROC-AUC
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

    # Identify the best performing model based on ROC-AUC score
    best_model_name = base_result.loc[base_result["ROC-AUC"].idxmax(), "Models"]
    console.print(
        f"\n[bold #C7009D]➤ BEST BASELINE MODEL (by ROC-AUC): {best_model_name}[/]\n"
    )

    """
    ███████╗ █████╗ ██╗   ██╗██╗███╗   ██╗ ██████╗
    ██╔════╝██╔══██╗██║   ██║██║████╗  ██║██╔════╝
    ███████╗███████║██║   ██║██║██╔██╗ ██║██║  ███╗
    ╚════██║██╔══██║╚██╗ ██╔╝██║██║╚██╗██║██║   ██║
    ███████║██║  ██║ ╚████╔╝ ██║██║ ╚████║╚██████╔╝
    ╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝
    """

    progress.update(
        parent_task, description="[#00FFFF]SAVING ARTIFACTS[/]", completed=96
    )

    # Save train/validation/evaluation data splits for reproducibility
    joblib.dump(
        {
            "X_train": X_train,
            "X_val": X_val,
            "X_eval": X_eval,
            "y_train": y_train,
            "y_val": y_val,
            "y_eval": y_eval,
        },
        SPLIT_DATA_PATH,
    )

    # Save all preprocessing pipelines for consistent feature transformation during inference
    joblib.dump(
        {
            "pp_elasticnet": pp_elasticnet,
            "pp_xg": pp_xg,
            "pp_lgbm": pp_lgbm,
            "pp_catbc": pp_catbc,
        },
        PREPROCESSORS_PATH,
    )

    # Save all trained baseline models for comparison and further tuning
    joblib.dump(
        {
            "sgd_base": sgd_base,
            "xgbc_base": xgbc_base,
            "lgbmc_base": lgbmc_base,
            "catbc_base": catbc_base,
        },
        BASELINE_MODELS_PATH,
    )

    # Save model performance metrics for analysis
    joblib.dump(base_result, BASELINE_RESULTS_PATH)

    console.print(
        f"[bold green]✅ Artifacts saved →[/] {SPLIT_DATA_PATH}, {PREPROCESSORS_PATH}, "
        f"{BASELINE_MODELS_PATH}, {BASELINE_RESULTS_PATH}\n"
    )

    progress.update(
        parent_task, description="[green]BASELINE TRAINING COMPLETE[/]", completed=100
    )
