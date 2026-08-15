"""
 ██████╗ ██████╗ ████████╗██╗   ██╗███╗   ██╗ █████╗ 
██╔═══██╗██╔══██╗╚══██╔══╝██║   ██║████╗  ██║██╔══██╗
██║   ██║██████╔╝   ██║   ██║   ██║██╔██╗ ██║███████║
██║   ██║██╔═══╝    ██║   ██║   ██║██║╚██╗██║██╔══██║
╚██████╔╝██║        ██║   ╚██████╔╝██║ ╚████║██║  ██║
 ╚═════╝ ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝

Hyperparameter tuning using Optuna — but only for the TOP 2 models from the baseline run
(ranked by ROC-AUC), so trial budget isn't wasted on models that were already weaker.
Loads the exact train/val split saved by the baseline script — no re-splitting, so tuning
is evaluated on the same val set the baseline used.

This script's ONLY job is to search and save best hyperparameters (+ the val ROC-AUC each
one achieved) to BEST_PARAMS_PATH. It does not refit final models, does not touch X_eval,
and does not produce a submission. That's handled by a separate downstream script, which
loads BEST_PARAMS_PATH, picks the better of the two tuned models, refits it on the FULL
training data (train + val + eval combined), and predicts on the real test set.
"""

import numpy as np
import pandas as pd
import warnings
import joblib
import optuna

from sklearn.exceptions import ConvergenceWarning
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback, CatBoostPruningCallback

from sklearn.metrics import roc_auc_score

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from config import (
    SPLIT_DATA_PATH,
    PREPROCESSORS_PATH,
    BASELINE_RESULTS_PATH,
    BEST_PARAMS_PATH,
    OPTUNA_DB_PATH,
    N_TRIALS,
)

console = Console()

progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console,
)


def finish_sub_task(progress_obj, task_id, label, total):
    progress_obj.update(task_id, completed=total)
    console.print(f"[#BDFF08]{label} • Complete[/]  " + "━" * 40 + "  100%")
    progress_obj.remove_task(task_id)


def make_rich_trial_callback(progress_obj, task_id, parent_task_id):
    """Optuna calls this after every trial (pruned, failed, or completed) —
    advance the per-model bar AND the overall pipeline bar by 1."""
    def _callback(study, trial):
        progress_obj.update(task_id, advance=1)
        progress_obj.update(parent_task_id, advance=1)
    return _callback

"""
██╗      ██████╗  █████╗ ██████╗ 
██║     ██╔═══██╗██╔══██╗██╔══██╗
██║     ██║   ██║███████║██║  ██║
██║     ██║   ██║██╔══██║██║  ██║
███████╗╚██████╔╝██║  ██║██████╔╝
╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ 
"""

# Load the exact split and preprocessors saved by the baseline run — reuse, don't recompute
split_data = joblib.load(SPLIT_DATA_PATH)
X_train, X_val = split_data["X_train"], split_data["X_val"]
y_train, y_val = split_data["y_train"], split_data["y_val"]
# X_eval / y_eval intentionally not loaded — this script never touches the held-out eval set

preprocessors = joblib.load(PREPROCESSORS_PATH)
pp_elasticnet = preprocessors["pp_elasticnet"]
pp_xg = preprocessors["pp_xg"]
pp_lgbm = preprocessors["pp_lgbm"]
pp_catbc = preprocessors["pp_catbc"]

category_cols_X_train = X_train.select_dtypes(include=["object", "category", "str"]).columns

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

console.print(f"[cyan]Loaded split → Train: {len(X_train)} | Val: {len(X_val)}[/]\n")

# Map display names (as they appear in baseline_result) to internal short keys
MODEL_NAME_TO_KEY = {
    "Elastic Net log reg": "sgd",
    "XG Boost Classifier": "xgb",
    "LightGBM Classifier": "lgbm",
    "CatBoost Classifier": "catbc",
}

# Only tune the top 2 baseline models by ROC-AUC — skip wasting trials on the weaker two
baseline_result = joblib.load(BASELINE_RESULTS_PATH)
top2_names = baseline_result.nlargest(2, "ROC-AUC")["Models"].tolist()
top2_keys = [MODEL_NAME_TO_KEY[name] for name in top2_names]

"""
████████╗██████╗  █████╗ ███╗   ██╗███████╗███████╗ ██████╗ ██████╗ ███╗   ███╗
╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║
   ██║   ██████╔╝███████║██╔██╗ ██║███████╗█████╗  ██║   ██║██████╔╝██╔████╔██║
   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║
   ██║   ██║  ██║██║  ██║██║ ╚████║███████║██║      ╚██████╔╝██║  ██║██║ ╚═╝ ██║
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝       ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝

(objective functions — defined here, executed later, inside the progress block)
"""


def objective_sgd(trial):
    params = {
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
    }
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        class_weight=class_weight_dict,
        random_state=42,
        learning_rate="optimal",
        n_jobs=-1,
        **params,
    )
    best_score = -np.inf
    epochs_no_improve = 0
    for epoch in range(100):
        model.partial_fit(X_train_en, y_train, classes=classes)
        val_score = roc_auc_score(y_val, model.predict_proba(X_val_en)[:, 1])
        if val_score > best_score:
            best_score, epochs_no_improve = val_score, 0
        else:
            epochs_no_improve += 1
        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if epochs_no_improve >= 10:
            break
    return best_score


def objective_xgb(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": 1000,
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": neg_count / max(pos_count, 1),
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 100,
    }
    model = XGBClassifier(**params, callbacks=[XGBoostPruningCallback(trial, "validation_0-auc")])
    model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val_xgb)[:, 1])


def objective_lgbm(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": 1000,
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": neg_count / pos_count,
        "metric": "auc",
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = LGBMClassifier(**params)
    model.fit(
        X_train_lgbm,
        y_train,
        categorical_feature=category_cols_X_train.tolist(),
        eval_set=[(X_val_lgbm, y_val)],
        callbacks=[early_stopping(100, verbose=False), LightGBMPruningCallback(trial, "auc")],
    )
    return roc_auc_score(y_val, model.predict_proba(X_val_lgbm)[:, 1])


def objective_catbc(trial):
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": 1000,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 100,
        "allow_writing_files": False,
    }

    pruning_callback = CatBoostPruningCallback(trial, "AUC")

    model = CatBoostClassifier(**params)

    model.fit(
        X_train_catbc, y_train,
        cat_features=category_cols_X_train.tolist(),
        eval_set=(X_val_catbc, y_val),
        callbacks=[pruning_callback],
    )

    pruning_callback.check_pruned()  # must be called after fit — this is what actually raises TrialPruned

    return roc_auc_score(y_val, model.predict_proba(X_val_catbc)[:, 1])


"""
██████╗ ██╗   ██╗███╗   ██╗    ███████╗████████╗██╗   ██╗██████╗ ██╗███████╗███████╗
██╔══██╗██║   ██║████╗  ██║    ██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║██╔════╝██╔════╝
██████╔╝██║   ██║██╔██╗ ██║    ███████╗   ██║   ██║   ██║██║  ██║██║█████╗  ███████╗
██╔══██╗██║   ██║██║╚██╗██║    ╚════██║   ██║   ██║   ██║██║  ██║██║██╔══╝  ╚════██║
██║  ██║╚██████╔╝██║ ╚████║    ███████║   ██║   ╚██████╔╝██████╔╝██║███████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝╚══════╝╚══════╝
"""

all_objectives = {
    "sgd": objective_sgd,
    "xgb": objective_xgb,
    "lgbm": objective_lgbm,
    "catbc": objective_catbc,
}
studies_config = {key: all_objectives[key] for key in top2_keys}
total_trial_budget = N_TRIALS * len(studies_config)

# ═══════════════════════════════════════════════════════════════════════════
# EVERYTHING BELOW RUNS INSIDE THE PROGRESS BAR — baseline ranking display,
# preprocessing, tuning (per-trial), and the final save all advance one
# shared "PIPELINE PROGRESS" bar.
# ═══════════════════════════════════════════════════════════════════════════
with progress:
    parent_task = progress.add_task(
        "[blue]PIPELINE PROGRESS[/]",
        total=total_trial_budget + 2,  # +1 preprocessing, +1 save
    )

    # ── baseline ranking display ──
    ranking_display = baseline_result[["Models", "ROC-AUC"]].sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    RANK_EMOJIS = ["🥇", "🥈", "🥉", "🏅"]  # 4th+ falls back to the generic medal
    ranking_display.insert(0, "Rank", [RANK_EMOJIS[i] if i < len(RANK_EMOJIS) else "🏅" for i in range(len(ranking_display))])

    console.print(f"[bold yellow]Baseline ranking (ROC-AUC):[/]")
    console.print(ranking_display.round(5).to_string(index=False))
    console.print(f"\n[bold yellow]➤ Tuning only the top 2: {top2_names}[/]\n")

    # ── preprocessing ──
    # Re-fit each preprocessor fresh (fit_transform on train, transform on val) —
    # X_eval is deliberately NOT transformed here; this script never touches it.
    X_train_en = pp_elasticnet.fit_transform(X_train)
    X_val_en = pp_elasticnet.transform(X_val)

    X_train_xgb = pp_xg.fit_transform(X_train)
    X_val_xgb = pp_xg.transform(X_val)

    X_train_lgbm = pp_lgbm.fit_transform(X_train)
    X_val_lgbm = pp_lgbm.transform(X_val)

    for col in category_cols_X_train:
        categories = X_train_lgbm[col].astype("category").cat.categories
        X_train_lgbm[col] = pd.Categorical(X_train_lgbm[col], categories=categories)
        X_val_lgbm[col] = pd.Categorical(X_val_lgbm[col], categories=categories)

    X_train_catbc = pp_catbc.fit_transform(X_train)
    X_val_catbc = pp_catbc.transform(X_val)

    for col in category_cols_X_train:
        X_train_catbc[col] = X_train_catbc[col].astype(str)
        X_val_catbc[col] = X_val_catbc[col].astype(str)

    console.print("[bold green]✅ PREPROCESSING COMPLETE[/]\n")
    progress.update(parent_task, advance=1, description="[blue]HYPERPARAMETER SEARCH[/]")

    # ── tuning ──
    best_params = {}
    for name, objective_fn in studies_config.items():
        console.print(f"[bold magenta]▶ Tuning {name.upper()}[/]")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            study_name=f"{name}_home_credit",
            storage=OPTUNA_DB_PATH,
            load_if_exists=True,
        )
        trial_task = progress.add_task(f"[#BDFF08]{name.upper()} • Optuna trials[/]", total=N_TRIALS)
        study.optimize(
            objective_fn,
            n_trials=N_TRIALS,
            callbacks=[make_rich_trial_callback(progress, trial_task, parent_task)],
            show_progress_bar=False,
        )
        finish_sub_task(progress, trial_task, f"{name.upper()} tuning", N_TRIALS)

        # Store both the winning hyperparameters and the val ROC-AUC they achieved —
        # the next script (refit-best-and-predict.py) picks the better of the two from this,
        # without needing to re-run Optuna or recompute anything.
        best_params[name] = {"params": study.best_params, "val_roc_auc": study.best_value}

        console.print(f"[green]  Best ROC-AUC (val): {study.best_value:.5f}[/]")
        console.print(f"[green]  Best params: {study.best_params}[/]\n")

    # ── save ──
    joblib.dump(best_params, BEST_PARAMS_PATH)
    progress.update(parent_task, advance=1, description="[green]TUNING COMPLETE[/]")

# ═══════════════════════════════════════════════════════════════════════════
# summary — outside the bar, bar is closed by now
# ═══════════════════════════════════════════════════════════════════════════
console.print("\n______________________________________ TUNING SUMMARY (VALIDATION ROC-AUC) ______________________________________\n")
for name, info in best_params.items():
    console.print(f"  {name.upper():<8} → {info['val_roc_auc']:.5f}")
console.print("________________________________________________________________________________________________________________\n")

winner = max(best_params, key=lambda k: best_params[k]["val_roc_auc"])
console.print(f"[bold #C7009D]➤ Leading candidate so far (by val ROC-AUC): {winner.upper()}[/]")
console.print(
    "[dim]Note: this is the val-set ranking used for tuning, not a final decision — "
    "confirm on held-out eval in the next script before committing.[/]\n"
)

console.print(f"[bold green]✅ Best params saved →[/] {BEST_PARAMS_PATH}\n")