"""
██╗███╗   ███╗██████╗  ██████╗ ██████╗ ████████╗██╗███╗   ██╗ ██████╗    ██╗     ██╗██████╗ ██████╗  █████╗ ██████╗ ██╗███████╗███████╗
██║████╗ ████║██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██║████╗  ██║██╔════╝    ██║     ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝
██║██╔████╔██║██████╔╝██║   ██║██████╔╝   ██║   ██║██╔██╗ ██║██║  ███╗   ██║     ██║██████╔╝██████╔╝███████║██████╔╝██║█████╗  ███████╗
██║██║╚██╔╝██║██╔═══╝ ██║   ██║██╔══██╗   ██║   ██║██║╚██╗██║██║   ██║   ██║     ██║██╔══██╗██╔══██╗██╔══██║██╔══██╗██║██╔══╝  ╚════██║
██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║   ██║   ██║██║ ╚████║╚██████╔╝   ███████╗██║██████╔╝██║  ██║██║  ██║██║  ██║██║███████╗███████║
╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝    ╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝
"""

import numpy as np
import pandas as pd
import warnings
# for progress bar
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
# defining the paths
from config import RAW_TRAIN_PATH, RAW_TEST_PATH, CLEANED_TRAIN_PATH, CLEANED_TEST_PATH

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

with progress:

    parent_task = progress.add_task("[#C7009D]DATA CLEANING PIPELINE[/]", total=100)

    """
    ██████╗  █████╗ ████████╗ █████╗     ██╗      ██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗
    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║     ██╔═══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝
    ██║  ██║███████║   ██║   ███████║    ██║     ██║   ██║███████║██║  ██║██║██╔██╗ ██║██║  ███╗
    ██║  ██║██╔══██║   ██║   ██╔══██║    ██║     ██║   ██║██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
    ██████╔╝██║  ██║   ██║   ██║  ██║    ███████╗╚██████╔╝██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝
    """

    progress.update(parent_task, description="[#00FFFF]DATA LOADING[/]")

    train_df = pd.read_csv(RAW_TRAIN_PATH)
    test_df = pd.read_csv(RAW_TEST_PATH)

    # Replace known sentinel values
    train_df.loc[train_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    test_df.loc[test_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # Replace infinities
    train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    progress.update(parent_task, completed=25)

    """
     ██████╗ ██╗   ██╗████████╗██╗     ██╗███████╗██████╗      █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
    ██╔═══██╗██║   ██║╚══██╔══╝██║     ██║██╔════╝██╔══██╗    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
    ██║   ██║██║   ██║   ██║   ██║     ██║█████╗  ██████╔╝    ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
    ██║   ██║██║   ██║   ██║   ██║     ██║██╔══╝  ██╔══██╗    ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
    ╚██████╔╝╚██████╔╝   ██║   ███████╗██║███████╗██║  ██║    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
     ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
    """

    progress.update(parent_task, description="[#00FFFF]DATA CLEANING[/]", completed=40)

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

    progress.update(parent_task, completed=75)

    """
    ███████╗ █████╗ ██╗   ██╗███████╗
    ██╔════╝██╔══██╗██║   ██║██╔════╝
    ███████╗███████║██║   ██║█████╗
    ╚════██║██╔══██║╚██╗ ██╔╝██╔══╝
    ███████║██║  ██║ ╚████╔╝ ███████╗
    ╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝
    """

    progress.update(parent_task, description="[#00FFFF]SAVING CLEANED DATA[/]", completed=90)

    train_df.to_csv(CLEANED_TRAIN_PATH, index=False)
    test_df.to_csv(CLEANED_TEST_PATH, index=False)

    console.print(
        f"[bold green]✅ Cleaned data saved →[/] [underline #00FFFF]{CLEANED_TRAIN_PATH}[/], [underline #00FFFF]{CLEANED_TEST_PATH}[/]\n"
    )

    progress.update(
        parent_task, description="[green]DATA CLEANING COMPLETE[/]", completed=100
    )
