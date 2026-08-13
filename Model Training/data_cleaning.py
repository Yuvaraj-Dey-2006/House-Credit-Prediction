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

# For creating styled console output and progress bars
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Import file paths from configuration
from config import RAW_TRAIN_PATH, RAW_TEST_PATH, CLEANED_TRAIN_PATH, CLEANED_TEST_PATH

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize console for styled output formatting
console = Console()

# Configure progress bar with spinner, bar, percentage, and time display
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

    # Load raw training and test datasets from CSV files
    train_df = pd.read_csv(RAW_TRAIN_PATH)
    test_df = pd.read_csv(RAW_TEST_PATH)

    # Replace anomalous sentinel value (365243) in DAYS_EMPLOYED with NaN for proper handling
    train_df.loc[train_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    test_df.loc[test_df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # Replace infinite values with NaN to prevent numerical computation issues
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

    # Select only numeric columns for outlier analysis
    numeric_cols = train_df.select_dtypes(include="number")
    summary = []

    # Process each numeric column to detect outliers and calculate statistics
    for col in numeric_cols.columns:
        # Calculate IQR (Interquartile Range) for outlier detection
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier boundaries using 1.5 * IQR rule
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (train_df[col] < lower) | (train_df[col] > upper)
        outlier_count = outlier_mask.sum()
        outlier_pct = round(outlier_count / len(train_df) * 100, 2)

        # Calculate distribution and missing data metrics
        skew = round(train_df[col].skew(), 2)
        missing_pct = round(train_df[col].isna().mean() * 100, 2)
        minimum = train_df[col].min()
        maximum = train_df[col].max()
        unique = train_df[col].nunique()

        # Determine action decision for each feature based on characteristics
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

        # Collect statistics for summary report
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

    # Create summary dataframe sorted by outlier percentage (descending) for easy review
    outlier_summary = (
        pd.DataFrame(summary)
        .sort_values("Outlier %", ascending=False)
        .reset_index(drop=True)
    )

    # Display formatted outlier analysis summary report
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

    progress.update(
        parent_task, description="[#00FFFF]SAVING CLEANED DATA[/]", completed=90
    )

    # Export cleaned training and test datasets to CSV files without index column
    train_df.to_csv(CLEANED_TRAIN_PATH, index=False)
    test_df.to_csv(CLEANED_TEST_PATH, index=False)

    # Display confirmation message with paths to saved cleaned data files
    console.print(
        f"[bold green]✅ Cleaned data saved →[/] [underline #00FFFF]{CLEANED_TRAIN_PATH}[/], [underline #00FFFF]{CLEANED_TEST_PATH}[/]\n"
    )

    progress.update(
        parent_task, description="[green]DATA CLEANING COMPLETE[/]", completed=100
    )
