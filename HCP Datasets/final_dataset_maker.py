import pandas as pd

from io import StringIO
def print_dataframe_info(df, name):
    buffer = StringIO()
    df.info(buf=buffer)

    console.print(f"[bold #FF7800]{name}[/bold #FF7800]")
    console.print(buffer.getvalue())


def count_value(value):
    return lambda x: (x == value).sum()

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


def load_datasets():
    # Reading the bureau.csv to dataframe
    bureau = pd.read_csv('HCP Datasets/bureau.csv')

    # Reading the bureau_balance.csv to dataframe
    bureau_balance = pd.read_csv('HCP Datasets/bureau_balance.csv')

    # Reading the previous_applications.csv to dataframe
    previous_applications = pd.read_csv('HCP Datasets/previous_application.csv')

    # Reading the installments_payments.csv to dataframe
    installments_payments = pd.read_csv('HCP Datasets/installments_payments.csv')

    # Reading the credit_card_balance.csv to dataframe
    credit_card_balance = pd.read_csv('HCP Datasets/credit_card_balance.csv')

    # Reading the POS_CASH_balance.csv to dataframe
    pos_cash_balance = pd.read_csv('HCP Datasets/POS_CASH_balance.csv')

    return {
        'bureau': bureau,
        'bureau_balance': bureau_balance,
        'previous_applications': previous_applications,
        'installments_payments': installments_payments,
        'credit_card_balance': credit_card_balance,
        'pos_cash_balance': pos_cash_balance
    }

def aggregate_bureau(bureau_df, bureau_balance_df):

    console.rule("[bold #FF7800]⚡ Bureau Feature Engineering")

    # ----------------------------
    # Validation
    # ----------------------------
    assert "SK_ID_CURR" in bureau_df.columns
    assert "SK_ID_BUREAU" in bureau_df.columns
    assert "SK_ID_BUREAU" in bureau_balance_df.columns

    console.print("[bold #00FFA0]Input Dataset Summary[/bold #00FFA0]")

    print_dataframe_info(bureau_df, "Bureau")
    print_dataframe_info(bureau_balance_df, "Bureau Balance")

    # ----------------------------
    # Bureau Balance Aggregation
    # ----------------------------
    bureau_balance_agg = (
        bureau_balance_df
        .groupby("SK_ID_BUREAU", as_index=False)
        .agg(
            bureau_month_min=("MONTHS_BALANCE", "min"),
            bureau_month_max=("MONTHS_BALANCE", "max"),
            bureau_month_mean=("MONTHS_BALANCE", "mean"),
        )
    )

    assert bureau_balance_agg["SK_ID_BUREAU"].is_unique

    print_dataframe_info(
        bureau_balance_agg,
        "Aggregated Bureau Balance"
    )

    # ----------------------------
    # Merge
    # ----------------------------
    bureau_merged = bureau_df.merge(
        bureau_balance_agg,
        how="left",
        on="SK_ID_BUREAU",
    )

    assert bureau_df.shape[0] == bureau_merged.shape[0]

    print_dataframe_info(
        bureau_merged,
        "Merged Bureau"
    )

    # ----------------------------
    # Customer Aggregation
    # ----------------------------
    bureau_features = (
        bureau_merged
        .groupby("SK_ID_CURR", as_index=False)
        .agg(

            # Loan
            bureau_loan_count=("SK_ID_BUREAU", "count"),
            bureau_active_loans=("CREDIT_ACTIVE", count_value("Active")),
            bureau_closed_loans=("CREDIT_ACTIVE", count_value("Closed")),
            bureau_sold_loans=("CREDIT_ACTIVE", count_value("Sold")),
            bureau_bad_debt_loans=("CREDIT_ACTIVE", count_value("Bad debt")),

            # Bureau Balance
            bureau_month_min=("bureau_month_min", "min"),
            bureau_month_max=("bureau_month_max", "max"),
            bureau_month_mean=("bureau_month_mean", "mean"),

            # Credit
            bureau_total_credit=("AMT_CREDIT_SUM", "sum"),
            bureau_avg_credit=("AMT_CREDIT_SUM", "mean"),
            bureau_max_credit=("AMT_CREDIT_SUM", "max"),
            bureau_min_credit=("AMT_CREDIT_SUM", "min"),

            # Credit Age
            bureau_oldest_credit=("DAYS_CREDIT", "min"),
            bureau_latest_credit=("DAYS_CREDIT", "max"),
            bureau_avg_credit_age=("DAYS_CREDIT", "mean"),

            # Credit Limit
            bureau_total_credit_limit=("AMT_CREDIT_SUM_LIMIT", "sum"),
            bureau_avg_credit_limit=("AMT_CREDIT_SUM_LIMIT", "mean"),

            # Prolong
            bureau_credit_prolong_count=("CNT_CREDIT_PROLONG", "sum"),

            # Updates
            bureau_last_update=("DAYS_CREDIT_UPDATE", "max"),
            bureau_avg_end_date=("DAYS_CREDIT_ENDDATE", "mean"),

            # Debt
            bureau_total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
            bureau_avg_debt=("AMT_CREDIT_SUM_DEBT", "mean"),
            bureau_max_debt=("AMT_CREDIT_SUM_DEBT", "max"),

            # Overdue
            bureau_total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            bureau_avg_overdue=("AMT_CREDIT_SUM_OVERDUE", "mean"),
            bureau_max_overdue=("AMT_CREDIT_SUM_OVERDUE", "max"),
        )
    )

    assert bureau_features["SK_ID_CURR"].is_unique

    console.print(
        f"[bold #92FF03]Generated Features :[/bold #92FF03] {bureau_features.shape[1]-1}"
    )

    print_dataframe_info(
        bureau_features,
        "Bureau Features"
    )

    console.rule("[bold green]✅ Bureau Feature Engineering Complete")

    return bureau_features
    

data = load_datasets()

with Progress(
    SpinnerColumn(style="#FF7800"),
    TextColumn("[bold cyan]{task.description}"),
    BarColumn(bar_width=40),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console,
) as progress:

    task = progress.add_task(
        "[#FF7800]Feature Engineering Pipeline",
        total=5
    )

    bureau_agg = aggregate_bureau(
        data["bureau"],
        data["bureau_balance"]
    )
    progress.advance(task)

    # previous_agg = aggregate_previous_application(...)
    progress.advance(task)

    # installment_agg = aggregate_installments(...)
    progress.advance(task)

    # credit_card_agg = aggregate_credit_card(...)
    progress.advance(task)

    # pos_cash_agg = aggregate_pos_cash(...)
    progress.advance(task)

console.print("\n[bold green]🎉 Feature Engineering Pipeline Completed Successfully![/bold green]")

bureau_agg = aggregate_bureau(data['bureau'], data['bureau_balance'])

