# =========================================
#   IMPORTING IMPORTANT MODULES            |================================================================================================================
# =========================================

import numpy as np
import pandas as pd

from pathlib import Path

from io import StringIO

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

# ==========================================================================================================================================================

# =========================================================
#   FUNCTION FOR TO PRINT DATAFRAME INFORMATION            |================================================================================================================
# =========================================================

def print_dataframe_info(df, name):
    buffer = StringIO()
    df.info(buf=buffer)

    console.print(f"[bold #FF7800]{name}[/bold #FF7800]")
    console.print(f"""
        [cyan]Customers :[/] {len(df):,}
        [cyan]Features  :[/] {df.shape[1]-1}
        [cyan]Missing   :[/] {df.isna().sum().sum():,}\n"""
    )
    console.print(buffer.getvalue())

# ==========================================================================================================================================================================

# =========================================================
#   FUNCTION FOR TO COUNT SPECIFIC VALUE IN COLUMN         |================================================================================================================
# =========================================================

def count_value(value):
    return lambda x: (x == value).sum()

# ==========================================================================================================================================================================

console.rule("[bold green]🏃 Feature Engineering Started Successfully![/bold green]")

# =========================================
#   CONVERTING EVERY CSVs TO DATAFRAME     |================================================================================================================
# =========================================

def load_datasets():
    # Reading the application_train to dataframe
    application_train = pd.read_csv("HCP kaggle Datasets/application_train.csv")

    # Reading the application_test to dataframe
    application_test = pd.read_csv("HCP kaggle Datasets/application_test.csv")

    # Reading the bureau.csv to dataframe
    bureau = pd.read_csv('HCP kaggle Datasets/bureau.csv')

    # Reading the bureau_balance.csv to dataframe
    bureau_balance = pd.read_csv('HCP kaggle Datasets/bureau_balance.csv')

    # Reading the previous_applications.csv to dataframe
    previous_applications = pd.read_csv('HCP kaggle Datasets/previous_application.csv')

    # Reading the installments_payments.csv to dataframe
    installments_payments = pd.read_csv('HCP kaggle Datasets/installments_payments.csv')

    # Reading the credit_card_balance.csv to dataframe
    credit_card_balance = pd.read_csv('HCP kaggle Datasets/credit_card_balance.csv')

    # Reading the POS_CASH_balance.csv to dataframe
    pos_cash_balance = pd.read_csv('HCP kaggle Datasets/POS_CASH_balance.csv')

    return {
        'application_train': application_train,
        'application_test': application_test,
        'bureau': bureau,
        'bureau_balance': bureau_balance,
        'previous_applications': previous_applications,
        'installments_payments': installments_payments,
        'credit_card_balance': credit_card_balance,
        'pos_cash_balance': pos_cash_balance
    }
# ==========================================================================================================================================================   

# ==============================================
#   FEATURE ENGINNERING OF BUREAU DATASET       |===========================================================================================================
# ==============================================

def aggregate_bureau(bureau_df, bureau_balance_df):

    console.print("[bold #FF7800]⚡ Bureau Feature Engineering Started")
    
    # --------------------------------
    # Maximum Dependencies
    # --------------------------------
    status_map = {
                   "0": 0,
                   "1": 1,
                   "2": 2,
                   "3": 3,
                   "4": 4,
                   "5": 5,
                   "C": 0,
                   "X": 0
                }
    
    bureau_balance_df["STATUS_NUM"] = bureau_balance_df["STATUS"].map(status_map)

    # ----------------------------
    # Validation
    # ----------------------------
    assert "SK_ID_CURR" in bureau_df.columns
    assert "SK_ID_BUREAU" in bureau_df.columns
    assert "SK_ID_BUREAU" in bureau_balance_df.columns

    console.print("\n[bold #0080FF].....Before Feature Engineering.....")
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

            #Status value count
            status_0_count = ("STATUS", count_value("0")),
            status_1_count = ("STATUS", count_value("1")),
            status_2_count = ("STATUS", count_value("2")),
            status_3_count = ("STATUS", count_value("3")),
            status_4_count = ("STATUS", count_value("4")),
            status_5_count = ("STATUS", count_value("5")),
            status_C_count = ("STATUS", count_value("C")),
            status_X_count = ("STATUS", count_value("X")),

            # Status related features
            bureau_max_status=("STATUS_NUM", "max"),
            bureau_avg_status=("STATUS_NUM", "mean"),

            # total overdue month
            bureau_overdue_months = ("STATUS", lambda x: x.isin(["1", "2", "3", "4", "5"]).sum()),

            # Ever defaulted
            bureau_ever_overdue = ("STATUS", lambda x: int(x.isin(["1", "2", "3", "4", "5"]).any())),

            # Closed Months
            bureau_closed_months = ("STATUS", count_value("C"))

        )
    )

    assert bureau_balance_agg["SK_ID_BUREAU"].is_unique, \
        "Duplicate SK_ID_BUREAU after aggregation."

    print_dataframe_info(bureau_balance_agg, "Aggregated Bureau Balance")

    # ----------------------------
    # Merge
    # ----------------------------
    bureau_merged = bureau_df.merge(bureau_balance_agg, how="left", on="SK_ID_BUREAU")

    assert bureau_df.shape[0] == bureau_merged.shape[0]

    print_dataframe_info(bureau_merged, "Merged Bureau")

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
            bureau_overdue_months=("bureau_overdue_months","sum"),
            bureau_ever_overdue=("bureau_ever_overdue","max"),

            # Status related features
            bureau_total_status0=("status_0_count","sum"),
            bureau_total_status1=("status_1_count","sum"),
            bureau_total_status2=("status_2_count","sum"),
            bureau_total_status3=("status_3_count","sum"),
            bureau_total_status4=("status_4_count","sum"),
            bureau_total_status5=("status_5_count","sum"),

            bureau_max_status=("bureau_max_status","max"),
            bureau_avg_status=("bureau_avg_status","mean"),
            bureau_closed_months=("bureau_closed_months", "sum")        
        ))

    assert bureau_features["SK_ID_CURR"].is_unique, \
        "Duplicate customers found after bureau aggregation!"

    print_dataframe_info(bureau_features, "Bureau Features")

    console.print("[bold green]✅ Bureau Feature Engineering Complete")

    return bureau_features

# ==========================================================================================================================================================

# ==============================================================
#   FEATURE ENGINNERING OF PREVIOUS APPLICATIONS DATASETS       |===========================================================================================
# ==============================================================

def aggregate_previous_applications(previous_application_df):
    console.print("[bold #FF7800]⚡ Previous Application Feature Engineering Started")

    # --------------------------
    #   Validation
    # --------------------------
    assert "SK_ID_CURR" in previous_application_df.columns

    console.print("[bold #0080FF].....Before Aggregation.....")
    print_dataframe_info(previous_application_df, "Previous Applications")

    # ------------------------------------------
    #   PREVIOUS APPLICATIONS AGGREGATION
    # ------------------------------------------
    previous_application_features = (
        previous_application_df
            .groupby("SK_ID_CURR", as_index=False)
            .agg(
                # Application Statics
                prev_application_count = ("SK_ID_PREV", "count"),
                prev_total_application_amount = ("AMT_APPLICATION", "sum"),
                prev_avg_application_amount = ("AMT_APPLICATION", "mean"),
                prev_max_application_amount = ("AMT_APPLICATION", "max"),
                # Credit Statictics
                prev_total_credit = ("AMT_CREDIT", "sum"),
                prev_avg_credit = ("AMT_CREDIT", "mean"),
                prev_max_credit = ("AMT_CREDIT", "max"),
                # Approved vs Refused
                prev_approved_count = ("NAME_CONTRACT_STATUS", count_value("Approved")),
                prev_refused_count = ("NAME_CONTRACT_STATUS", count_value("Refused")),
                prev_canceled_count = ("NAME_CONTRACT_STATUS", count_value("Canceled")),
                prev_unused_offer_count = ("NAME_CONTRACT_STATUS", count_value("Unused offer")),
                # Loan timing,
                prev_last_application_days = ("DAYS_DECISION", "max"),
                prev_first_application_days = ("DAYS_DECISION", "min"),
                prev_avg_application_days = ("DAYS_DECISION", "mean"),
                # Down Payment,
                prev_avg_down_payment =("AMT_DOWN_PAYMENT", "mean"),
                prev_max_down_payment = ("AMT_DOWN_PAYMENT", "max"),
                # Annuity
                prev_avg_annuity = ("AMT_ANNUITY", "mean"),
                prev_max_annuity = ("AMT_ANNUITY", "max"),
                # Good Price
                prev_avg_goods_price = ("AMT_GOODS_PRICE", "mean"),
                prev_total_goods_price = ("AMT_GOODS_PRICE", "sum"),
                # Sellar Area
                prev_avg_seller_area = ("SELLERPLACE_AREA", "mean"),
                # CNT Payment
                prev_avg_payment_term = ("CNT_PAYMENT", "mean"),
                prev_max_payment_term = ("CNT_PAYMENT", "max")
                ))
                
        
    assert previous_application_features["SK_ID_CURR"].is_unique, \
        "Duplicate customers found after previous application aggregation!"
    
    console.print("[bold #0080FF].....After Aggregation.....")
    print_dataframe_info(previous_application_features, "Previous Applications")

    console.print("[bold green]✅ Previous Application Feature Engineering Complete")

    return previous_application_features

# ==========================================================================================================================================================

# ==============================================================
#   FEATURE ENGINNERING OF INSTALLMENTS_PAYMENTS DATASETS       |===========================================================================================
# ==============================================================

def aggregate_installments(installments_df):
    console.print("[bold #FF7800]⚡ Installments Feature Engineering Started")

    # -------------------------------
    #  Validations
    # -------------------------------
    assert "SK_ID_CURR" in installments_df.columns

    console.print("[bold #0080FF].....Before Aggregation.....")
    print_dataframe_info(installments_df, "Installments Payments")

    # ------------------------------------------
    #   Installments Payments Aggregation
    # ------------------------------------------
    installments_df = installments_df.copy()

    # Late Payment Criteria             
    installments_df["payment_delay"] = (installments_df["DAYS_ENTRY_PAYMENT"] - installments_df["DAYS_INSTALMENT"])

    # Underpayment Criteria
    installments_df["payment_ratio"] = (installments_df["AMT_PAYMENT"] / installments_df["AMT_INSTALMENT"])

    installments_features = (
        installments_df
            .groupby("SK_ID_CURR", as_index=False)
            .agg(
                  # Payment Amount Features
                  inst_total_instalment = ("AMT_INSTALMENT", "sum"),
                  inst_avg_instalment = ("AMT_INSTALMENT", "mean"),
                  inst_max_instalment=("AMT_INSTALMENT", "max"),
  
                  inst_total_payment = ("AMT_PAYMENT", "sum"),
                  inst_avg_payment = ("AMT_PAYMENT", "mean"),
                  inst_max_payment=("AMT_PAYMENT", "max"),
                  inst_min_payment=("AMT_PAYMENT", "min"),

                  # Late Payment Features
                  inst_avg_delay = ("payment_delay", "mean"),
                  inst_max_delay = ("payment_delay", "max"),
                  inst_late_payment_count = ("payment_delay", lambda x: (x > 0).sum()),

                  # Underpayment Features
                  inst_avg_payment_ratio = ("payment_ratio", "mean"),
                  inst_min_payment_ratio = ("payment_ratio", "min"),
                  inst_underpaid_count = ("payment_ratio", lambda x: (x < 1).sum()),

                  # Installment History
                  inst_count = ("NUM_INSTALMENT_NUMBER", "count"),
                  inst_max_installment_number = ("NUM_INSTALMENT_NUMBER", "max")

                )
        )
    
    assert installments_features["SK_ID_CURR"].is_unique, \
        "Duplicate customers found after installments payments aggregation!"
    
    console.print("[bold #0080FF].....After Aggregation.....")
    print_dataframe_info(installments_features, "Installments Payments")

    console.print("[bold green]✅ Installments Payments Feature Engineering Complete")

    return installments_features

# ==========================================================================================================================================================

# ==============================================================
#   FEATURE ENGINNERING OF INSTALLMENTS_PAYMENTS DATASETS       |===========================================================================================
# ==============================================================

def aggregate_credit_card_bal(credit_card_bal_df):
    console.print("[bold #FF7800]⚡ Credit Card Balance Feature Engineering Started")

    # -------------------------------
    #  Validations
    # -------------------------------
    assert "SK_ID_CURR" in credit_card_bal_df.columns

    console.print("[bold #0080FF].....Before Aggregation.....")
    print_dataframe_info(credit_card_bal_df, "Credit Card Balance")

    credit_card_bal_df = credit_card_bal_df.copy()

    credit_card_bal_df["credit_utilization"] = (
        credit_card_bal_df["AMT_BALANCE"] /credit_card_bal_df["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
        ).clip(lower=0, upper=5)

    # ------------------------------------------
    #   Credit Card Balance Aggregation
    # ------------------------------------------
    credit_card_features = (
        credit_card_bal_df
            .groupby("SK_ID_CURR", as_index=False)
            .agg(
                  # Number of unique credit card accounts.
                  cc_card_count=("SK_ID_PREV", "nunique"),

                  # Shows how much money the customer usually owes.
                  cc_total_balance=("AMT_BALANCE", "sum"),
                  cc_avg_balance=("AMT_BALANCE", "mean"),
                  cc_max_balance=("AMT_BALANCE", "max"),

                  # Credit Limit
                  cc_avg_credit_limit=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
                  cc_max_credit_limit=("AMT_CREDIT_LIMIT_ACTUAL", "max"),

                  # Credit Utilization
                  cc_avg_utilization=("credit_utilization", "mean"),
                  cc_max_utilization=("credit_utilization", "max"),

                  # ATM Withdrawals
                  cc_total_atm_drawings=("AMT_DRAWINGS_ATM_CURRENT", "sum"),
                  cc_avg_atm_drawings=("AMT_DRAWINGS_ATM_CURRENT", "mean"),

                  # Total Drawings
                  cc_total_drawings=("AMT_DRAWINGS_CURRENT", "sum"),
                  cc_avg_drawings=("AMT_DRAWINGS_CURRENT", "mean"),

                  # Payments
                  cc_total_payment=("AMT_PAYMENT_TOTAL_CURRENT", "sum"),
                  cc_avg_payment=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
                  cc_max_payment=("AMT_PAYMENT_TOTAL_CURRENT", "max"),

                  # Receivable Amount
                  cc_avg_receivable=("AMT_TOTAL_RECEIVABLE", "mean"),
                  cc_max_receivable=("AMT_TOTAL_RECEIVABLE", "max"),

                  # Installments
                  cc_avg_installment=("CNT_INSTALMENT_MATURE_CUM", "mean"),
                  cc_max_installment=("CNT_INSTALMENT_MATURE_CUM", "max"),

                  # DPD (Days Past Due)
                  cc_avg_dpd=("SK_DPD", "mean"),
                  cc_max_dpd=("SK_DPD", "max"),

                  # Defaulters
                  cc_late_payment_count=("SK_DPD", lambda x: (x > 0).sum()),

                  # Serious Delinquency
                  cc_serious_dpd_count=("SK_DPD_DEF", lambda x: (x > 0).sum()),

                  # History length
                  cc_oldest_record=("MONTHS_BALANCE", "min"),
                  cc_latest_record=("MONTHS_BALANCE", "max"),

                ))
    
    assert credit_card_features["SK_ID_CURR"].is_unique, \
        "Duplicate customers found after Credit Card Balance aggregation!"
    
    console.print("[bold #0080FF].....After Aggregation.....")
    print_dataframe_info(credit_card_features, "Credit Card Balance")

    console.print("[bold green]✅ Credit Card Balance Feature Engineering Complete")

    return credit_card_features

# ==========================================================================================================================================================

# ==============================================================
#   FEATURE ENGINNERING OF INSTALLMENTS_PAYMENTS DATASETS       |===========================================================================================
# ==============================================================

def aggregate_POS_cash_bal(pos_cash_bal_df):
    console.print("[bold #FF7800]⚡ POS Cash Balance Feature Engineering Started")

    # -------------------------------
    #  Validations
    # -------------------------------
    assert "SK_ID_CURR" in pos_cash_bal_df.columns

    console.print("[bold #0080FF].....Before Aggregation.....")
    print_dataframe_info(pos_cash_bal_df, "POS Cash Balance")

    # ------------------------------------------
    #   Credit Card Balance Aggregation
    # ------------------------------------------
    pos_features = (
        pos_cash_bal_df
            .groupby("SK_ID_CURR", as_index=False)
            .agg(
                  # Number of POS/Cash loans
                  pos_loan_count=("SK_ID_PREV", "nunique"),

                  # History Length
                  pos_oldest_record=("MONTHS_BALANCE", "min"),
                  pos_latest_record=("MONTHS_BALANCE", "max"),

                  # Installments
                  pos_avg_instalment=("CNT_INSTALMENT", "mean"),
                  pos_max_instalment=("CNT_INSTALMENT", "max"),

                  pos_avg_future_instalment=("CNT_INSTALMENT_FUTURE", "mean"),
                  pos_max_future_instalment=("CNT_INSTALMENT_FUTURE", "max"),

                  # DPD (Days Past Due)
                  pos_avg_dpd=("SK_DPD", "mean"),
                  pos_max_dpd=("SK_DPD", "max"),

                  # DPD Def
                  pos_avg_dpd_def=("SK_DPD_DEF", "mean"),
                  pos_max_dpd_def=("SK_DPD_DEF", "max"),

                  # Late Payments
                  pos_late_payment_count=("SK_DPD", lambda x: (x > 0).sum()),

                  # Serious Delinquency
                  pos_serious_dpd_count=("SK_DPD_DEF", lambda x: (x > 0).sum()),

                  # Contract Status
                  pos_active_contracts=("NAME_CONTRACT_STATUS", count_value("Active")),

                  pos_completed_contracts=("NAME_CONTRACT_STATUS", count_value("Completed")),

                  pos_signed_contracts=("NAME_CONTRACT_STATUS", count_value("Signed")),

                  pos_demand_contracts=("NAME_CONTRACT_STATUS", count_value("Demand")),
                ))
    
    assert pos_features["SK_ID_CURR"].is_unique, \
        "Duplicate customers found after POS Cash Balance aggregation!"
    
    console.print("[bold #0080FF].....After Aggregation.....")
    print_dataframe_info(pos_features, "POS Cash Balance")

    console.print("[bold green]✅ POS Cash Balance Feature Engineering Complete")

    return pos_features


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

    task = progress.add_task("[#FF7800]Feature Engineering Pipeline", total=5)

    # feature enginerring and merging of Bureau and Bureau balance
    bureau_agg = aggregate_bureau(data["bureau"], data["bureau_balance"])
    progress.advance(task)

    # feature engineering of Previous application
    prev_app_agg = aggregate_previous_applications(data["previous_applications"])
    progress.advance(task)

    # feature engineering of installments payments
    installments_agg = aggregate_installments(data["installments_payments"])
    progress.advance(task)

    # feature engineering of credit card balance
    credit_card_bal_agg = aggregate_credit_card_bal(data["credit_card_balance"])
    progress.advance(task)

    # feature engineering of POS cash balance
    pos_cash_bal_agg = aggregate_POS_cash_bal(data["pos_cash_balance"])
    progress.advance(task)

console.rule("\n[bold green]🎉 Feature Engineering Completed Successfully![/bold green]")

def merge_feature_datasets(application_df, feature_datasets):
    console.print("[bold #FF7800]⚡ Merging Feature Datasets[/bold #FF7800]")

    merged_df = application_df.copy()

    console.print(f"\n[bold #0080FF]Starting Shape:[/bold #0080FF] {merged_df.shape}")

    for name, feature_df in feature_datasets:

        before_rows = merged_df.shape[0]

        merged_df = merged_df.merge(
            feature_df,
            on="SK_ID_CURR",
            how="left"
        )

        assert merged_df.shape[0] == before_rows, \
            f"Row count changed after merging {name}"

        assert merged_df["SK_ID_CURR"].is_unique, \
            f"Duplicate SK_ID_CURR after merging {name}"

        console.print(
            f"[green]✔ {name:<25}[/green] "
            f"Shape: {merged_df.shape}"
        )

    console.print("\n[bold green]✅ All Feature Datasets Merged Successfully[/bold green]")
    print_dataframe_info(merged_df, "Final Dataset")

    return merged_df

feature_datasets = [
    ("Bureau", bureau_agg),
    ("Previous Applications", prev_app_agg),
    ("Installments", installments_agg),
    ("Credit Card", credit_card_bal_agg),
    ("POS Cash", pos_cash_bal_agg),
]

final_train = merge_feature_datasets(data["application_train"],feature_datasets)

final_test = merge_feature_datasets(data["application_test"],feature_datasets)



output_dir = Path("Processed Datasets/")
output_dir.mkdir(exist_ok=True)

final_train.to_csv(output_dir / "Processed Datasets/final_train.csv", index=False)
final_test.to_csv(output_dir / "Processed Datasets/final_test.csv", index=False)

console.print("[bold green]💾 Final datasets saved successfully![/bold green]")