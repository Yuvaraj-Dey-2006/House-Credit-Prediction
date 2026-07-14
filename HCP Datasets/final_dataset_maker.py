import pandas as pd

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

data = load_datasets()
