import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import data_cleaning as dc

def main():
    raw_data_path = os.path.join('..', 'data', 'raw', 'Project.xlsx')
    
    try:
        xl = pd.ExcelFile(raw_data_path)
        print(f"Available sheets: {xl.sheet_names}")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    # Load and transpose the financial statements
    df_income_raw = pd.read_excel(raw_data_path, sheet_name='Income Statement', usecols="A:E")
    df_income = df_income_raw.set_index(df_income_raw.columns[0]).T.reset_index().rename(columns={'index': 'Year'})
    
    df_balance_raw = pd.read_excel(raw_data_path, sheet_name='Balance Sheet', usecols="A:E")
    df_balance = df_balance_raw.set_index(df_balance_raw.columns[0]).T.reset_index().rename(columns={'index': 'Year'})
    
    df_cashflow_raw = pd.read_excel(raw_data_path, sheet_name='Cash Flow Statement', usecols="A:E")
    df_cashflow = df_cashflow_raw.set_index(df_cashflow_raw.columns[0]).T.reset_index().rename(columns={'index': 'Year'})
    
    df_ratios = pd.read_excel(raw_data_path, sheet_name='4. Financial Ratios')
    
    try:
        df_industry = pd.read_excel(raw_data_path, sheet_name='raw_finacial_ratio')
    except Exception:
        df_industry = pd.DataFrame()

    # Explicitly rename to standard names expected by EDA scripts
    df_income.rename(columns={'Total Revenue': 'revenue', 'Profit After Tax (PAT)': 'net_profit'}, inplace=True)
    df_cashflow.rename(columns={'Net Cash from Operating Activities': 'operating_cash_flow'}, inplace=True)

    dataframes = {
        'Income Statement': df_income,
        'Balance Sheet': df_balance,
        'Cash Flow': df_cashflow,
        'Financial Ratios': df_ratios,
        'Industry Data': df_industry
    }

    print("\n--- Standardizing Columns and Adding Identifiers ---")
    for name, df in dataframes.items():
        if not df.empty:
            if 'Company' not in df.columns:
                dc.add_identifiers(df, company='Infosys')
            dc.standardize_columns(df)

    print("\n--- Missing Values Treatment ---")
    missing_report = dc.report_missing_values(dataframes)
    print(missing_report.to_string())

    for name, df in dataframes.items():
        if not df.empty:
            df.dropna(how='all', inplace=True)

    print("\n--- Outlier Analysis ---")
    if 'revenue' in df_income.columns:
        outliers_iqr = dc.detect_outliers_iqr(df_income, 'revenue')
        print(f"Detected {len(outliers_iqr)} potential outliers in Revenue via IQR. Retaining genuine financial extremes.")

    print("\n--- Exporting Processed Data ---")
    os.makedirs(os.path.join('..', 'data', 'processed'), exist_ok=True)
    df_income.to_csv(os.path.join('..', 'data', 'processed', 'income_statement.csv'), index=False)
    df_balance.to_csv(os.path.join('..', 'data', 'processed', 'balance_sheet.csv'), index=False)
    df_cashflow.to_csv(os.path.join('..', 'data', 'processed', 'cash_flow.csv'), index=False)
    df_ratios.to_csv(os.path.join('..', 'data', 'processed', 'financial_ratios.csv'), index=False)
    if not df_industry.empty:
        df_industry.to_csv(os.path.join('..', 'data', 'processed', 'industry_data.csv'), index=False)

    print("Data cleaning complete. Files exported to 'data/processed/'.")

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), '01_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("Executing 01_data_acquisition_cleaning.py...")
        main()
        sys.stdout = sys.__stdout__
    print(f"Execution complete. Output written to {output_file}")
