import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import eda

def main():
    try:
        df_income = pd.read_csv(os.path.join('..', 'data', 'processed', 'income_statement.csv'))
        df_balance = pd.read_csv(os.path.join('..', 'data', 'processed', 'balance_sheet.csv'))
        df_cashflow = pd.read_csv(os.path.join('..', 'data', 'processed', 'cash_flow.csv'))
        df_industry = pd.read_csv(os.path.join('..', 'data', 'processed', 'industry_data.csv'))
    except FileNotFoundError:
        print("Processed files not found. Ensure script 01 was run successfully.")
        return

    print("--- Descriptive Statistics ---")
    if not df_income.empty:
        print("Income Statement Summary:")
        print(df_income.describe().to_string())
    if not df_balance.empty:
        print("\nBalance Sheet Summary:")
        print(df_balance.describe().to_string())

    print("\n--- Generating Trend Plots ---")
    if not df_income.empty and 'revenue' in df_income.columns and 'year' in df_income.columns:
        eda.plot_trend(df_income, 'year', 'revenue', 'Infosys Revenue Trend', 'Financial Year', 'Revenue', 
                       save_path=os.path.join('..', 'visualizations', 'revenue', 'revenue_trend.png'))
        print("Generated revenue trend plot.")

    if not df_income.empty and 'net_profit' in df_income.columns and 'year' in df_income.columns:
        eda.plot_trend(df_income, 'year', 'net_profit', 'Infosys PAT Trend', 'Financial Year', 'Profit After Tax', 
                       save_path=os.path.join('..', 'visualizations', 'profitability', 'pat_trend.png'))
        print("Generated PAT trend plot.")

    print("\n--- Generating Distribution Plots ---")
    if not df_cashflow.empty and 'operating_cash_flow' in df_cashflow.columns:
        eda.plot_distribution(df_cashflow, 'operating_cash_flow', 'Distribution of Operating Cash Flow', 
                              save_path=os.path.join('..', 'visualizations', 'cash_flow', 'ocf_dist.png'))
        print("Generated OCF distribution plot.")

    print("\n--- Generating Industry Benchmarking Plots ---")
    if not df_industry.empty and 'roe' in df_industry.columns and 'company' in df_industry.columns:
        print("Benchmarking ROE against Industry. (Chart suppressed in static script run, implement comparison logic here)")
    
    print("\nEDA visualizations generated and saved.")

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), '02_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("Executing 02_eda.py...")
        main()
        sys.stdout = sys.__stdout__
    print(f"Execution complete. Output written to {output_file}")
