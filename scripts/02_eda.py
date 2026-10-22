import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import eda

def main():
    try:
        df_income = pd.read_csv(os.path.join('..', 'data', 'processed', 'income_statement.csv'))
        df_balance = pd.read_csv(os.path.join('..', 'data', 'processed', 'balance_sheet.csv'))
        df_cashflow = pd.read_csv(os.path.join('..', 'data', 'processed', 'cash_flow.csv'))
        df_industry = pd.read_csv(os.path.join('..', 'data', 'processed', 'industry_data.csv'))
    except FileNotFoundError:
        print("Processed files not found. Ensure script 01 was run successfully.")
        return

    print("=========================================================")
    print("                1. DESCRIPTIVE STATISTICS                ")
    print("=========================================================")
    if not df_income.empty:
        print("\nIncome Statement Summary:")
        print(df_income.describe().to_string())
    if not df_balance.empty:
        print("\nBalance Sheet Summary:")
        print(df_balance.describe().to_string())

    print("\n=========================================================")
    print("                2. UNIVARIATE ANALYSIS                   ")
    print("=========================================================")
    # Univariate 1: Revenue Trend
    if not df_income.empty and 'revenue' in df_income.columns and 'year' in df_income.columns:
        revenue_path = os.path.join('..', 'visualizations', 'revenue', 'revenue_trend.png')
        eda.plot_trend(df_income, 'year', 'revenue', 'Infosys Revenue Trend (2022-2025)', 'Financial Year', 'Revenue (₹ Cr)', 
                       save_path=revenue_path)
        print(f"Generated revenue trend plot -> {revenue_path}")

    # Univariate 2: PAT Trend
    if not df_income.empty and 'net_profit' in df_income.columns and 'year' in df_income.columns:
        pat_path = os.path.join('..', 'visualizations', 'profitability', 'pat_trend.png')
        eda.plot_trend(df_income, 'year', 'net_profit', 'Infosys PAT Trend (2022-2025)', 'Financial Year', 'Profit After Tax (₹ Cr)', 
                       save_path=pat_path)
        print(f"Generated PAT trend plot -> {pat_path}")

    # Univariate 3: Operating Cash Flow Distribution
    if not df_cashflow.empty and 'operating_cash_flow' in df_cashflow.columns:
        ocf_path = os.path.join('..', 'visualizations', 'cash_flow', 'ocf_dist.png')
        eda.plot_distribution(df_cashflow, 'operating_cash_flow', 'Distribution of Operating Cash Flow', 
                              save_path=ocf_path)
        print(f"Generated OCF distribution plot -> {ocf_path}")

    print("\n=========================================================")
    print("                3. BIVARIATE ANALYSIS                    ")
    print("=========================================================")
    # Bivariate 1: Revenue vs Net Profit (Company Level)
    if not df_income.empty and 'revenue' in df_income.columns and 'net_profit' in df_income.columns:
        biv1_path = os.path.join('..', 'visualizations', 'bivariate', 'revenue_vs_net_profit.png')
        eda.plot_bivariate_scatter(df_income, 'revenue', 'net_profit', 
                                   title="Infosys: Revenue vs. Net Profit (Bivariate)",
                                   xlabel="Total Revenue (₹ Cr)", ylabel="Profit After Tax (₹ Cr)",
                                   add_trendline=True, save_path=biv1_path)
        print(f"Generated Bivariate Plot 1: Revenue vs Net Profit -> {biv1_path}")

    # Bivariate 2: Operating Cash Flow vs Net Profit (Earnings Quality)
    if not df_income.empty and not df_cashflow.empty and 'net_profit' in df_income.columns and 'operating_cash_flow' in df_cashflow.columns:
        df_merged_infy = pd.merge(df_income[['year', 'net_profit']], df_cashflow[['year', 'operating_cash_flow']], on='year')
        biv2_path = os.path.join('..', 'visualizations', 'bivariate', 'ocf_vs_net_profit.png')
        eda.plot_bivariate_scatter(df_merged_infy, 'operating_cash_flow', 'net_profit',
                                   title="Infosys: Operating Cash Flow vs. Net Profit (Earnings Quality)",
                                   xlabel="Operating Cash Flow (₹ Cr)", ylabel="Net Profit (₹ Cr)",
                                   add_trendline=True, save_path=biv2_path)
        print(f"Generated Bivariate Plot 2: OCF vs Net Profit -> {biv2_path}")

    # Bivariate 3: Operating Profit Margin vs ROE (Industry Level)
    if not df_industry.empty and 'opm_5year' in df_industry.columns and 'average_return_on_equity_5years' in df_industry.columns:
        # Filter extreme ratios for clean visualization
        clean_industry = df_industry[(df_industry['opm_5year'].between(-50, 80)) & 
                                     (df_industry['average_return_on_equity_5years'].between(-50, 100))].copy()
        biv3_path = os.path.join('..', 'visualizations', 'bivariate', 'opm_vs_roe_industry.png')
        eda.plot_bivariate_scatter(clean_industry, 'opm_5year', 'average_return_on_equity_5years',
                                   title="Industry: 5-Yr OPM vs. 5-Yr Average ROE (Bivariate)",
                                   xlabel="5-Year Operating Profit Margin (%)", 
                                   ylabel="5-Year Average ROE (%)",
                                   add_trendline=True, save_path=biv3_path)
        print(f"Generated Bivariate Plot 3: Industry OPM vs ROE -> {biv3_path}")

    # Bivariate 4: ROE across Top Industries (Categorical Bivariate)
    if not df_industry.empty and 'industry' in df_industry.columns and 'average_return_on_equity_5years' in df_industry.columns:
        clean_ind = df_industry[(df_industry['average_return_on_equity_5years'].between(-30, 80))].copy()
        biv4_path = os.path.join('..', 'visualizations', 'bivariate', 'roe_by_top_industries.png')
        eda.plot_bivariate_box(clean_ind, 'industry', 'average_return_on_equity_5years',
                               title="ROE Distribution across Top 8 Industries (Sector Comparison)",
                               top_n=8, save_path=biv4_path)
        print(f"Generated Bivariate Plot 4: ROE across Top Sectors -> {biv4_path}")

    print("\n=========================================================")
    print("                4. MULTIVARIATE ANALYSIS                 ")
    print("=========================================================")
    # Multivariate 1: Pairplot across 4 core financial ratios
    multi_cols = ['opm_5year', 'return_on_assets_5years', 'average_return_on_capital_employed_5years', 'average_return_on_equity_5years']
    if not df_industry.empty and all(col in df_industry.columns for col in multi_cols):
        clean_multi = df_industry[multi_cols].dropna()
        clean_multi = clean_multi[(clean_multi['opm_5year'].between(-30, 70)) &
                                  (clean_multi['return_on_assets_5years'].between(-20, 50)) &
                                  (clean_multi['average_return_on_capital_employed_5years'].between(-20, 60)) &
                                  (clean_multi['average_return_on_equity_5years'].between(-20, 60))].sample(min(800, len(clean_multi)), random_state=42)
        multi1_path = os.path.join('..', 'visualizations', 'multivariate', 'financial_metrics_pairplot.png')
        eda.plot_multivariate_pairplot(clean_multi, multi_cols, 
                                       title="Multivariate Pairwise Relationships: OPM, ROA, ROCE, ROE",
                                       save_path=multi1_path)
        print(f"Generated Multivariate Plot 1: Financial Metrics Pairplot -> {multi1_path}")

    # Multivariate 2: Bubble Chart (OPM vs ROE, Sized by Market Cap)
    if not df_industry.empty and all(c in df_industry.columns for c in ['opm_5year', 'average_return_on_equity_5years', 'market_capitalization']):
        bubble_df = df_industry[(df_industry['opm_5year'].between(-20, 60)) &
                                (df_industry['average_return_on_equity_5years'].between(-10, 60)) &
                                (df_industry['market_capitalization'] > 0)].copy()
        multi2_path = os.path.join('..', 'visualizations', 'multivariate', 'opm_vs_roe_marketcap_bubble.png')
        eda.plot_multivariate_bubble(bubble_df, 'opm_5year', 'average_return_on_equity_5years', 'market_capitalization',
                                     title="Multivariate: OPM vs. ROE (Bubble Size = Market Capitalization)",
                                     save_path=multi2_path)
        print(f"Generated Multivariate Plot 2: OPM vs ROE Market Cap Bubble -> {multi2_path}")

    print("\nEDA execution complete. All univariate, bivariate, and multivariate visualizations saved.")

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), '02_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("Executing 02_eda.py with Univariate, Bivariate, and Multivariate analysis...")
        main()
        sys.stdout = sys.__stdout__
    print(f"Execution complete. Output written to {output_file}")
