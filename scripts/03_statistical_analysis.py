import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import stats_analysis as stat

def main():
    try:
        df_industry = pd.read_csv(os.path.join('..', 'data', 'processed', 'industry_data.csv'))
        print(f"Loaded industry data with {len(df_industry)} records.")
    except FileNotFoundError:
        print("Industry data not found.")
        return

    print("\n--- Correlation Analysis ---")
    if not df_industry.empty:
        heatmap_path = os.path.join('..', 'visualizations', 'industry', 'correlation_heatmap.png')
        stat.plot_correlation_heatmap(df_industry, "Industry Financial Metrics Correlation", save_path=heatmap_path)
        print(f"Correlation heatmap generated at {heatmap_path}")

    print("\n--- Hypothesis Testing ---")
    print("H₀: There is no statistically significant linear relationship between Operating Profit Margin (OPM) and ROE.")
    print("H₁: There is a statistically significant linear relationship between Operating Profit Margin (OPM) and ROE.")
    
    col_x = 'opm_5year'
    col_y = 'average_return_on_equity_5years'
    
    if not df_industry.empty and col_x in df_industry.columns and col_y in df_industry.columns:
        corr, p_val = stat.calculate_pearson_correlation(df_industry, col_x, col_y)
        print(f"\nPearson Correlation: {corr:.4f}")
        print(stat.interpret_p_value(p_val))
    else:
        print(f"\nRequired columns '{col_x}' or '{col_y}' missing from dataset.")

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), '03_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("Executing 03_statistical_analysis.py...")
        main()
        sys.stdout = sys.__stdout__
    print(f"Execution complete. Output written to {output_file}")
