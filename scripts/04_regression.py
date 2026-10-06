import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import regression as reg

def main():
    try:
        df_industry = pd.read_csv(os.path.join('..', 'data', 'processed', 'industry_data.csv'))
        print("Industry data loaded.")
    except FileNotFoundError:
        print("Industry data not found.")
        return

    print("\n--- Data Prep & Train/Test Split ---")
    col_x = 'opm_5year'
    col_y = 'average_return_on_equity_5years'
    
    if not df_industry.empty and col_x in df_industry.columns and col_y in df_industry.columns:
        X_train, X_test, y_train, y_test = reg.prepare_data(df_industry, col_x, col_y)
        print(f"Training set size: {len(X_train)} | Testing set size: {len(X_test)}")
    else:
        print("Cannot proceed. Required columns missing.")
        return

    print("\n--- Model Training & Metrics ---")
    model, y_pred, metrics = reg.train_and_evaluate(X_train, X_test, y_train, y_test)
    p_value, summary = reg.calculate_statistical_significance(X_train, y_train)
    
    print("Regression Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  P-Value ({col_x}): {p_value}")

    print("\n--- Visualizations ---")
    reg_path = os.path.join('..', 'visualizations', 'regression', 'regression_line.png')
    res_path = os.path.join('..', 'visualizations', 'regression', 'residuals.png')
    
    reg.plot_regression(X_test, y_test, y_pred, col_x, col_y, save_path=reg_path)
    print(f"Saved regression plot to {reg_path}")
    
    reg.plot_residuals(y_test, y_pred, save_path=res_path)
    print(f"Saved residuals plot to {res_path}")

    print("\n--- Model Limitations ---")
    print("- Linearity Assumption: Linear regression assumes a strict linear relationship.")
    print("- Comparability: Industry/company observations may not be perfectly comparable.")
    print("- Scale & Outliers: Financial data can contain extreme scale differences.")
    print("- Causation: A strong correlation (R²) does not necessarily imply causation.")
    print("- Dataset Size: The model's predictive performance depends heavily on the quality of the industry dataset.")
    print("- Variance: Train/test split results may vary depending on the chosen random state.")

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), '04_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("Executing 04_regression.py...")
        main()
        sys.stdout = sys.__stdout__
    print(f"Execution complete. Output written to {output_file}")
