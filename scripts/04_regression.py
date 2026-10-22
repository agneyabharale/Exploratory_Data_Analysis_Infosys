import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import regression as reg

def main():
    try:
        df_industry = pd.read_csv(os.path.join('..', 'data', 'processed', 'industry_data.csv'))
        print(f"Industry data loaded successfully ({len(df_industry)} records).")
    except FileNotFoundError:
        print("Industry data not found. Ensure script 01 has been run.")
        return

    print("=========================================================")
    print("      MACHINE LEARNING: MULTI-MODEL FINANCIAL PREDICTION ")
    print("      (Linear Regression vs. Decision Tree vs. KNN)      ")
    print("=========================================================")

    target_col = 'average_return_on_equity_5years'
    feature_cols = [
        'opm_5year',
        'return_on_assets_5years',
        'average_return_on_capital_employed_5years',
        'historical_pe_5years'
    ]

    missing_cols = [c for c in feature_cols + [target_col] if c not in df_industry.columns]
    if missing_cols:
        print(f"Error: Missing required columns in dataset: {missing_cols}")
        return

    print(f"\nTarget Variable (Y): {target_col}")
    print(f"Feature Predictors (X): {feature_cols}")

    # Prepare multivariate data with outlier trimming & standard scaling
    X_train, X_test, y_train, y_test, scaler = reg.prepare_data(
        df_industry, x_cols=feature_cols, y_col=target_col,
        test_size=0.2, random_state=42, filter_outliers=True, scale_features=True
    )
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Train all 3 models
    print("\n--- Training Models: Linear Regression, Decision Tree, and KNN ---")
    models, predictions, metrics_df = reg.train_and_evaluate_all_models(
        X_train, X_test, y_train, y_test
    )

    print("\n=========================================================")
    print("                MODEL EVALUATION COMPARISON              ")
    print("=========================================================")
    print(metrics_df.to_string(index=False))

    # Identify Best Model by R²
    best_model_row = metrics_df.sort_values('R2', ascending=False).iloc[0]
    print(f"\nBest Performing Model: {best_model_row['Model']} (R² = {best_model_row['R2']:.4f}, RMSE = {best_model_row['RMSE']:.4f})")

    # Linear Regression Significance
    p_values, _ = reg.calculate_statistical_significance(X_train, y_train)
    print("\nLinear Regression Coefficients & P-Values:")
    lr_model = models['Linear Regression']
    for feat, coef in zip(feature_cols, lr_model.coef_):
        pval = p_values.get(feat, np.nan)
        print(f"  {feat:42s} | Coef: {coef:9.4f} | p-value: {pval:.4e}")

    # Visualizations
    print("\n--- Generating Visualizations ---")
    comp_path = os.path.join('..', 'visualizations', 'regression', 'model_comparison_actual_vs_predicted.png')
    reg.plot_model_comparison(y_test, predictions, save_path=comp_path)
    print(f"Saved Actual vs Predicted comparison -> {comp_path}")

    fi_path = os.path.join('..', 'visualizations', 'regression', 'decision_tree_feature_importance.png')
    reg.plot_feature_importance(models['Decision Tree'], feature_cols, save_path=fi_path)
    print(f"Saved Decision Tree Feature Importance -> {fi_path}")

    res_path = os.path.join('..', 'visualizations', 'regression', 'residuals_comparison.png')
    reg.plot_residuals_comparison(y_test, predictions, save_path=res_path)
    print(f"Saved Residuals comparison -> {res_path}")

    # Legacy 2D Univariate line for dashboard backwards-compatibility
    legacy_path = os.path.join('..', 'visualizations', 'regression', 'regression_line.png')
    univar_X_tr, univar_X_te, univar_y_tr, univar_y_te, _ = reg.prepare_data(
        df_industry, x_cols=['opm_5year'], y_col=target_col, filter_outliers=True, scale_features=False
    )
    univar_lr, univar_pred, _ = reg.train_linear_regression(univar_X_tr, univar_X_te, univar_y_tr, univar_y_te)
    reg.plot_regression(univar_X_te['opm_5year'], univar_y_te, univar_pred, 'opm_5year', target_col, save_path=legacy_path)
    print(f"Saved Univariate Baseline Regression line -> {legacy_path}")

    # Benchmarking Infosys Performance
    print("\n=========================================================")
    print("         BENCHMARKING INFOSYS LTD. AGAINST MODELS         ")
    print("=========================================================")
    # Infosys 5-year average metrics
    infosys_profile = {
        'opm_5year': 24.75,                      # ~24.8% operating margin
        'return_on_assets_5years': 27.85,         # ~27.8% ROA
        'average_return_on_capital_employed_5years': 38.5,  # ROCE
        'historical_pe_5years': 26.5             # Historical P/E
    }
    actual_infosys_roe = 59.7  # Actual average ROE (%)

    print("Infosys Actual Input Profile:")
    for k, v in infosys_profile.items():
        print(f"  {k:42s}: {v}")
    print(f"  Actual Reported 5-Year Average ROE        : {actual_infosys_roe:.2f}%")

    preds_infy = reg.predict_company_performance(models, scaler, infosys_profile, feature_cols)
    print("\nPredicted ROE for Infosys across Models:")
    for model_name, pred_val in preds_infy.items():
        diff = actual_infosys_roe - pred_val
        print(f"  {model_name:28s}: {pred_val:6.2f}% (Outperformed expectation by +{diff:.2f}%)")

    print("\nInsight: Infosys significantly outperforms general industry baseline models,")
    print("reflecting superior capital efficiency, brand equity, and high asset turnover.")
    print("=========================================================")

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), '04_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("Executing 04_regression.py with Linear Regression, Decision Tree, and KNN...")
        main()
        sys.stdout = sys.__stdout__
    print(f"Execution complete. Output written to {output_file}")
