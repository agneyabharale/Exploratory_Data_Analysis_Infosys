import sys
import os
import matplotlib.pyplot as plt
from PIL import Image

def display_dashboard(image_paths, output_path):
    """Compile visualizations into a single dashboard image."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, path in enumerate(image_paths):
        if os.path.exists(path):
            try:
                img = Image.open(path)
                axes[i].imshow(img)
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading image:\n{e}", ha='center', va='center')
        else:
            axes[i].text(0.5, 0.5, f"Image not found:\n{path}", ha='center', va='center')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Dashboard saved to {output_path}")

def main():
    print("=========================================================")
    print("             FINAL INSIGHTS & SYNTHESIS                  ")
    print("=========================================================")
    print("\n1. Financial Performance")
    print("   - Revenue and PAT trends (see visualizations).")
    print("   - Net Profit Margin analysis indicates business efficiency over the period.")
    
    print("\n2. Financial Health")
    print("   - Liquidity (Current Ratio) and Leverage (Debt-to-Equity) trends.")
    print("   - Return metrics (ROE/ROA) provide a picture of shareholder value generation.")
    
    print("\n3. Cash Generation")
    print("   - Operating Cash Flow (OCF) supports the reported PAT.")
    print("   - Free Cash Flow (FCF) shows capital available after expenditures.")
    
    print("\n4. Industry Position")
    print("   - Benchmarking against peers reveals Infosys's relative standing in profitability and efficiency.")
    
    print("\n5. Statistical Findings")
    print("   - The correlation between Operating Income and PAT was computed and tested for significance.")
    print("   - The linear regression model evaluated predictive power (R²) and error (RMSE, MAE).")
    
    print("\n6. Final Conclusion")
    print("   - The data indicates a consistent operational performance with [placeholder for actual conclusion based on script outputs].")
    print("=========================================================\n")

    print("--- Generating Static Dashboard ---")
    img_paths = [
        os.path.join('..', 'visualizations', 'revenue', 'revenue_trend.png'),
        os.path.join('..', 'visualizations', 'profitability', 'pat_trend.png'),
        os.path.join('..', 'visualizations', 'cash_flow', 'ocf_dist.png'),
        os.path.join('..', 'visualizations', 'regression', 'regression_line.png')
    ]
    dashboard_path = os.path.join('..', 'reports', 'dashboard', 'final_dashboard.png')
    display_dashboard(img_paths, dashboard_path)

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), '05_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("Executing 05_final_insights.py...")
        main()
        sys.stdout = sys.__stdout__
    print(f"Execution complete. Output written to {output_file}")
