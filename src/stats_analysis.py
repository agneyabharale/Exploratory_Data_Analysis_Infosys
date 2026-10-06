import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df: pd.DataFrame, title: str, save_path: str = None):
    """Plot a correlation heatmap for numeric columns."""
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def calculate_pearson_correlation(df: pd.DataFrame, col1: str, col2: str):
    """Calculate Pearson correlation and return coefficient and p-value."""
    clean_df = df[[col1, col2]].dropna()
    if clean_df.empty or len(clean_df) < 2:
        return None, None
    corr, p_value = stats.pearsonr(clean_df[col1], clean_df[col2])
    return corr, p_value

def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """Interpret the statistical significance of a p-value."""
    if p_value is None:
        return "Not enough data to compute p-value."
    if p_value < alpha:
        return f"Statistically significant (p = {p_value:.4f} < {alpha}). Reject H0."
    else:
        return f"Not statistically significant (p = {p_value:.4f} >= {alpha}). Fail to reject H0."
