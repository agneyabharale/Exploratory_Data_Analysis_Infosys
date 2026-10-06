import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_trend(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, save_path: str = None):
    """Plot a line chart for trend analysis over time."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, marker='o', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_distribution(df: pd.DataFrame, col: str, title: str, save_path: str = None):
    """Plot a histogram with KDE for distribution analysis."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col].dropna(), kde=True, color='skyblue', bins=15)
    plt.title(title, fontsize=14)
    plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_comparison_bar(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, title: str, save_path: str = None):
    """Plot a grouped bar chart for comparisons."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='viridis')
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
