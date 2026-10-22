import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_trend(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, save_path: str = None):
    """Plot a line chart for trend analysis over time."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, marker='o', linewidth=2, color='#2563eb')
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_distribution(df: pd.DataFrame, col: str, title: str, save_path: str = None):
    """Plot a histogram with KDE for distribution analysis."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col].dropna(), kde=True, color='#0284c7', bins=15)
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_comparison_bar(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, title: str, save_path: str = None):
    """Plot a grouped bar chart for comparisons."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# BIVARIATE ANALYSIS
# ==========================================

def plot_bivariate_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, 
                           xlabel: str = None, ylabel: str = None, hue_col: str = None, 
                           add_trendline: bool = True, save_path: str = None):
    """Bivariate scatter plot with optional OLS trendline and categorical hue."""
    plt.figure(figsize=(10, 6))
    clean_df = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
    
    if add_trendline and clean_df[x_col].nunique() > 1:
        sns.regplot(data=clean_df, x=x_col, y=y_col, scatter_kws={'alpha': 0.5, 'color': '#3b82f6'},
                    line_kws={'color': '#dc2626', 'linewidth': 2})
    else:
        sns.scatterplot(data=clean_df, x=x_col, y=y_col, hue=hue_col, alpha=0.7, palette='tab10')

    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel(xlabel if xlabel else x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(ylabel if ylabel else y_col.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_bivariate_box(df: pd.DataFrame, cat_col: str, num_col: str, title: str,
                       top_n: int = 8, save_path: str = None):
    """Bivariate boxplot showing distribution of a numeric metric across top categories."""
    clean_df = df[[cat_col, num_col]].dropna()
    top_cats = clean_df[cat_col].value_counts().nlargest(top_n).index
    filtered_df = clean_df[clean_df[cat_col].isin(top_cats)]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x=cat_col, y=num_col, palette='Set2')
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel(cat_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(num_col.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# MULTIVARIATE ANALYSIS
# ==========================================

def plot_multivariate_pairplot(df: pd.DataFrame, cols: list, title: str = "Multivariate Financial Relationships",
                               save_path: str = None):
    """Multivariate pairplot analyzing pairwise distributions and joint relationships."""
    clean_df = df[cols].dropna()
    g = sns.pairplot(clean_df, diag_kind='kde', plot_kws={'alpha': 0.6, 'color': '#2563eb'},
                     diag_kws={'color': '#0284c7', 'fill': True})
    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle(title, fontsize=14, fontweight='bold')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        g.savefig(save_path, dpi=300)
    plt.close()

def plot_multivariate_bubble(df: pd.DataFrame, x_col: str, y_col: str, size_col: str, 
                             hue_col: str = None, title: str = "Multivariate Bubble Analysis",
                             save_path: str = None):
    """Multivariate bubble chart sizing data points by a 3rd dimension (e.g. Market Cap)."""
    clean_df = df[[x_col, y_col, size_col] + ([hue_col] if hue_col else [])].dropna()
    # Normalize size for display
    sizes = np.clip(clean_df[size_col], a_min=1, a_max=None)
    norm_sizes = 20 + 300 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-5)

    plt.figure(figsize=(11, 7))
    scatter = plt.scatter(clean_df[x_col], clean_df[y_col], s=norm_sizes, 
                          alpha=0.6, c='#6366f1', edgecolors='black', linewidth=0.5)
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
