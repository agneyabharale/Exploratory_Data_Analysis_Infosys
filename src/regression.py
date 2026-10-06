import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm

def prepare_data(df: pd.DataFrame, x_col: str, y_col: str, test_size: float = 0.2, random_state: int = 42):
    """Clean missing values and split into train and test sets."""
    clean_df = df[[x_col, y_col]].dropna()
    if len(clean_df) < 10:
        raise ValueError("Not enough data to perform a meaningful regression.")
    X = clean_df[[x_col]]
    y = clean_df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train a scikit-learn model and calculate evaluation metrics."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Coefficient': model.coef_[0],
        'Intercept': model.intercept_
    }
    return model, y_pred, metrics

def calculate_statistical_significance(X_train, y_train):
    """Calculate p-value using statsmodels."""
    X_train_sm = sm.add_constant(X_train)
    sm_model = sm.OLS(y_train, X_train_sm).fit()
    p_value = sm_model.pvalues.iloc[1] # the p-value for the feature
    return p_value, sm_model.summary()

def plot_regression(X_test, y_test, y_pred, x_col: str, y_col: str, save_path: str = None):
    """Plot actual vs predicted regression line."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title(f'Linear Regression: {x_col} vs {y_col}', fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_residuals(y_test, y_pred, save_path: str = None):
    """Plot residuals against predicted values."""
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals vs Predicted Values', fontsize=14)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_plotly_regression(df: pd.DataFrame, x_col: str, y_col: str):
    """Interactive Plotly scatter with OLS trendline."""
    fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", 
                     title=f'Interactive Regression: {x_col} vs {y_col}')
    fig.show()
