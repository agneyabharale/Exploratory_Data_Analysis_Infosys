import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm

def prepare_data(df: pd.DataFrame, x_cols, y_col: str, test_size: float = 0.2, 
                 random_state: int = 42, filter_outliers: bool = True, scale_features: bool = False):
    """
    Clean missing values, filter extreme financial ratio anomalies,
    and split into train and test sets. Supports single or multiple features.
    """
    if isinstance(x_cols, str):
        x_cols = [x_cols]

    cols_needed = list(x_cols) + [y_col]
    clean_df = df[cols_needed].dropna()

    if filter_outliers:
        # Filter extreme target anomalies for financial ratios
        q_low = clean_df[y_col].quantile(0.01)
        q_high = clean_df[y_col].quantile(0.99)
        clean_df = clean_df[(clean_df[y_col] >= q_low) & (clean_df[y_col] <= q_high)]

        # Filter extreme predictor anomalies
        for col in x_cols:
            c_low = clean_df[col].quantile(0.01)
            c_high = clean_df[col].quantile(0.99)
            clean_df = clean_df[(clean_df[col] >= c_low) & (clean_df[col] <= c_high)]

    if len(clean_df) < 20:
        raise ValueError("Not enough data points after cleaning to perform a meaningful regression.")

    X = clean_df[x_cols]
    y = clean_df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=x_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=x_cols, index=X_test.index)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    return X_train, X_test, y_train, y_test, None

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train an OLS Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return model, y_pred, metrics

def train_decision_tree(X_train, X_test, y_train, y_test, max_depth: int = 5, random_state: int = 42):
    """Train a Decision Tree Regressor to capture non-linear interactions."""
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=15, min_samples_leaf=10, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return model, y_pred, metrics

def train_knn(X_train, X_test, y_train, y_test, n_neighbors: int = 7, weights: str = 'distance'):
    """Train a K-Nearest Neighbors (KNN) Regressor."""
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return model, y_pred, metrics

def train_and_evaluate_all_models(X_train, X_test, y_train, y_test):
    """Train Linear Regression, Decision Tree, and KNN, returning models, predictions, and summary table."""
    lr_model, lr_pred, lr_metrics = train_linear_regression(X_train, X_test, y_train, y_test)
    dt_model, dt_pred, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test)
    knn_model, knn_pred, knn_metrics = train_knn(X_train, X_test, y_train, y_test)

    models = {
        'Linear Regression': lr_model,
        'Decision Tree': dt_model,
        'K-Nearest Neighbors (KNN)': knn_model
    }

    predictions = {
        'Linear Regression': lr_pred,
        'Decision Tree': dt_pred,
        'K-Nearest Neighbors (KNN)': knn_pred
    }

    metrics_df = pd.DataFrame([
        {'Model': 'Linear Regression', **lr_metrics},
        {'Model': 'Decision Tree', **dt_metrics},
        {'Model': 'K-Nearest Neighbors (KNN)', **knn_metrics}
    ])

    return models, predictions, metrics_df

def calculate_statistical_significance(X_train, y_train):
    """Calculate p-values using statsmodels for Linear Regression."""
    X_train_sm = sm.add_constant(X_train)
    sm_model = sm.OLS(y_train, X_train_sm).fit()
    return sm_model.pvalues, sm_model.summary()

def plot_model_comparison(y_test, predictions_dict: dict, save_path: str = None):
    """Plot Actual vs. Predicted values across Linear Regression, Decision Tree, and KNN."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = list(predictions_dict.keys())
    colors = ['#2563eb', '#16a34a', '#9333ea']

    min_val = min(y_test.min(), min(pred.min() for pred in predictions_dict.values()))
    max_val = max(y_test.max(), max(pred.max() for pred in predictions_dict.values()))

    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[i]
        ax.scatter(y_test, y_pred, color=colors[i], alpha=0.4, edgecolors='none', s=25)
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
        ax.set_title(f'{name}\nActual vs. Predicted', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual ROE (%)', fontsize=11)
        if i == 0:
            ax.set_ylabel('Predicted ROE (%)', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(dt_model, feature_names: list, save_path: str = None):
    """Plot Feature Importance from the Decision Tree Regressor."""
    importances = dt_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 5))
    plt.barh(fi_df['Feature'], fi_df['Importance'], color='#059669', edgecolor='black', alpha=0.8)
    plt.title('Decision Tree: Feature Importance for ROE Prediction', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Normalized Importance Score', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_residuals_comparison(y_test, predictions_dict: dict, save_path: str = None):
    """Plot residual distributions across all models."""
    plt.figure(figsize=(12, 6))
    colors = ['#2563eb', '#16a34a', '#9333ea']

    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        residuals = y_test - y_pred
        sns.kdeplot(residuals, label=name, color=colors[i], linewidth=2)

    plt.axvline(0, color='black', linestyle='--', alpha=0.7)
    plt.title('Residuals Density Comparison (Error Distributions)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Residual Error (Actual - Predicted)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def predict_company_performance(models: dict, scaler, feature_values: dict, feature_names: list):
    """Predict performance for a specific company (e.g. Infosys) using all trained models."""
    input_df = pd.DataFrame([[feature_values[f] for f in feature_names]], columns=feature_names)
    if scaler is not None:
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
    else:
        input_scaled = input_df

    predictions = {}
    for name, model in models.items():
        pred_val = model.predict(input_scaled)[0]
        predictions[name] = pred_val
    return predictions

# Legacy wrapper for backward compatibility
def train_and_evaluate(X_train, X_test, y_train, y_test):
    return train_linear_regression(X_train, X_test, y_train, y_test)

def plot_regression(X_test, y_test, y_pred, x_col: str, y_col: str, save_path: str = None):
    """Legacy 2D single regression scatter with regression line."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.4, label='Actual Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title(f'Linear Regression: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_residuals(y_test, y_pred, save_path: str = None):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='purple', alpha=0.5)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
