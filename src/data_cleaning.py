import pandas as pd
import numpy as np

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def add_identifiers(df: pd.DataFrame, company: str, year: int = None) -> pd.DataFrame:
    """Add Company and Year identifiers to the dataset."""
    df.insert(0, 'Company', company)
    if year is not None:
        df.insert(1, 'Year', year)
    return df

def report_missing_values(dataframes: dict) -> pd.DataFrame:
    """Generate a report of missing values across multiple dataframes."""
    report = []
    for name, df in dataframes.items():
        missing_count = df.isnull().sum().sum()
        treatment = "None" if missing_count == 0 else "Investigated / Imputed"
        report.append({
            "Dataset": name,
            "Missing Values": missing_count,
            "Treatment": treatment
        })
    return pd.DataFrame(report)

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Detect outliers using the Interquartile Range (IQR) method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers using the Z-score method."""
    mean = df[column].mean()
    std = df[column].std()
    
    # Avoid division by zero
    if std == 0:
        return pd.DataFrame(columns=df.columns)
        
    z_scores = np.abs((df[column] - mean) / std)
    outliers = df[z_scores > threshold]
    return outliers
