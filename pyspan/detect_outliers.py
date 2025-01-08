import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import json
from typing import Union, List
from .logging_utils import log_function_call
from .state_management import track_changes

@log_function_call
@track_changes
def detect_outliers(
    df,
    columns,
    method='iqr',
    threshold=1.5,
    handle_missing=True,
    anomaly_method=None,
    contamination=0.05,
    n_neighbors=20,
    eps=0.5,
    min_samples=5
):
    """
    Detects outliers or anomalies in specified columns using IQR, Z-Score, or anomaly detection methods.

    Parameters:
    - df (DataFrame, Series, list, dict, tuple, np.ndarray, str): The data to analyze.
    - columns (str or list): The column name or list of column names to detect outliers/anomalies.
    - method (str): The method to use ('iqr' for Interquartile Range or 'z-score' for Z-Score). Default is 'iqr'.
    - threshold (float): The threshold for outlier detection. Default is 1.5 for IQR and 3 for Z-Score.
    - handle_missing (bool): If True, handles missing values by removing rows with missing data.
    - anomaly_method (str): Specify 'isolation_forest', 'lof', or 'dbscan' for anomaly detection methods.
    - contamination (float): The proportion of anomalies for 'isolation_forest' and 'lof'. Default is 0.05.
    - n_neighbors (int): Number of neighbors for LOF. Default is 20.
    - eps (float): The maximum distance between samples for DBSCAN. Default is 0.5.
    - min_samples (int): The minimum number of samples in a neighborhood for DBSCAN. Default is 5.

    Returns:
    - The processed data with outliers removed, in the original data format.
    """
    # Convert dictionary to DataFrame if input is a dictionary
    if isinstance(df, dict):
        df = pd.DataFrame(df)

    # Detect if input is DataFrame, Series, List, Tuple, Dict, or String
    if isinstance(df, pd.DataFrame):
        data_type = 'DataFrame'
    elif isinstance(df, pd.Series):
        data_type = 'Series'
    elif isinstance(df, (list, tuple)):
        data_type = 'list/tuple'
    elif isinstance(df, str):
        data_type = 'str'
    elif isinstance(df, np.ndarray):
        data_type = 'ndarray'
    else:
        raise ValueError(f"Unsupported data type: {type(df)}")

    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]

    # Handle missing values, depending on the type
    if handle_missing:
        if data_type == 'DataFrame':
            df = df.dropna(subset=columns)
        elif data_type == 'Series':
            df = df.dropna()
        elif data_type in ['list/tuple', 'ndarray']:
            df = [item for item in df if item is not None]
        elif data_type == 'str':
            df = df.strip()  # Strip any surrounding whitespace from string

    # Initialize mask for outliers
    combined_outliers = np.zeros(len(df), dtype=bool)

    for column in columns:
        if data_type == 'DataFrame' or data_type == 'Series':
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame.")
        elif data_type in ['list/tuple', 'ndarray']:
            # Handle different data structures that don't have columns (i.e., for list/tuple/ndarray)
            if isinstance(column, int):  # Handle case if column is an index for list/tuple/ndarray
                if data_type == 'list/tuple' and column >= len(df):
                    raise ValueError(f"Index '{column}' is out of bounds for the data.")
            elif isinstance(column, str) and column not in df:
                raise ValueError(f"Key '{column}' not found in the data.")

        # Outlier detection logic (IQR, Z-Score, or anomaly methods)
        if anomaly_method is None:
            outlier_type = "outliers"
            if method == 'iqr':
                if data_type == 'DataFrame':
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                elif data_type == 'Series':
                    Q1 = df.quantile(0.25)
                    Q3 = df.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                if data_type == 'DataFrame':
                    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                elif data_type == 'Series':
                    outliers = (df < lower_bound) | (df > upper_bound)
            elif method == 'z-score':
                if data_type == 'DataFrame':
                    mean = df[column].mean()
                    std = df[column].std()
                    z_scores = (df[column] - mean) / std
                    outliers = np.abs(z_scores) > threshold
                elif data_type == 'Series':
                    mean = df.mean()
                    std = df.std()
                    z_scores = (df - mean) / std
                    outliers = np.abs(z_scores) > threshold
        else:
            outlier_type = "anomalies"
            if anomaly_method == 'isolation_forest':
                model = IsolationForest(contamination=contamination, random_state=42)
                if data_type == 'DataFrame':
                    outliers = model.fit_predict(df[[column]]) == -1
                elif data_type == 'Series':
                    outliers = model.fit_predict(df.values.reshape(-1, 1)) == -1
                elif data_type in ['list/tuple', 'ndarray']:
                    outliers = model.fit_predict(np.array(df).reshape(-1, 1)) == -1
            elif anomaly_method == 'lof':
                lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                if data_type == 'DataFrame':
                    outliers = lof_model.fit_predict(df[[column]]) == -1
                elif data_type == 'Series':
                    outliers = lof_model.fit_predict(df.values.reshape(-1, 1)) == -1
                elif data_type in ['list/tuple', 'ndarray']:
                    outliers = lof_model.fit_predict(np.array(df).reshape(-1, 1)) == -1
            elif anomaly_method == 'dbscan':
                dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
                if data_type == 'DataFrame':
                    outliers = dbscan_model.fit_predict(df[[column]]) == -1
                elif data_type == 'Series':
                    outliers = dbscan_model.fit_predict(df.values.reshape(-1, 1)) == -1
                elif data_type in ['list/tuple', 'ndarray']:
                    outliers = dbscan_model.fit_predict(np.array(df).reshape(-1, 1)) == -1
            else:
                raise ValueError(f"Anomaly method '{anomaly_method}' is not supported.")

        combined_outliers |= outliers

        num_outliers = np.sum(outliers)
        print(f"Total number of {outlier_type} detected in column '{column}': {num_outliers}")

    # Remove all detected outliers
    if data_type == 'DataFrame':
        df_cleaned = df[~combined_outliers]
    elif data_type == 'Series':
        df_cleaned = df[~combined_outliers]
    elif data_type in ['list/tuple', 'ndarray']:
        df_cleaned = [item for i, item in enumerate(df) if not combined_outliers[i]]
    elif data_type == 'dict':
        df_cleaned = {key: value for i, (key, value) in enumerate(df.items()) if not combined_outliers[i]}
    elif data_type == 'str':
        df_cleaned = df  # No removal for string types, but can adjust based on use case

    # Print original and final data shape or size
    if data_type == 'DataFrame':
        print(f"Original data shape: {df.shape}")
        print(f"Data shape after removing {outlier_type}: {df_cleaned.shape}")
    elif data_type == 'Series':
        print(f"Original data size: {df.size}")
        print(f"Data size after removing {outlier_type}: {df_cleaned.size}")
    elif data_type in ['list/tuple', 'ndarray']:
        print(f"Original data size: {len(df)}")
        print(f"Data size after removing {outlier_type}: {len(df_cleaned)}")
    elif data_type == 'dict':
        print(f"Original data size: {len(df)}")
        print(f"Data size after removing {outlier_type}: {len(df_cleaned)}")
    elif data_type == 'str':
        print(f"Original data size: {len(df)}")
        print(f"Data size after removing {outlier_type}: {len(df_cleaned)}")

    return df_cleaned 