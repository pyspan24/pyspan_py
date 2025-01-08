import pandas as pd
import numpy as np
import json
from typing import Union, List, Optional
from .logging_utils import log_function_call
from .state_management import track_changes

@log_function_call
@track_changes
def scale_data(
    df: Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray],
    columns: Union[str, List[str], None] = None,
    method: str = 'min-max'
) -> Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]:
    """
    Scale numerical data using various scaling methods.

    Parameters:
    - df (Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]): The input data to scale.
    - columns (str, list of str, or None): The specific column(s) to scale. If None, scales all numeric columns.
    - method (str): The scaling method to use. Options are 'min-max', 'robust', or 'standard'.

    Returns:
    - Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]: The scaled data.

    Raises:
    - ValueError: If the scaling method is not recognized or if columns don't exist.
    - TypeError: If the input data type is not supported.
    """
    # Detect original data format
    original_format = type(df)

    # Handle DataFrame input
    if isinstance(df, pd.DataFrame):
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            columns = numeric_columns
        elif isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

        if method == 'min-max':
            for col in columns:
                X = df[col]
                X_min = X.min()
                X_max = X.max()
                df[col] = (X - X_min) / (X_max - X_min)

        elif method == 'robust':
            for col in columns:
                X = df[col]
                median = X.median()
                IQR = X.quantile(0.75) - X.quantile(0.25)
                df[col] = (X - median) / IQR

        elif method == 'standard':
            for col in columns:
                X = df[col]
                mean = X.mean()
                std = X.std()
                df[col] = (X - mean) / std

        else:
            raise ValueError(f"Scaling method '{method}' is not recognized. Choose from 'min-max', 'robust', or 'standard'.")

        return df

    # Handle Series input
    elif isinstance(df, pd.Series):
        if isinstance(columns, list):
            raise ValueError("For Series, 'columns' should not be specified.")
        
        if method == 'min-max':
            X = df
            X_min = X.min()
            X_max = X.max()
            return (X - X_min) / (X_max - X_min)

        elif method == 'robust':
            X = df
            median = X.median()
            IQR = X.quantile(0.75) - X.quantile(0.25)
            return (X - median) / IQR

        elif method == 'standard':
            X = df
            mean = X.mean()
            std = X.std()
            return (X - mean) / std

        else:
            raise ValueError(f"Scaling method '{method}' is not recognized. Choose from 'min-max', 'robust', or 'standard'.")

    # Handle list input
    elif isinstance(df, list):
        if method == 'min-max':
            X_min = min(df)
            X_max = max(df)
            return [(x - X_min) / (X_max - X_min) for x in df]

        elif method == 'robust':
            median = np.median(df)
            IQR = np.percentile(df, 75) - np.percentile(df, 25)
            return [(x - median) / IQR for x in df]

        elif method == 'standard':
            mean = np.mean(df)
            std = np.std(df)
            return [(x - mean) / std for x in df]

        else:
            raise ValueError(f"Scaling method '{method}' is not recognized. Choose from 'min-max', 'robust', or 'standard'.")

    # Handle tuple input
    elif isinstance(df, tuple):
        df_list = list(df)
        
        if method == 'min-max':
            X_min = min(df_list)
            X_max = max(df_list)
            return tuple((x - X_min) / (X_max - X_min) for x in df_list)

        elif method == 'robust':
            median = np.median(df_list)
            IQR = np.percentile(df_list, 75) - np.percentile(df_list, 25)
            return tuple((x - median) / IQR for x in df_list)

        elif method == 'standard':
            mean = np.mean(df_list)
            std = np.std(df_list)
            return tuple((x - mean) / std for x in df_list)

        else:
            raise ValueError(f"Scaling method '{method}' is not recognized. Choose from 'min-max', 'robust', or 'standard'.")

    # Handle dictionary input
    elif isinstance(df, dict):
        scaled_dict = {}
        for key, value in df.items():
            if isinstance(value, list):  # Handle list values within dict
                if method == 'min-max':
                    X_min = min(value)
                    X_max = max(value)
                    scaled_dict[key] = [(x - X_min) / (X_max - X_min) for x in value]

                elif method == 'robust':
                    median = np.median(value)
                    IQR = np.percentile(value, 75) - np.percentile(value, 25)
                    scaled_dict[key] = [(x - median) / IQR for x in value]

                elif method == 'standard':
                    mean = np.mean(value)
                    std = np.std(value)
                    scaled_dict[key] = [(x - mean) / std for x in value]

                else:
                    raise ValueError(f"Scaling method '{method}' is not recognized. Choose from 'min-max', 'robust', or 'standard'.")
            else:
                scaled_dict[key] = value  # If not a list, leave the value unchanged
        return scaled_dict

    # Handle numpy array input
    elif isinstance(df, np.ndarray):
        if method == 'min-max':
            X_min = df.min()
            X_max = df.max()
            return (df - X_min) / (X_max - X_min)

        elif method == 'robust':
            median = np.median(df)
            IQR = np.percentile(df, 75) - np.percentile(df, 25)
            return (df - median) / IQR

        elif method == 'standard':
            mean = np.mean(df)
            std = np.std(df)
            return (df - mean) / std

        else:
            raise ValueError(f"Scaling method '{method}' is not recognized. Choose from 'min-max', 'robust', or 'standard'.")

    # Handle string input (we assume no scaling on strings, just return as is)
    elif isinstance(df, str):
        return df

    else:
        raise TypeError(f"Unsupported data type: {original_format}. Please provide a DataFrame, Series, list, dict, tuple, numpy array, or JSON string.") 