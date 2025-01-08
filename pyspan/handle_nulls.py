import pandas as pd
import numpy as np
import json
from typing import Optional, Union, List
from .logging_utils import log_function_call
from .state_management import track_changes

@log_function_call
@track_changes
def handle_nulls(
    df: Union[pd.DataFrame, List, dict, tuple, np.ndarray, str],
    columns: Optional[Union[str, List[str]]] = None,
    action: str = 'remove',
    with_val: Optional[Union[int, float, str]] = None,
    by: Optional[str] = None,
    inplace: bool = False,
    threshold: Optional[float] = None,
    axis: str = 'rows'
) -> Optional[Union[pd.DataFrame, List, dict, tuple, np.ndarray, str]]:
    """
    Handle null values in various data types by removing, replacing, imputing, or applying a threshold.

    Parameters:
    - df (Union[pd.DataFrame, List, dict, tuple, np.ndarray, str]): Input data.
    - columns (Optional[Union[str, List[str]]]): Column(s) to handle nulls in.
    - action (str): Action to take ('remove', 'replace', or 'impute').
    - with_val (Optional[Union[int, float, str]]): Value to replace nulls with.
    - by (Optional[str]): Imputation strategy.
    - inplace (bool): Whether to modify the input data in place.
    - threshold (Optional[float]): Threshold for null handling.
    - axis (str): Axis to apply threshold ('rows' or 'columns').

    Returns:
    - Optional[Union[pd.DataFrame, List, dict, tuple, np.ndarray, str]]: Modified data or None if inplace=True.
    """
    # Validate axis
    if axis not in ['rows', 'columns']:
        raise ValueError("The 'axis' parameter must be either 'rows' or 'columns'.")

    # Validate threshold
    if threshold is not None and not (0 <= threshold <= 100):
        raise ValueError("The 'threshold' parameter must be between 0 and 100.")

    # Process Pandas DataFrame
    if isinstance(df, pd.DataFrame):
        if threshold is not None:
            if axis == 'rows':
                df_cleaned = df.dropna(thresh=int((1 - threshold / 100) * df.shape[1]), axis=0)
            else:
                df_cleaned = df.dropna(thresh=int((1 - threshold / 100) * df.shape[0]), axis=1)
        elif action == 'remove':
            df_cleaned = df.dropna(subset=columns if columns else None)
        elif action == 'replace':
            if with_val is None:
                raise ValueError("A value must be provided when action is 'replace'.")
            df_cleaned = df.fillna({col: with_val for col in columns} if columns else with_val)
        elif action == 'impute':
            strategies = {
                'mean': lambda col: col.fillna(col.mean()),
                'median': lambda col: col.fillna(col.median()),
                'mode': lambda col: col.fillna(col.mode().iloc[0] if not col.mode().empty else col),
                'interpolate': lambda col: col.interpolate(),
                'forward_fill': lambda col: col.ffill(),
                'backward_fill': lambda col: col.bfill(),
            }
            if by not in strategies:
                raise ValueError("Invalid impute strategy. Use one of ['mean', 'median', 'mode', 'interpolate', 'forward_fill', 'backward_fill'].")
            df_cleaned = df.copy()
            for col in (columns if columns else df.columns):
                df_cleaned[col] = strategies[by](df_cleaned[col])
        else:
            raise ValueError("Action must be either 'remove', 'replace', or 'impute'.")

    # Process List or Tuple
    elif isinstance(df, (list, tuple)):
        df_cleaned = [val for val in df if val is not None and not (isinstance(val, float) and np.isnan(val))]
        if isinstance(df, tuple):
            df_cleaned = tuple(df_cleaned)

    # Process Dictionary
    elif isinstance(df, dict):
        df_cleaned = {}
        for key, values in df.items():
            if not isinstance(values, list):
                raise ValueError("All dictionary values must be lists for processing.")
            cleaned_values = [val if val is not None else with_val for val in values]
            if threshold is not None:
                valid_count = len([val for val in cleaned_values if val is not None])
                if valid_count / len(cleaned_values) < (1 - threshold / 100):
                    cleaned_values = []
            df_cleaned[key] = cleaned_values

    # Process JSON String
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)  # Parse JSON
            if isinstance(parsed_data, dict):
                for key, values in parsed_data.items():
                    if not isinstance(values, list):
                        raise ValueError("All JSON dictionary values must be lists for processing.")
                    parsed_data[key] = [val if val is not None else with_val for val in values]
                df_cleaned = json.dumps(parsed_data)
            else:
                raise ValueError("JSON must represent a dictionary of lists.")
        except json.JSONDecodeError:
            # Treat as a simple string
            df_cleaned = df if df.strip() else with_val

    # Process NumPy Array
    elif isinstance(df, np.ndarray):
        df_cleaned = np.where(pd.isnull(df), with_val, df)
        if threshold is not None:
            valid_count = np.count_nonzero(~pd.isnull(df))
            if valid_count / df.size < (1 - threshold / 100):
                df_cleaned = np.array([])

    else:
        raise TypeError("Unsupported data type. Supported types: pd.DataFrame, list, tuple, dict, str, np.ndarray.")

    # Handle inplace modifications
    if inplace:
        if isinstance(df, pd.DataFrame):
            df.update(df_cleaned)
            return None
        elif isinstance(df, (list, dict)):
            df[:] = df_cleaned
            return None
        elif isinstance(df, tuple):
            raise ValueError("Inplace modification is not supported for tuples.")
    return df_cleaned 


