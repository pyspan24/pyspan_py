import pandas as pd
import numpy as np
import json
from typing import Union, List, Optional
from .logging_utils import log_function_call
from .state_management import track_changes

@log_function_call
@track_changes
def reformat(
    df: Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray],
    target_column: str,
    reference_column: str
) -> Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]:
    """
    Applies the data type and formatting from a reference column to a target column in the same DataFrame or equivalent structure.

    Parameters:
    - df (Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]): The input data.
    - target_column (str): The column or field to format.
    - reference_column (str): The column or field to borrow formatting from.

    Returns:
    - Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]: The formatted data.

    Raises:
    - ValueError: If the target column or reference column does not exist in the DataFrame or equivalent.
    - TypeError: If the reference column is not of a type that can be applied to the target column.
    """

    # Detect original data format
    original_format = type(df)

    # Handle different data types separately without converting them to DataFrame
    if isinstance(df, pd.DataFrame):
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")
        if reference_column not in df.columns:
            raise ValueError(f"Column '{reference_column}' does not exist in the DataFrame.")
        ref_dtype = df[reference_column].dtype

        # Check and apply formatting based on data type
        if pd.api.types.is_datetime64_any_dtype(ref_dtype):
            try:
                df[target_column] = pd.to_datetime(df[target_column], errors='coerce')
            except Exception as e:
                raise TypeError(f"Error converting '{target_column}' to datetime: {e}")
        elif pd.api.types.is_numeric_dtype(ref_dtype):
            try:
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            except Exception as e:
                raise TypeError(f"Error converting '{target_column}' to numeric: {e}")
        elif pd.api.types.is_string_dtype(ref_dtype):
            ref_sample = df[reference_column].dropna().astype(str).iloc[0]
            if ref_sample.isupper():
                df[target_column] = df[target_column].astype(str).str.upper()
            elif ref_sample.islower():
                df[target_column] = df[target_column].astype(str).str.lower()
            elif ref_sample.istitle():
                df[target_column] = df[target_column].astype(str).str.title()
            else:
                df[target_column] = df[target_column].astype(str)

        return df

    elif isinstance(df, pd.Series):
        if target_column not in df.index:
            raise ValueError(f"Column '{target_column}' does not exist in the Series.")
        if reference_column not in df.index:
            raise ValueError(f"Column '{reference_column}' does not exist in the Series.")
        ref_dtype = df[reference_column].dtype

        if pd.api.types.is_datetime64_any_dtype(ref_dtype):
            try:
                df[target_column] = pd.to_datetime(df[target_column], errors='coerce')
            except Exception as e:
                raise TypeError(f"Error converting '{target_column}' to datetime: {e}")
        elif pd.api.types.is_numeric_dtype(ref_dtype):
            try:
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            except Exception as e:
                raise TypeError(f"Error converting '{target_column}' to numeric: {e}")
        elif pd.api.types.is_string_dtype(ref_dtype):
            ref_sample = df[reference_column].dropna().astype(str).iloc[0]
            if ref_sample.isupper():
                df[target_column] = df[target_column].astype(str).str.upper()
            elif ref_sample.islower():
                df[target_column] = df[target_column].astype(str).str.lower()
            elif ref_sample.istitle():
                df[target_column] = df[target_column].astype(str).str.title()
            else:
                df[target_column] = df[target_column].astype(str)

        return df

    elif isinstance(df, list):
        if isinstance(df[0], dict):
            if target_column not in df[0] or reference_column not in df[0]:
                raise ValueError(f"Column '{target_column}' or '{reference_column}' does not exist in the list of dictionaries.")
            ref_dtype = type(df[0][reference_column])

            for item in df:
                if ref_dtype == str:
                    item[target_column] = str(item[target_column])
                elif isinstance(item[reference_column], (int, float)):
                    item[target_column] = float(item[target_column]) if isinstance(item[target_column], (int, float)) else float('nan')
                elif isinstance(item[reference_column], bool):
                    item[target_column] = bool(item[target_column])
                elif isinstance(item[reference_column], pd.Timestamp):
                    item[target_column] = pd.to_datetime(item[target_column], errors='coerce')

            return df
        else:
            raise TypeError("List elements must be dictionaries or records.")

    elif isinstance(df, tuple):
        if isinstance(df[0], dict):
            if target_column not in df[0] or reference_column not in df[0]:
                raise ValueError(f"Column '{target_column}' or '{reference_column}' does not exist in the tuple of dictionaries.")
            ref_dtype = type(df[0][reference_column])

            result = []
            for item in df:
                if ref_dtype == str:
                    item[target_column] = str(item[target_column])
                elif isinstance(item[reference_column], (int, float)):
                    item[target_column] = float(item[target_column]) if isinstance(item[target_column], (int, float)) else float('nan')
                elif isinstance(item[reference_column], bool):
                    item[target_column] = bool(item[target_column])
                elif isinstance(item[reference_column], pd.Timestamp):
                    item[target_column] = pd.to_datetime(item[target_column], errors='coerce')

                result.append(item)
            return tuple(result)
        else:
            raise TypeError("Tuple elements must be dictionaries or records.")

    elif isinstance(df, dict):
        if target_column not in df or reference_column not in df:
            raise ValueError(f"Column '{target_column}' or '{reference_column}' does not exist in the dictionary.")
        ref_dtype = type(df[reference_column])

        for key, value in df.items():
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, dict) and reference_column in item and target_column in item:
                        if ref_dtype == str:
                            item[target_column] = str(item[target_column])
                        elif isinstance(item[reference_column], (int, float)):
                            item[target_column] = float(item[target_column]) if isinstance(item[target_column], (int, float)) else float('nan')
                        elif isinstance(item[reference_column], bool):
                            item[target_column] = bool(item[target_column])
                        elif isinstance(item[reference_column], pd.Timestamp):
                            item[target_column] = pd.to_datetime(item[target_column], errors='coerce')

        return df

    elif isinstance(df, np.ndarray):
        # Handle numpy arrays (assuming structured array for column access)
        if target_column not in df.dtype.names or reference_column not in df.dtype.names:
            raise ValueError(f"Column '{target_column}' or '{reference_column}' does not exist in the numpy array.")

        ref_dtype = df[reference_column].dtype

        if pd.api.types.is_datetime64_any_dtype(ref_dtype):
            df[target_column] = pd.to_datetime(df[target_column], errors='coerce')
        elif pd.api.types.is_numeric_dtype(ref_dtype):
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        elif pd.api.types.is_string_dtype(ref_dtype):
            ref_sample = str(df[reference_column][0])
            if ref_sample.isupper():
                df[target_column] = np.char.upper(df[target_column])
            elif ref_sample.islower():
                df[target_column] = np.char.lower(df[target_column])
            elif ref_sample.istitle():
                df[target_column] = np.char.title(df[target_column])
            else:
                df[target_column] = np.char.mod('%s', df[target_column])

        return df

    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                if target_column not in parsed_data or reference_column not in parsed_data:
                    raise ValueError(f"Column '{target_column}' or '{reference_column}' does not exist in the JSON string.")
                ref_dtype = type(parsed_data[reference_column])
                if ref_dtype == str:
                    parsed_data[target_column] = str(parsed_data[target_column])
                elif isinstance(parsed_data[reference_column], (int, float)):
                    parsed_data[target_column] = float(parsed_data[target_column]) if isinstance(parsed_data[target_column], (int, float)) else float('nan')
                elif isinstance(parsed_data[reference_column], bool):
                    parsed_data[target_column] = bool(parsed_data[target_column])
                elif isinstance(parsed_data[reference_column], pd.Timestamp):
                    parsed_data[target_column] = pd.to_datetime(parsed_data[target_column], errors='coerce')

                return json.dumps(parsed_data)

        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string input.")

    else:
        raise TypeError(f"Unsupported data type: {type(df)}") 