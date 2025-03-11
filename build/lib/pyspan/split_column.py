import pandas as pd
import numpy as np
from collections import Counter
import json
from typing import Union
from .logging_utils import log_function_call
from .state_management import track_changes

def detect_delimiter(series: pd.Series) -> str:
    """
    Detect the most common delimiter in a Series of strings.

    Parameters:
    - series (pd.Series): A Series containing strings to analyze.

    Returns:
    - str: The most common delimiter found.
    """
    # Define a list of potential delimiters
    potential_delimiters = [',', ';', '|', '\t', ':', ' ']

    # Collect all delimiters used in the series
    delimiters = [delimiter for text in series.dropna() for delimiter in potential_delimiters if delimiter in text]

    # If no delimiters are found, return None
    if not delimiters:
        return None

    # Return the most common delimiter
    most_common_delimiter, _ = Counter(delimiters).most_common(1)[0]
    return most_common_delimiter

@log_function_call
@track_changes
def split_column(
    df: Union[pd.DataFrame, list, dict, tuple, np.ndarray, str],
    column: str,
    delimiter: str = None
) -> Union[pd.DataFrame, list, dict, tuple, str]:
    """
    Split a single column into multiple columns based on a delimiter.

    Parameters:
    - df (Union[pd.DataFrame, list, dict, tuple, np.ndarray, str]): The data containing the column to split.
    - column (str): The name of the column to be split.
    - delimiter (str): The delimiter to use for splitting. If None, detect the most common delimiter.

    Returns:
    - Union[pd.DataFrame, list, dict, tuple, str]: A new data structure with the specified column split into multiple columns.

    Raises:
    - ValueError: If the column does not exist, no delimiter is detected, or an invalid delimiter is provided.
    """
    # If input is a DataFrame
    if isinstance(df, pd.DataFrame):
        df_converted = df

        # Validate input
        if column not in df_converted.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        if delimiter is None:
            # Detect the delimiter if not provided
            delimiter = detect_delimiter(df_converted[column])
            if delimiter is None:
                raise ValueError("No delimiter detected and none provided.")
        else:
            # If the user provides a delimiter, use it regardless of the predefined list
            if not isinstance(delimiter, str) or len(delimiter) == 0:
                raise ValueError("Provided delimiter must be a non-empty string.")

        # Split the column based on the delimiter
        expanded_columns = df_converted[column].str.split(delimiter, expand=True)

        # Remove any columns that are entirely NaN (i.e., extra columns)
        expanded_columns = expanded_columns.dropna(how='all', axis=1)

        # Drop the original column and concatenate the new columns
        df_expanded = df_converted.drop(columns=[column]).join(expanded_columns)

        # Rename new columns with a suffix to identify them
        df_expanded.columns = list(df_expanded.columns[:-len(expanded_columns.columns)]) + \
                              [f"{column}_{i+1}" for i in range(len(expanded_columns.columns))]

        return df_expanded

    # If input is a list of dicts
    elif isinstance(df, list):
        # Ensure each element of the list is a dictionary (if list of dicts)
        if not all(isinstance(item, dict) for item in df):
            raise TypeError("List input must contain dictionaries.")

        for item in df:
            if column not in item:
                raise ValueError(f"Column '{column}' does not exist in one of the dictionaries.")

        if delimiter is None:
            # Detect the delimiter if not provided (based on first dictionary entry)
            delimiter = detect_delimiter(pd.Series([item[column] for item in df]))
            if delimiter is None:
                raise ValueError("No delimiter detected and none provided.")

        # Split each dictionary's column value based on the delimiter
        for item in df:
            item[column] = item[column].split(delimiter)

        return df

    # If input is a dictionary of lists
    elif isinstance(df, dict):
        if column not in df:
            raise ValueError(f"Column '{column}' does not exist in the dictionary.")

        if delimiter is None:
            # Detect the delimiter if not provided
            delimiter = detect_delimiter(pd.Series(df[column]))
            if delimiter is None:
                raise ValueError("No delimiter detected and none provided.")

        # Split each column's list value based on the delimiter
        df[column] = [value.split(delimiter) for value in df[column]]

        return df

    # If input is a tuple
    elif isinstance(df, tuple):
        df_converted = list(df)

        if len(df_converted) <= 1:
            raise ValueError("Tuple must have more than one item to split.")

        if delimiter is None:
            # Detect the delimiter if not provided (based on the first item in the tuple)
            delimiter = detect_delimiter(pd.Series(df_converted))
            if delimiter is None:
                raise ValueError("No delimiter detected and none provided.")

        # Split each element based on the delimiter
        for i in range(len(df_converted)):
            df_converted[i] = df_converted[i].split(delimiter)

        return tuple(tuple(x) for x in df_converted)

    # If input is a string (assuming JSON-like string)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                return split_column(parsed_data, column, delimiter)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")

    # If input is of an unsupported type
    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, list, tuple, dict, or JSON string.") 