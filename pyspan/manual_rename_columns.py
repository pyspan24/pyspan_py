import numpy as np
import pandas as pd
import json
from typing import Union, Dict
from .logging_utils import log_function_call
from .state_management import track_changes

@log_function_call
@track_changes
def manual_rename_columns(
    df: Union[pd.DataFrame, list, dict, tuple, np.ndarray, str],
    rename_dict: dict
) -> Union[pd.DataFrame, list, dict, tuple, str]:
    """
    Rename columns in a DataFrame using a provided dictionary mapping.

    Parameters:
    - df (Union[pd.DataFrame, list, dict, tuple, np.ndarray, str]): The DataFrame (or data) whose columns are to be renamed.
    - rename_dict (dict): A dictionary mapping current column names (keys) to new column names (values).

    Returns:
    - Union[pd.DataFrame, list, dict, tuple, str]: A new DataFrame (or data type) with columns renamed according to the dictionary.

    Raises:
    - KeyError: If any of the specified columns in the dictionary keys do not exist in the DataFrame.
    - ValueError: If the rename_dict is empty.
    """

    # Check for invalid rename_dict
    if not isinstance(rename_dict, dict) or not rename_dict:
        raise ValueError("rename_dict must be a non-empty dictionary.")

    if isinstance(df, pd.DataFrame):
        # Check for columns that are not in the DataFrame
        missing_columns = [col for col in rename_dict if col not in df.columns]
        if missing_columns:
            raise KeyError(f"These columns do not exist in the DataFrame: {missing_columns}")

        # Rename columns using the provided dictionary
        renamed_df = df.rename(columns=rename_dict)
        return renamed_df

    elif isinstance(df, list):
        # Handle case for list of dictionaries (which can be mapped to columns)
        if all(isinstance(item, dict) for item in df):
            renamed_list = []
            for item in df:
                renamed_item = {rename_dict.get(k, k): v for k, v in item.items()}
                renamed_list.append(renamed_item)
            return renamed_list
        else:
            raise TypeError("For a list, elements must be dictionaries with keys as column names.")

    elif isinstance(df, tuple):
        # Handle case for tuple of dictionaries (similar to list)
        if all(isinstance(item, dict) for item in df):
            renamed_tuple = tuple(
                {rename_dict.get(k, k): v for k, v in item.items()} for item in df
            )
            return renamed_tuple
        else:
            raise TypeError("For a tuple, elements must be dictionaries with keys as column names.")

    elif isinstance(df, np.ndarray):
        # Handle case for ndarray, assuming it's 2D (columns are present in the first row)
        if df.ndim == 2:
            columns = df[0]
            renamed_columns = [rename_dict.get(col, col) for col in columns]
            renamed_array = np.vstack([renamed_columns, df[1:]])
            return renamed_array
        else:
            raise ValueError("Numpy array must be 2D to map column names.")

    elif isinstance(df, dict):
        # Handle case for a dictionary of lists (columns as keys)
        if all(isinstance(v, list) for v in df.values()):
            renamed_dict = {rename_dict.get(k, k): v for k, v in df.items()}
            return renamed_dict
        else:
            raise TypeError("For a dictionary, values must be lists representing column data.")

    elif isinstance(df, str):
        # Handle case for JSON string
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                renamed_dict = {rename_dict.get(k, k): v for k, v in parsed_data.items()}
                return json.dumps(renamed_dict)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")

    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, list, tuple, dict, or JSON string.") 