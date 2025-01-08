import pandas as pd
import numpy as np
import json
from typing import Optional, Union, List
from .logging_utils import log_function_call
from .state_management import track_changes

@log_function_call
@track_changes
def remove(
    df: Union[pd.DataFrame, List, dict, tuple, np.ndarray, str],
    operation: str,
    columns: Optional[Union[str, List[str]]] = None,
    keep: Optional[str] = 'first',
    consider_all: bool = True,
    inplace: bool = False
) -> Optional[Union[pd.DataFrame, List, dict, tuple, str]]:
    """
    Remove duplicates or columns from a DataFrame or other data structures based on the specified operation.

    Parameters:
    - df (Union[pd.DataFrame, List, dict, tuple, np.ndarray, str]): Input data.
    - operation (str): Type of removal operation. Options are:
      - 'duplicates': Remove duplicate rows.
      - 'columns': Remove specified columns.
    - columns (Optional[Union[str, List[str]]]): Column(s) to consider for the operation.
      - For 'duplicates': Columns to check for duplicates.
      - For 'columns': Column(s) to be removed.
    - keep (Optional[str]): Determines which duplicates to keep. Options are:
      - 'first': Keep the first occurrence of each duplicate.
      - 'last': Keep the last occurrence of each duplicate.
      - 'none': Remove all duplicates.
      Default is 'first'.
    - consider_all (bool): Whether to consider all columns in the DataFrame if duplicates are found in the specified columns.
      True means removing the entire row if any duplicates are found in the specified columns. Default is True.
    - inplace (bool): If True, modify the data in place. If False, return a new data structure. Default is False.

    Returns:
    - Optional[Union[pd.DataFrame, List, dict, tuple, str]]: Updated data or None if inplace=True.

    Raises:
    - ValueError: If invalid columns are specified or operation is invalid.
    - TypeError: If input types are incorrect.
    """

    # Convert non-Pandas data to a DataFrame
    original_format = None
    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, (list, tuple, np.ndarray)):
        original_format = type(df)
        df = pd.DataFrame(df)
    elif isinstance(df, dict):
        original_format = 'dict'
        df = pd.DataFrame.from_dict(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                original_format = 'json'
                df = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String inputs must be JSON objects.")
        except json.JSONDecodeError:
            raise ValueError("String inputs must be valid JSON.")
    else:
        raise TypeError(
            "Unsupported data type. Supported types are: pd.DataFrame, list, tuple, dict, NumPy array, and JSON string."
        )

    # Handle 'duplicates' operation
    if operation == 'duplicates':
        if keep not in ['first', 'last', 'none', None]:
            raise ValueError("keep must be one of ['first', 'last', 'none'] or None.")

        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Columns {missing_columns} not found in DataFrame.")

        if keep == 'none':
            df_cleaned = df.drop_duplicates(subset=columns, keep=False)
        else:
            if consider_all:
                df_cleaned = df.drop_duplicates(subset=columns, keep=keep or 'first')
            else:
                df_cleaned = df.drop_duplicates(keep=keep or 'first')

    # Handle 'columns' operation
    elif operation == 'columns':
        if isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise TypeError("columns must be a string or a list of column names.")

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in DataFrame.")

        df_cleaned = df.drop(columns=columns)

    else:
        raise ValueError("operation must be either 'duplicates' or 'columns'.")

    # Handle inplace modification
    if inplace:
        if isinstance(df, pd.DataFrame):
            df.update(df_cleaned)
        elif original_format in [list, tuple, 'dict']:
            raise ValueError("Inplace modification is not supported for non-Pandas formats.")
        return None

    # Convert back to original format if needed
    if original_format == list:
        return df_cleaned.values.tolist()
    elif original_format == tuple:
        return tuple(map(tuple, df_cleaned.values))
    elif original_format == 'dict':
        return df_cleaned.to_dict(orient="list")
    elif original_format == 'json':
        return df_cleaned.to_json()
    else:
        return df_cleaned