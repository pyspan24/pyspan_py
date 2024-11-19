import inspect
import logging
import functools

# Store logs in a list for retrieval
log_entries = []

def log_function_call(func):
    @functools.wraps(func)  # This line preserves the original function's metadata
    def wrapper(*args, **kwargs):
        # Get the arguments passed to the decorated function (args and kwargs)
        arg_names = inspect.getfullargspec(func).args
        arg_vals = args[:len(arg_names)]  # Positional arguments
        kwarg_vals = kwargs  # Keyword arguments

        # Prepare a readable argument list, handling large objects (like DataFrames)
        def readable_repr(val):
            if isinstance(val, pd.DataFrame):
                return f'<DataFrame: {val.shape[0]} rows x {val.shape[1]} columns>'
            return val

        # Combine args and kwargs into a readable string
        arg_repr = ', '.join(f'{name}={readable_repr(val)}' for name, val in zip(arg_names, arg_vals))
        kwarg_repr = ', '.join(f'{key}={readable_repr(value)}' for key, value in kwarg_vals.items())

        # If no arguments are passed
        all_args_repr = f'{arg_repr}, {kwarg_repr}'.strip(', ')

        if not all_args_repr:
            all_args_repr = 'None'  # Show "None" if no arguments are passed
        
        # Get the calling line number
        caller_info = inspect.stack()[1]
        line_number = caller_info.lineno

        # Create a human-readable log entry with line number
        log_entry = f'Function "{func.__name__}" was called at line {line_number} with parameters: {all_args_repr}.'
        
        # Check for duplicates before appending
        if log_entry not in log_entries:
            log_entries.append(log_entry)

        # Optionally also log to a file
        logging.info(log_entry)
        
        return func(*args, **kwargs)
    return wrapper

def display_logs():
    if not log_entries:
        print("No logs available.")
    else:
        for entry in log_entries:
            print(entry)







import pandas as pd
import copy

# Global variables to track DataFrame and its history
_df = None
_history = []

def save_state(df):
    """
    Save the current state of the DataFrame before modification.
    """
    global _history
    _history.append(copy.deepcopy(df))

@log_function_call
def undo():
    """
    Undo the most recent change to the DataFrame.
    """
    global _df, _history
    if _history:
        _df = _history.pop()
    else:
        print("No recent changes to undo!")
    return _df

# Decorator to track changes and save the previous state
def track_changes(func):
    @functools.wraps(func)  # Add this line
    def wrapper(df, *args, **kwargs):
        save_state(df)  # Save the current state of the DataFrame
        result = func(df, *args, **kwargs)
        global _df
        _df = result
        return result
    return wrapper






import pandas as pd
import numpy as np
import json
from typing import Optional, Union, List

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
) -> Optional[Union[pd.DataFrame, List, dict, str]]:
    """
    Handle null values in a DataFrame by removing, replacing, imputing, or dropping rows/columns
    with more than x% missing values.

    Parameters:
    - df (Union[pd.DataFrame, List, dict, tuple, np.ndarray, str]): Input data.
    - columns (Optional[Union[str, List[str]]]): Column(s) to check for null values.
      If None, apply to the entire data. Default is None.
    - action (str): Action to perform on rows with null values.
      Options are 'remove' to drop rows, 'replace' to fill nulls with a custom value,
      or 'impute' to fill nulls using a strategy like 'mean', 'median', etc. Default is 'remove'.
    - with_val (Optional[Union[int, float, str]]): Custom value to replace nulls with, applicable if action is 'replace'.
    - by (Optional[str]): Strategy to use for imputing nulls ('mean', 'median', 'mode', etc.).
      Required if action is 'impute'.
    - inplace (bool): If True, modify the DataFrame in place. If False, return a new DataFrame. Default is False.
    - threshold (Optional[float]): Percentage of missing values (0-100) allowed. If exceeded, drop the row/column.
    - axis (str): Specify whether to apply the threshold to 'rows' or 'columns'. Default is 'rows'.

    Returns:
    - Optional[Union[pd.DataFrame, List, dict, str]]: Modified data or None if inplace=True.
    """

    # Convert non-Pandas inputs to DataFrame
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
            parsed_data = json.loads(df)  # Parse JSON if possible
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

    # Validate axis
    if axis not in ['rows', 'columns']:
        raise ValueError("The 'axis' parameter must be either 'rows' or 'columns'.")

    # Validate threshold
    if threshold is not None:
        if not (0 <= threshold <= 100):
            raise ValueError("The 'threshold' parameter must be between 0 and 100.")

    # Handle columns parameter
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        subset_df = df[columns]
    else:
        subset_df = df

    # Drop rows/columns based on threshold
    if threshold is not None:
        if axis == 'rows':
            df_cleaned = df.dropna(thresh=int((1 - threshold / 100) * df.shape[1]), axis=0)
        elif axis == 'columns':
            df_cleaned = df.dropna(thresh=int((1 - threshold / 100) * df.shape[0]), axis=1)
    else:
        # Process according to action (remove, replace, impute)
        if action == 'remove':
            df_cleaned = df.dropna(subset=subset_df.columns)
        elif action == 'replace':
            if with_val is None:
                raise ValueError("A value must be provided when action is 'replace'.")
            df_cleaned = df.fillna({col: with_val for col in subset_df.columns})
        elif action == 'impute':
            strategies = {
                'mean': lambda col: col.fillna(col.mean()),
                'median': lambda col: col.fillna(col.median()),
                'mode': lambda col: col.fillna(col.mode().iloc[0] if not col.mode().empty else col),
                'interpolate': lambda col: col.interpolate(),
                'forward_fill': lambda col: col.ffill(),
                'backward_fill': lambda col: col.bfill()
            }
            if by not in strategies:
                raise ValueError("Invalid impute strategy. Use one of ['mean', 'median', 'mode', 'interpolate', 'forward_fill', 'backward_fill'].")
            df_cleaned = df.copy()
            for col in subset_df.columns:
                df_cleaned[col] = strategies[by](df_cleaned[col])
        else:
            raise ValueError("Action must be either 'remove', 'replace', or 'impute'.")

    # Convert back to original format if needed
    if original_format == list:
        return df_cleaned.values.tolist()
    elif original_format == tuple:
        return tuple(map(tuple, df_cleaned.values))
    elif original_format == 'dict':
        return df_cleaned.to_dict(orient="list")
    elif original_format == 'json':
        return df_cleaned.to_json()
    elif inplace:
        df.update(df_cleaned)
        return None
    else:
        return df_cleaned
    




import pandas as pd
import numpy as np
import json
from typing import Optional, Union, List

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

    # Convert back to original format if needed
    if original_format == list:
        return df_cleaned.values.tolist()
    elif original_format == tuple:
        return tuple(map(tuple, df_cleaned.values))
    elif original_format == 'dict':
        return df_cleaned.to_dict(orient="list")
    elif original_format == 'json':
        return df_cleaned.to_json()
    elif inplace:
        df.update(df_cleaned)
        return None
    else:
        return df_cleaned





import pandas as pd
import numpy as np
import re
import json
from spellchecker import SpellChecker
from typing import Union, Optional

def clean_text(text: str) -> str:
    """
    Cleans text by removing special characters and extra spaces.

    Parameters:
    - text (str): The original text to clean.

    Returns:
    - str: The cleaned text.
    """
    if text is None:
        return None
    clean_text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return ' '.join(clean_text.split()).strip()


def clean_column_name(column_name: str) -> str:
    """
    Cleans and recommends a new column name by correcting spelling and removing special characters.

    Parameters:
    - column_name (str): The original column name.

    Returns:
    - str: The recommended clean column name.
    """
    if column_name is None:
        return None

    spell = SpellChecker()
    clean_text = re.sub(r'[^a-zA-Z0-9 ]', ' ', column_name)
    words = clean_text.split()
    corrected_words = [spell.correction(word) or word for word in words]
    cleaned_column_name = ' '.join(corrected_words).title()

    return cleaned_column_name


def clean_row_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the rows of a DataFrame by removing special characters and extra spaces.

    Parameters:
    - df (pd.DataFrame): The DataFrame to clean.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    return df.applymap(lambda x: clean_text(str(x)) if isinstance(x, str) else x)


def rename_columns(df: pd.DataFrame) -> dict:
    """
    Automatically recommend readable column names for a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame with columns to be analyzed.

    Returns:
    - dict: A dictionary where keys are current column names and values are recommended new names.
    """
    recommendations = {col: clean_column_name(col) for col in df.columns if clean_column_name(col) != col}
    return recommendations


def apply_column_renames(df: pd.DataFrame, rename_map: dict) -> None:
    """
    Apply the recommended column name changes to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to rename columns in.
    - rename_map (dict): The dictionary of column renaming recommendations.
    """
    df.rename(columns=rename_map, inplace=True)


def convert_to_dataframe(data: Union[pd.DataFrame, list, dict, tuple, np.ndarray, str]) -> pd.DataFrame:
    """
    Converts supported data types to a Pandas DataFrame.

    Parameters:
    - data (Union[pd.DataFrame, list, dict, tuple, np.ndarray, str]): Input data.

    Returns:
    - pd.DataFrame: Converted DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, (list, tuple, np.ndarray)):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame.from_dict(data)
    elif isinstance(data, str):
        try:
            parsed_data = json.loads(data)
            if isinstance(parsed_data, dict):
                return pd.DataFrame.from_dict(parsed_data)
        except json.JSONDecodeError:
            raise ValueError("String inputs must be valid JSON objects.")
    raise TypeError("Unsupported data type. Provide a DataFrame, list, tuple, dict, NumPy array, or JSON string.")

# Refined function
@log_function_call
@track_changes
def refine(
    df: Union[pd.DataFrame, list, dict, tuple, np.ndarray, str],
    clean_rows: bool = True
) -> Union[pd.DataFrame, list, dict, tuple, str]:
    """
    Refines the DataFrame by cleaning both column names and optionally row data.

    Parameters:
    - df (Union[pd.DataFrame, list, dict, tuple, np.ndarray, str]): The DataFrame or data to refine.
    - clean_rows (bool): Whether to clean the row data as well as the column names. Default is True.

    Returns:
    - Union[pd.DataFrame, list, dict, tuple, str]: Refined data in the same format as input.
    """
    # Convert input to DataFrame if necessary
    original_format = type(df)
    df_converted = convert_to_dataframe(df)

    # Get column rename recommendations
    rename_recommendations = rename_columns(df_converted)

    if rename_recommendations:
        print("\nRecommended Column Renames:")
        for original, recommended in rename_recommendations.items():
            print(f"Original: {original}, Recommended: {recommended}")
        apply_changes = input("\nDo you want to apply these column name changes? (yes/no): ").strip().lower()
        if apply_changes == 'yes':
            apply_column_renames(df_converted, rename_recommendations)
            print("\nRenamed DataFrame Column Names:")
            print(df_converted.columns)

    # Clean rows if specified
    if clean_rows:
        clean_row_data_prompt = input("\nDo you want to clean row data (remove special characters and extra spaces)? (yes/no): ").strip().lower()
        if clean_row_data_prompt == 'yes':
            df_converted = clean_row_data(df_converted)
            print("\nRow data has been cleaned.")

    # Convert back to original format
    if original_format == pd.DataFrame:
        return df_converted
    elif original_format == list:
        return df_converted.values.tolist()
    elif original_format == tuple:
        return tuple(map(tuple, df_converted.values))
    elif original_format == dict:
        return df_converted.to_dict(orient="list")
    elif original_format == str:
        return df_converted.to_json()
    else:
        return df_converted





import pandas as pd
import json
from typing import Union, Dict

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

    # Convert input to DataFrame if necessary
    original_format = type(df)

    if isinstance(df, pd.DataFrame):
        df_converted = df
    elif isinstance(df, (list, tuple, np.ndarray)):
        df_converted = pd.DataFrame(df)
    elif isinstance(df, dict):
        df_converted = pd.DataFrame.from_dict(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                df_converted = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, list, tuple, dict, or JSON string.")

    # Check for invalid rename_dict
    if not isinstance(rename_dict, dict) or not rename_dict:
        raise ValueError("rename_dict must be a non-empty dictionary.")

    # Check for columns that are not in the DataFrame
    missing_columns = [col for col in rename_dict if col not in df_converted.columns]
    if missing_columns:
        raise KeyError(f"These columns do not exist in the DataFrame: {missing_columns}")

    # Rename columns using the provided dictionary
    renamed_df = df_converted.rename(columns=rename_dict)

    # Convert back to the original format if necessary
    if original_format == pd.DataFrame:
        return renamed_df
    elif original_format == list:
        return renamed_df.values.tolist()
    elif original_format == tuple:
        return tuple(map(tuple, renamed_df.values))
    elif original_format == dict:
        return renamed_df.to_dict(orient="list")
    elif original_format == str:
        return renamed_df.to_json()
    else:
        return renamed_df
    





import pandas as pd
import pytz
import json
from typing import Union, List, Optional

@log_function_call
@track_changes
def format_dt(
    df: Union[pd.DataFrame, list, dict, tuple, np.ndarray, str],
    columns: Union[str, List[str]],
    day: bool = False,
    month: bool = False,
    year: bool = False,
    quarter: bool = False,
    hour: bool = False,
    minute: bool = False,
    day_of_week: bool = False,
    date_format: str = "%Y-%m-%d",
    time_format: str = "%H:%M:%S",
    from_timezone: Optional[str] = None,
    to_timezone: Optional[str] = None
) -> Union[pd.DataFrame, list, dict, tuple, str]:
    """
    Add additional date/time-based columns and format date/time columns in a DataFrame.

    Parameters:
    - df (Union[pd.DataFrame, list, dict, tuple, np.ndarray, str]): The data (could be a DataFrame, list, dict, tuple, etc.) to which new date/time features will be added and formatted.
    - columns (str or List[str]): The name(s) of the column(s) containing date/time data.
    - day (bool): If True, add a column with the day of the month.
    - month (bool): If True, add a column with the month.
    - year (bool): If True, add a column with the year.
    - quarter (bool): If True, add a column with the quarter of the year.
    - hour (bool): If True, add a column with the hour of the day.
    - minute (bool): If True, add a column with the minute of the hour.
    - day_of_week (bool): If True, add a column with the day of the week as a string (e.g., 'Monday').
    - date_format (str): The desired date format (default: "%Y-%m-%d").
    - time_format (str): The desired time format (default: "%H:%M:%S").
    - from_timezone (Optional[str]): The original timezone of the datetime column(s).
      If None, no timezone conversion will be applied (default: None).
    - to_timezone (Optional[str]): The desired timezone for the datetime column(s).
      If None, no timezone conversion will be applied (default: None).

    Returns:
    - Union[pd.DataFrame, list, dict, tuple, str]: The data (same type as input) with added date/time features and formatted date/time columns.

    Raises:
    - ValueError: If the specified column does not exist in the DataFrame or conversion fails.
    """

    # Convert input to DataFrame if necessary
    original_format = type(df)

    if isinstance(df, pd.DataFrame):
        df_converted = df
    elif isinstance(df, (list, tuple, np.ndarray)):
        df_converted = pd.DataFrame(df)
    elif isinstance(df, dict):
        df_converted = pd.DataFrame.from_dict(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                df_converted = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, list, tuple, dict, or JSON string.")

    # Ensure columns is a list to handle both single and multiple columns
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        # Check if the DataFrame contains the specified column
        if column not in df_converted.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        # Convert the column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_converted[column]):
            try:
                df_converted[column] = pd.to_datetime(df_converted[column])
            except Exception as e:
                raise ValueError(f"Failed to convert column '{column}' to datetime. Error: {e}")

        # Adding requested datetime features
        if day:
            df_converted[f'{column}_day'] = df_converted[column].dt.day
        if month:
            df_converted[f'{column}_month'] = df_converted[column].dt.month
        if year:
            df_converted[f'{column}_year'] = df_converted[column].dt.year
        if quarter:
            df_converted[f'{column}_quarter'] = df_converted[column].dt.quarter
        if hour:
            df_converted[f'{column}_hour'] = df_converted[column].dt.hour
        if minute:
            df_converted[f'{column}_minute'] = df_converted[column].dt.minute
        if day_of_week:
            df_converted[f'{column}_day_of_week'] = df_converted[column].dt.day_name()

        # Apply date and time format and timezone conversion
        if from_timezone and to_timezone:
            # Convert timezone if both from and to timezones are specified
            df_converted[column] = (
                df_converted[column]
                .dt.tz_localize(from_timezone, ambiguous='NaT', nonexistent='NaT')
                .dt.tz_convert(to_timezone)
            )
        elif from_timezone or to_timezone:
            raise ValueError("Both from_timezone and to_timezone must be specified for timezone conversion.")

        # Apply date and time format
        df_converted[column] = df_converted[column].dt.strftime(f"{date_format} {time_format}")

    # Convert back to the original format if necessary
    if original_format == pd.DataFrame:
        return df_converted
    elif original_format == list:
        return df_converted.values.tolist()
    elif original_format == tuple:
        return tuple(map(tuple, df_converted.values))
    elif original_format == dict:
        return df_converted.to_dict(orient="list")
    elif original_format == str:
        return df_converted.to_json()
    else:
        return df_converted
    





import pandas as pd
from collections import Counter
import json

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
    - df (Union[pd.DataFrame, list, dict, tuple, np.ndarray, str]): The data (could be a DataFrame, list, dict, tuple, etc.) containing the column to split.
    - column (str): The name of the column to be split.
    - delimiter (str): The delimiter to use for splitting. If None, detect the most common delimiter.

    Returns:
    - Union[pd.DataFrame, list, dict, tuple, str]: A new data structure with the specified column split into multiple columns.

    Raises:
    - ValueError: If the column does not exist, no delimiter is detected, or an invalid delimiter is provided.
    """
    # Convert input to DataFrame if necessary
    original_format = type(df)

    if isinstance(df, pd.DataFrame):
        df_converted = df
    elif isinstance(df, (list, tuple, np.ndarray)):
        df_converted = pd.DataFrame(df)
    elif isinstance(df, dict):
        df_converted = pd.DataFrame.from_dict(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                df_converted = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, list, tuple, dict, or JSON string.")

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

    # Convert back to the original format if necessary
    if original_format == pd.DataFrame:
        return df_expanded
    elif original_format == list:
        return df_expanded.values.tolist()
    elif original_format == tuple:
        return tuple(map(tuple, df_expanded.values))
    elif original_format == dict:
        return df_expanded.to_dict(orient="list")
    elif original_format == str:
        return df_expanded.to_json()
    else:
        return df_expanded







import pandas as pd
from spellchecker import SpellChecker
import re
import json
import numpy as np

# Initialize SpellCheckers for different dictionaries
spell_checker_dict = {
    'en_US': SpellChecker(language='en'),
    # You can add more dictionaries if needed
}

def spell_check_dataframe(df, dictionary='en_US', columns=None):
    """
    Perform spell check on the specified columns of a DataFrame using the specified dictionary.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns

    if dictionary not in spell_checker_dict:
        raise ValueError("Dictionary must be one of 'en_US'")

    spell_checker = spell_checker_dict[dictionary]
    misspelled_words = {}

    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column {column} is not in the DataFrame.")

        text_data = df[column].dropna()
        misspelled = []
        for text in text_data:
            words = re.findall(r'\b\w+\b', text)  # Tokenize words
            words = [word.lower() for word in words if not word.isdigit()]  # Exclude numeric values
            misspelled.extend([word for word in words if word not in spell_checker])

        misspelled_words[column] = list(set(misspelled))  # Remove duplicates

    return misspelled_words

def detect_invalid_dates(series, date_format=None):
    """
    Detect invalid date values in a pandas Series.
    """
    if date_format:
        return pd.to_datetime(series, format=date_format, errors='coerce').isna()
    else:
        return pd.to_datetime(series, errors='coerce').isna()
    
@log_function_call
@track_changes

def detect_errors(
    df: Union[pd.DataFrame, list, dict, tuple, np.ndarray, str],
    date_columns=None,
    numeric_columns=None,
    text_columns=None,
    date_format=None
):
    """
    Detect and flag data entry errors in a DataFrame, including invalid dates and misspelled words.

    Parameters:
    - df (Union[pd.DataFrame, list, dict, tuple, np.ndarray, str]): The data to process.
    - date_columns (list): List of columns to check for invalid dates.
    - numeric_columns (list): List of columns to check for invalid numeric formats.
    - text_columns (list): List of columns to spell check.
    - date_format (str): Optional date format for validation.

    Returns:
    - pd.DataFrame: A DataFrame listing detected errors.
    """
    # Convert input to DataFrame if necessary
    original_format = type(df)

    if isinstance(df, pd.DataFrame):
        df_converted = df
    elif isinstance(df, (list, tuple, np.ndarray)):
        df_converted = pd.DataFrame(df)
    elif isinstance(df, dict):
        df_converted = pd.DataFrame.from_dict(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                df_converted = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, list, tuple, dict, or JSON string.")

    errors = []

    # Default to American English dictionary
    spellcheck_dict = 'en_US'

    # Check for invalid dates if date_columns is provided
    if date_columns:
        for col in date_columns:
            if col in df_converted.columns:
                invalid_dates = detect_invalid_dates(df_converted[col], date_format=date_format)
                for idx in invalid_dates[invalid_dates].index:
                    errors.append({'Column': col, 'Error Type': 'Invalid Date', 'Value': df_converted.at[idx, col]})

    # Check for invalid numeric formats if numeric_columns is provided
    if numeric_columns:
        for col in numeric_columns:
            if col in df_converted.columns:
                non_numeric = pd.to_numeric(df_converted[col], errors='coerce').isna() & df_converted[col].notna()
                for idx in non_numeric[non_numeric].index:
                    errors.append({'Column': col, 'Error Type': 'Invalid Numeric', 'Value': df_converted.at[idx, col]})

    # Spell check on text columns if text_columns is provided
    if text_columns:
        misspelled_words = spell_check_dataframe(df_converted, dictionary=spellcheck_dict, columns=text_columns)
        for col, words in misspelled_words.items():
            for word in words:
                indices = df_converted[col].apply(lambda x: word in x if isinstance(x, str) else False)
                for idx in indices[indices].index:
                    errors.append({'Column': col, 'Error Type': 'Misspelled Word', 'Value': word})

    # Convert result to a DataFrame
    errors_df = pd.DataFrame(errors)

    # Return the errors in original format if necessary
    if original_format == pd.DataFrame:
        return errors_df
    elif original_format == list:
        return errors_df.to_dict('records')
    elif original_format == tuple:
        return tuple(errors_df.to_dict('records'))
    elif original_format == dict:
        return errors_df.to_dict(orient='list')
    elif original_format == str:
        return errors_df.to_json()
    else:
        return errors_df






import pandas as pd
import numpy as np
import json
from typing import Union, Optional, List

@log_function_call
@track_changes
def convert_type(
    df: Union[pd.DataFrame, pd.Series, list, dict, tuple, np.ndarray, str],
    columns: Optional[Union[str, List[str]]] = None
) -> Union[pd.DataFrame, pd.Series]:
    """
    Recommend and apply data type conversions for a given DataFrame, Series, or other supported formats.

    Parameters:
    - df (Union[pd.DataFrame, pd.Series, list, dict, tuple, np.ndarray, str]): The input data to analyze.
    - columns (str, list of str, or None): The specific column(s) to analyze. If None, the function analyzes all columns.

    Returns:
    - pd.DataFrame or pd.Series: The data with applied type conversions.
    """
    # Convert input to DataFrame or Series
    original_format = type(df)

    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        df_converted = df
    elif isinstance(df, (list, tuple, np.ndarray)):
        df_converted = pd.DataFrame(df)
    elif isinstance(df, dict):
        df_converted = pd.DataFrame.from_dict(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                df_converted = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, Series, list, dict, tuple, or JSON string.")

    # If a single column is provided as a string, convert it to a list
    if isinstance(columns, str):
        columns = [columns]

    # If columns are provided, ensure they exist in the DataFrame
    if columns:
        for column in columns:
            if column not in df_converted.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        data_to_analyze = df_converted[columns]
    else:
        data_to_analyze = df_converted

    recommendations = {}

    # Helper function to suggest data type conversions
    def suggest_conversion(col):
        suggestions = []

        # Check if the column can be converted to numeric
        if pd.api.types.is_object_dtype(col):
            try:
                pd.to_numeric(col, errors='raise')
                if pd.api.types.is_float_dtype(col):
                    # Check if there are any decimal places
                    if col.dropna().apply(lambda x: x % 1 != 0).sum() == 0:
                        suggestions.append('Convert to integer')
                else:
                    suggestions.append('Convert to numeric')
            except ValueError:
                pass

        # Check if the column can be converted to category
        if pd.api.types.is_object_dtype(col) and col.nunique() < 10:
            suggestions.append('Convert to category')

        # Check if the column can be converted to datetime
        if pd.api.types.is_object_dtype(col):
            try:
                pd.to_datetime(col, errors='raise')
                suggestions.append('Convert to datetime')
            except ValueError:
                pass

        # Check if the column is already a boolean type
        if pd.api.types.is_bool_dtype(col):
            suggestions.append('Column is already boolean')

        return suggestions

    # Analyze each column
    for col_name, col_data in data_to_analyze.items():
        current_dtype = col_data.dtype
        suggestions = suggest_conversion(col_data)

        if suggestions:
            recommendations[col_name] = {
                'current_dtype': current_dtype,
                'suggestions': suggestions
            }

    # Display recommendations and apply user-selected conversions
    for col_name, rec in recommendations.items():
        print(f"\nColumn: {col_name}")
        print(f"Current Data Type: {rec['current_dtype']}")
        print(f"Recommended Conversions: {', '.join(rec['suggestions'])}")

        # Apply suggested conversions based on user confirmation
        for suggestion in rec['suggestions']:
            # Auto-confirmation for this example; replace with interactive input in production if needed.
            user_input = "yes"  # Simulate user input for testing
            if user_input.lower() == 'yes':
                if suggestion == 'Convert to integer':
                    df_converted[col_name] = pd.to_numeric(df_converted[col_name], errors='coerce').fillna(0).astype(int)
                elif suggestion == 'Convert to numeric':
                    df_converted[col_name] = pd.to_numeric(df_converted[col_name], errors='coerce')
                elif suggestion == 'Convert to category':
                    df_converted[col_name] = df_converted[col_name].astype('category')
                elif suggestion == 'Convert to datetime':
                    df_converted[col_name] = pd.to_datetime(df_converted[col_name], errors='coerce')
                print(f"Column '{col_name}' converted to {df_converted[col_name].dtype}.")
            else:
                print(f"Column '{col_name}' not converted.")

    # Convert back to the original format if necessary
    if original_format == list:
        return df_converted.to_dict(orient='records')
    elif original_format == tuple:
        return tuple(df_converted.to_dict(orient='records'))
    elif original_format == dict:
        return df_converted.to_dict(orient='list')
    elif original_format == str:
        return df_converted.to_json()
    else:
        return df_converted






import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import json
from typing import Union, List

@log_function_call
@track_changes


def detect_outliers(
    df: Union[pd.DataFrame, pd.Series, list, dict, tuple, np.ndarray, str],
    columns: Union[str, List[str]],
    method: str = 'iqr',
    threshold: float = 1.5,
    handle_missing: bool = True,
    anomaly_method: Optional[str] = None,
    contamination: float = 0.05,
    n_neighbors: int = 20,
    eps: float = 0.5,
    min_samples: int = 5
) -> pd.DataFrame:
    """
    Detects outliers or anomalies in specified columns using IQR, Z-Score, or anomaly detection methods.

    Parameters:
    - df (Union[pd.DataFrame, pd.Series, list, dict, tuple, np.ndarray, str]): Input data in various formats.
    - columns (Union[str, List[str]]): Column name(s) to detect outliers/anomalies.
    - method (str): Method for outlier detection ('iqr' or 'z-score'). Default is 'iqr'.
    - threshold (float): Threshold for detection (1.5 for IQR, 3 for Z-Score). Default is 1.5.
    - handle_missing (bool): Remove rows with missing data if True. Default is True.
    - anomaly_method (Optional[str]): 'isolation_forest', 'lof', or 'dbscan' for anomaly detection.
    - contamination (float): Proportion of anomalies for 'isolation_forest' and 'lof'. Default is 0.05.
    - n_neighbors (int): Number of neighbors for LOF. Default is 20.
    - eps (float): Maximum distance for DBSCAN. Default is 0.5.
    - min_samples (int): Minimum samples for DBSCAN. Default is 5.

    Returns:
    - pd.DataFrame: A DataFrame with outliers removed.
    """
    # Convert input to DataFrame
    original_format = type(df)
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        df_cleaned = df
    elif isinstance(df, (list, tuple, np.ndarray)):
        df_cleaned = pd.DataFrame(df)
    elif isinstance(df, dict):
        df_cleaned = pd.DataFrame.from_dict(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                df_cleaned = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, Series, list, dict, tuple, or JSON string.")

    # Ensure `columns` is a list for uniform processing
    if isinstance(columns, str):
        columns = [columns]

    # Check for column existence
    for column in columns:
        if column not in df_cleaned.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # Validate method or anomaly detection choice
    if method not in ['iqr', 'z-score'] and anomaly_method not in ['isolation_forest', 'lof', 'dbscan', None]:
        raise ValueError("Invalid method or anomaly detection method.")

    # Handle missing values
    original_shape = df_cleaned.shape
    if handle_missing:
        df_cleaned = df_cleaned.dropna(subset=columns)

    # Initialize mask for combined outliers
    combined_outliers = np.zeros(df_cleaned.shape[0], dtype=bool)

    # Outlier detection
    for column in columns:
        if anomaly_method is None:
            outlier_type = "outliers"
            if method == 'iqr':
                Q1 = df_cleaned[column].quantile(0.25)
                Q3 = df_cleaned[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)
            elif method == 'z-score':
                mean = df_cleaned[column].mean()
                std = df_cleaned[column].std()
                z_scores = (df_cleaned[column] - mean) / std
                outliers = np.abs(z_scores) > threshold
        else:
            outlier_type = "anomalies"
            if anomaly_method == 'isolation_forest':
                model = IsolationForest(contamination=contamination, random_state=42)
                outliers = model.fit_predict(df_cleaned[[column]]) == -1
            elif anomaly_method == 'lof':
                lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                outliers = lof_model.fit_predict(df_cleaned[[column]]) == -1
            elif anomaly_method == 'dbscan':
                dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
                outliers = dbscan_model.fit_predict(df_cleaned[[column]]) == -1
            else:
                raise ValueError(f"Anomaly method '{anomaly_method}' is not supported.")

        combined_outliers |= outliers
        print(f"Total {outlier_type} detected in column '{column}': {np.sum(outliers)}")

    # Remove detected outliers
    df_cleaned = df_cleaned[~combined_outliers]

    # Print original and new data shapes
    print(f"Original shape: {original_shape}, After removing {outlier_type}: {df_cleaned.shape}")

    # Convert back to the original format if applicable
    if original_format == list:
        return df_cleaned.to_dict(orient='records')
    elif original_format == tuple:
        return tuple(df_cleaned.to_dict(orient='records'))
    elif original_format == dict:
        return df_cleaned.to_dict(orient='list')
    elif original_format == str:
        return df_cleaned.to_json()
    else:
        return df_cleaned
    








import re
import pandas as pd
from typing import Union, List, Optional
import json

@log_function_call
@track_changes

def remove_chars(
    df: Union[pd.DataFrame, pd.Series, list, dict, tuple, str],
    columns: Optional[Union[str, List[str]]] = None,
    strip_all: bool = False,
    custom_characters: Optional[str] = None
) -> Union[pd.DataFrame, pd.Series, list, dict, tuple, str]:
    """
    Cleans specified columns in a DataFrame (or equivalent data structure) by trimming spaces and optionally removing custom characters.

    Parameters:
    - df (Union[pd.DataFrame, pd.Series, list, dict, tuple, str]): The input data in various formats.
    - columns (str or list of str or None): Column name(s) to clean. If None, applies to all string columns in DataFrame or Series.
    - strip_all (bool): If True, removes all spaces; otherwise, trims leading/trailing spaces and reduces multiple spaces to one.
    - custom_characters (Optional[str]): A string of characters to be removed from text. If None, no additional characters are removed.

    Returns:
    - Union[pd.DataFrame, pd.Series, list, dict, tuple, str]: The cleaned data in its original format.
    """
    def clean_text(text):
        if isinstance(text, str):
            # Trim leading and trailing spaces
            trimmed_text = text.strip()

            if strip_all:
                # Remove all spaces
                cleaned_text = re.sub(r'\s+', '', trimmed_text)
            else:
                # Replace multiple spaces with a single space
                cleaned_text = re.sub(r'\s+', ' ', trimmed_text)

            if custom_characters:
                # Remove custom characters
                cleaned_text = re.sub(f"[{re.escape(custom_characters)}]", '', cleaned_text)

            return cleaned_text
        else:
            return text  # Return unchanged if not a string

    # Handle input formats
    original_format = type(df)
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        df_cleaned = df
    elif isinstance(df, (list, tuple)):
        df_cleaned = pd.DataFrame(df)
    elif isinstance(df, dict):
        df_cleaned = pd.DataFrame.from_dict(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                df_cleaned = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, Series, list, dict, tuple, or JSON string.")

    # Determine columns to process
    if columns is None:
        if isinstance(df_cleaned, pd.DataFrame):
            # Apply to all string columns if columns are not specified
            columns = df_cleaned.select_dtypes(include=['object']).columns.tolist()
        else:
            raise ValueError("For non-DataFrame inputs, 'columns' must be specified.")
    elif isinstance(columns, str):
        columns = [columns]

    # Clean specified columns
    for col in columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].map(clean_text)
        else:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    # Convert back to original format if applicable
    if original_format == list:
        return df_cleaned.to_dict(orient='records')
    elif original_format == tuple:
        return tuple(df_cleaned.to_dict(orient='records'))
    elif original_format == dict:
        return df_cleaned.to_dict(orient='list')
    elif original_format == str:
        return df_cleaned.to_json()
    else:
        return df_cleaned
    







import pandas as pd
import numpy as np
import json
from typing import Union, List, Optional

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

    # Convert to DataFrame for uniform processing if the input is not already a DataFrame or Series
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        data_cleaned = df
    elif isinstance(df, (list, tuple)):
        data_cleaned = pd.DataFrame(df)
    elif isinstance(df, dict):
        data_cleaned = pd.DataFrame.from_dict(df)
    elif isinstance(df, np.ndarray):
        data_cleaned = pd.DataFrame(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                data_cleaned = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must represent a JSON object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError(f"Unsupported data type: {original_format}. Please provide a DataFrame, Series, list, dict, tuple, numpy array, or JSON string.")

    # Check if the target and reference columns exist in the DataFrame
    if target_column not in data_cleaned.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")
    if reference_column not in data_cleaned.columns:
        raise ValueError(f"Column '{reference_column}' does not exist in the DataFrame.")

    # Get the data type of the reference column
    ref_dtype = data_cleaned[reference_column].dtype

    # Check and apply formatting based on data type
    if pd.api.types.is_datetime64_any_dtype(ref_dtype):
        # If the reference column is datetime, convert the target column to datetime
        try:
            data_cleaned[target_column] = pd.to_datetime(data_cleaned[target_column], errors='coerce')
        except Exception as e:
            raise TypeError(f"Error converting '{target_column}' to datetime: {e}")
    elif pd.api.types.is_numeric_dtype(ref_dtype):
        # If the reference column is numeric, convert the target column to numeric
        try:
            data_cleaned[target_column] = pd.to_numeric(data_cleaned[target_column], errors='coerce')
        except Exception as e:
            raise TypeError(f"Error converting '{target_column}' to numeric: {e}")
    elif pd.api.types.is_string_dtype(ref_dtype):
        # If the reference column is string, apply string formatting based on the reference column's format
        ref_sample = data_cleaned[reference_column].dropna().astype(str).iloc[0]

        if ref_sample.isupper():
            # If reference column is uppercase, convert target column to uppercase
            data_cleaned[target_column] = data_cleaned[target_column].astype(str).str.upper()
        elif ref_sample.islower():
            # If reference column is lowercase, convert target column to lowercase
            data_cleaned[target_column] = data_cleaned[target_column].astype(str).str.lower()
        elif ref_sample.istitle():
            # If reference column is title case, convert target column to title case
            data_cleaned[target_column] = data_cleaned[target_column].astype(str).str.title()
        else:
            # For other string formats, simply convert to string
            data_cleaned[target_column] = data_cleaned[target_column].astype(str)
    else:
        # For unsupported data types, raise an error
        raise TypeError(f"Data type '{ref_dtype}' of reference column is not supported for formatting.")

    # Return the cleaned data in its original format
    if original_format == list:
        return data_cleaned.to_dict(orient='records')
    elif original_format == tuple:
        return tuple(data_cleaned.to_dict(orient='records'))
    elif original_format == dict:
        return data_cleaned.to_dict(orient='list')
    elif original_format == str:
        return data_cleaned.to_json()
    elif original_format == np.ndarray:
        return data_cleaned.to_numpy()
    else:
        return data_cleaned









import pandas as pd
import numpy as np
import json
from typing import Union, List, Optional

@log_function_call
@track_changes
def scale_data(
    df: Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray],
    columns: Union[str, List[str]],
    method: str = 'min-max'
) -> Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]:
    """
    Scales the specified column(s) in the DataFrame (or equivalent) using the given scaling method.

    Parameters:
    - df (Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]): The input data.
    - columns (str or list of str): The column name(s) to scale.
    - method (str): The scaling method to use. Options are 'min-max', 'robust', and 'standard'.

    Returns:
    - Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]: The scaled data.

    Raises:
    - ValueError: If the scaling method is not recognized.
    """

    # Detect original data format
    original_format = type(df)

    # Convert to DataFrame for uniform processing if input is not already a DataFrame or Series
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        data_cleaned = df
    elif isinstance(df, (list, tuple)):
        data_cleaned = pd.DataFrame(df)
    elif isinstance(df, dict):
        data_cleaned = pd.DataFrame.from_dict(df)
    elif isinstance(df, np.ndarray):
        data_cleaned = pd.DataFrame(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                data_cleaned = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must represent a JSON object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError(f"Unsupported data type: {original_format}. Please provide a DataFrame, Series, list, dict, tuple, numpy array, or JSON string.")

    # If a single column name is provided as a string, convert it to a list
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column not in data_cleaned.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        if method == 'min-max':
            # Min-Max Scaling
            X = data_cleaned[column]
            X_min = X.min()
            X_max = X.max()
            data_cleaned[column] = (X - X_min) / (X_max - X_min)

        elif method == 'robust':
            # Robust Scaling
            X = data_cleaned[column]
            median = X.median()
            IQR = X.quantile(0.75) - X.quantile(0.25)
            data_cleaned[column] = (X - median) / IQR

        elif method == 'standard':
            # Standard Scaling
            X = data_cleaned[column]
            mean = X.mean()
            std = X.std()
            data_cleaned[column] = (X - mean) / std

        else:
            raise ValueError(f"Scaling method '{method}' is not recognized. Choose from 'min-max', 'robust', or 'standard'.")

    # Return the cleaned data in its original format
    if original_format == list:
        return data_cleaned.to_dict(orient='records')
    elif original_format == tuple:
        return tuple(data_cleaned.to_dict(orient='records'))
    elif original_format == dict:
        return data_cleaned.to_dict(orient='list')
    elif original_format == str:
        return data_cleaned.to_json()
    elif original_format == np.ndarray:
        return data_cleaned.to_numpy()
    else:
        return data_cleaned








import pkg_resources
import pandas as pd

@log_function_call
def sample_data():
    """
    Load the Customer Sales Data dataset.

    This dataset includes simulated customer sales records with the following columns:
    - CustomerID: Unique identifier for each customer.
    - Age: Age of the customer (may include missing values and inconsistent formats).
    - Gender: Gender of the customer (may include missing values).
    - PurchaseHistory: List of previous purchases.
    - ProductCategory: Category of the purchased product.
    - PurchaseDate: Date of the purchase (may include inconsistent formats).
    - AmountSpent: Total amount spent on the purchase (includes outliers).
    - PaymentMethod: Method of payment used (includes mixed data types).
    - Country: Country of the customer.
    - MembershipStatus: Membership status (may include missing values).
    - PhoneNumber: Phone number of the customer (includes various formats).
    - DiscountCode: Discount code applied (includes duplicates).

    The dataset is stored in a CSV file located in the 'data' folder within the package.

    Returns:
        pandas.DataFrame: A DataFrame containing the Customer Sales Data.
    """
    # Use pkg_resources to access the file within the package
    data_path = pkg_resources.resource_filename('pyspan', 'data/customer_sales_data.csv')

    # Load the dataset using pandas
    df = pd.read_csv(data_path)
    return df







import pandas as pd
import numpy as np
import json
from typing import Union, List, Optional

# Expanded unit conversion mappings (same as provided)
default_unit_conversion_factors = {
    'length': {
        'mm': 0.001,
        'cm': 0.01,
        'm': 1,
        'km': 1000,
        'in': 0.0254,
        'ft': 0.3048,
        'yd': 0.9144,
        'mi': 1609.34,
        'nmi': 1852,  # Nautical Mile
        'fathom': 1.8288  # Fathom
    },
    'mass': {
        'mg': 1e-6,
        'g': 0.001,
        'kg': 1,
        'ton': 1000,
        'lb': 0.453592,
        'oz': 0.0283495,
        'stone': 6.35029,
        'grain': 6.479891e-5  # Grain
    },
    'time': {
        's': 1,
        'min': 60,
        'h': 3600,
        'day': 86400,
        'week': 604800,
        'month': 2628000,  # Average month (30.44 days)
        'year': 31536000  # Average year
    },
    'volume': {
        'ml': 0.001,
        'l': 1,
        'm3': 1000,
        'ft3': 28.3168,
        'gal': 3.78541,
        'pt': 0.473176,  # Pint
        'qt': 0.946353,  # Quart
        'cup': 0.24  # Cup
    },
    'temperature': {
        'C': ('C', lambda x: x),  # Celsius to Celsius
        'F': ('C', lambda f: (f - 32) * 5.0 / 9.0),  # Fahrenheit to Celsius
        'K': ('C', lambda k: k - 273.15)  # Kelvin to Celsius
    },
    'speed': {
        'm/s': 1,
        'km/h': 0.277778,
        'mph': 0.44704,
        'knot': 0.514444  # Knot
    },
    'energy': {
        'J': 1,
        'kJ': 1000,
        'cal': 4.184,
        'kcal': 4184,
        'kWh': 3.6e+6
    },
    'area': {
        'm2': 1,
        'km2': 1e+6,
        'ft2': 0.092903,
        'ac': 4046.86,  # Acre
        'ha': 10000  # Hectare
    },
    'pressure': {
        'Pa': 1,
        'bar': 1e+5,
        'atm': 101325,
        'psi': 6894.76
    }
}

# Temperature conversion helper functions (same as provided)
def temperature_to_celsius(value, from_unit):
    """ Convert any temperature unit to Celsius """
    if from_unit == 'C':
        return value
    elif from_unit == 'F':
        return (value - 32) * 5.0 / 9.0
    elif from_unit == 'K':
        return value - 273.15
    else:
        raise ValueError(f"Unsupported temperature unit: {from_unit}")

def celsius_to_target(value, to_unit):
    """ Convert Celsius to the target temperature unit """
    if to_unit == 'C':
        return value
    elif to_unit == 'F':
        return (value * 9.0 / 5.0) + 32
    elif to_unit == 'K':
        return value + 273.15
    else:
        raise ValueError(f"Unsupported temperature unit: {to_unit}")

def convert_temperature(value, from_unit, to_unit):
    """ Convert between any two temperature units """
    celsius_value = temperature_to_celsius(value, from_unit)
    return celsius_to_target(celsius_value, to_unit)

# General unit conversion function for length, mass, etc. (same as provided)
def convert_to_base_unit(value, from_unit, to_unit, category):
    """
    Converts a value from the specified unit to the target unit.

    Parameters:
    - value: The numeric value to be converted.
    - from_unit: The unit of the value (e.g., 'cm', 'm', 'kg').
    - to_unit: The unit to which the value will be converted.
    - category: The category of units (e.g., 'length', 'mass').

    Returns:
    - The converted value.
    """
    unit_conversion_factors = default_unit_conversion_factors.get(category, {})

    if not unit_conversion_factors:
        raise ValueError(f"Unsupported category: {category}. Please choose from available categories.")

    # Handle temperature conversions separately
    if category == 'temperature':
        return convert_temperature(value, from_unit, to_unit)

    # Handle predefined conversions for other categories
    if from_unit not in unit_conversion_factors or to_unit not in unit_conversion_factors:
        raise ValueError(f"Unsupported unit conversion from '{from_unit}' to '{to_unit}' in category '{category}'.")

    conversion_factor = unit_conversion_factors[from_unit]
    target_factor = unit_conversion_factors[to_unit]
    return value * conversion_factor / target_factor


@log_function_call
@track_changes
def convert_unit(
    df: Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray],
    columns: Union[str, List[str]],
    unit_category: str,
    from_unit: str,
    to_unit: str
) -> Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]:
    """
    Detects units in the specified columns and converts them to the target unit.

    Parameters:
    - df (Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]): The DataFrame or other data structure to process.
    - columns (str or list): The column(s) to check for unit conversion.
    - unit_category (str): The category of units to convert (e.g., 'length', 'mass', 'volume').
    - from_unit (str): The unit to convert from.
    - to_unit (str): The unit to convert to.

    Returns:
    - Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]: The data with converted values.
    """
    # Detect original data format
    original_format = type(df)

    # Convert to DataFrame for uniform processing if input is not already a DataFrame or Series
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        data_cleaned = df
    elif isinstance(df, (list, tuple)):
        data_cleaned = pd.DataFrame(df)
    elif isinstance(df, dict):
        data_cleaned = pd.DataFrame.from_dict(df)
    elif isinstance(df, np.ndarray):
        data_cleaned = pd.DataFrame(df)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                data_cleaned = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must represent a JSON object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    else:
        raise TypeError(f"Unsupported data type: {original_format}. Please provide a DataFrame, Series, list, dict, tuple, numpy array, or JSON string.")

    # Ensure columns is a list, even if a single string is passed
    if isinstance(columns, str):
        columns = [columns]

    # Validate inputs
    for column in columns:
        if column not in data_cleaned.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        if unit_category not in default_unit_conversion_factors:
            raise ValueError(f"Unit category '{unit_category}' is not defined.")
        if from_unit not in default_unit_conversion_factors.get(unit_category, {}):
            raise ValueError(f"Invalid 'from_unit': {from_unit} for unit category '{unit_category}'.")
        if to_unit not in default_unit_conversion_factors.get(unit_category, {}):
            raise ValueError(f"Invalid 'to_unit': {to_unit} for unit category '{unit_category}'.")

    # Copy DataFrame to avoid modifying the original
    converted_data = data_cleaned.copy()

    for column in columns:
        for idx, value in converted_data[column].items():
            if isinstance(value, (int, float)):
                try:
                    converted_value = convert_to_base_unit(value, from_unit, to_unit, unit_category)
                    converted_data.at[idx, column] = converted_value
                except ValueError as e:
                    print(f"Error converting value {value} in column '{column}': {e}")
            else:
                # Handle non-numeric values
                print(f"Skipping non-numeric value in column '{column}': {value}")

    # Return in the original format (if input was not DataFrame/Series)
    if original_format == pd.DataFrame:
        return converted_data
    elif original_format == pd.Series:
        return converted_data.squeeze()
    elif original_format == list:
        return converted_data.values.tolist()
    elif original_format == dict:
        return converted_data.to_dict(orient='records')
    elif original_format == tuple:
        return tuple(converted_data.values.tolist())
    elif original_format == str:
        return json.dumps(converted_data.to_dict(orient='records'))
    elif original_format == np.ndarray:
        return converted_data.to_numpy()
