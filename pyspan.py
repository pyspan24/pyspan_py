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
) -> Optional[Union[pd.DataFrame, List, dict, tuple, np.ndarray, str]]:
    """
    Handle null values in various data types by removing, replacing, imputing, or applying a threshold.

    Parameters:
    (same as the original)

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
#changes by Uzair
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
    return df.map(lambda x: clean_text(str(x)) if isinstance(x, str) else x)


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

# Refined function
@log_function_call
@track_changes
def refine(
    df: Union[pd.DataFrame, list, dict, tuple, np.ndarray, str],
    clean_rows: bool = True
) -> Union[pd.DataFrame, list, dict, tuple, str]:
    """
    Refines the data by cleaning column names and optionally row data.

    Parameters:
    - df (Union[pd.DataFrame, list, dict, tuple, np.ndarray, str]): The data to refine.
    - clean_rows (bool): Whether to clean row data as well as column names. Default is True.

    Returns:
    - Union[pd.DataFrame, list, dict, tuple, str]: Refined data in the same format as input.
    """
    # Handle list type
    if isinstance(df, list):
        return [clean_text(str(item)) if isinstance(item, str) else item for item in df]

    # Handle tuple type
    elif isinstance(df, tuple):
        return tuple(clean_text(str(item)) if isinstance(item, str) else item for item in df)

    # Handle numpy array type
    elif isinstance(df, np.ndarray):
        return np.vectorize(lambda x: clean_text(str(x)) if isinstance(x, str) else x)(df)

    # Handle dictionary type
    elif isinstance(df, dict):
        cleaned_data = {}
        for key, value in df.items():
            cleaned_key = clean_column_name(key)  # Clean the column name (key)
            if isinstance(value, list):
                cleaned_value = [clean_text(str(item)) if isinstance(item, str) else item for item in value]
            else:
                cleaned_value = clean_text(str(value)) if isinstance(value, str) else value
            cleaned_data[cleaned_key] = cleaned_value
        return cleaned_data

    # Handle JSON string type
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                cleaned_data = refine(parsed_data)  # Reuse dictionary logic
                return json.dumps(cleaned_data)
        except json.JSONDecodeError:
            raise ValueError("String inputs must be valid JSON objects.")

    # Handle pandas DataFrame type
    elif isinstance(df, pd.DataFrame):
        # Get column rename recommendations
        rename_recommendations = rename_columns(df)

        if rename_recommendations:
            print("\nRecommended Column Renames:")
            for original, recommended in rename_recommendations.items():
                print(f"Original: {original}, Recommended: {recommended}")
            apply_changes = input("\nDo you want to apply these column name changes? (yes/no): ").strip().lower()
            if apply_changes == 'yes':
                apply_column_renames(df, rename_recommendations)
                print("\nRenamed DataFrame Column Names:")
                print(df.columns)

        # Clean rows if specified
        if clean_rows:
            clean_row_data_prompt = input("\nDo you want to clean row data (remove special characters and extra spaces)? (yes/no): ").strip().lower()
            if clean_row_data_prompt == 'yes':
                df = clean_row_data(df)
                print("\nRow data has been cleaned.")

        return df

    else:
        raise TypeError("Unsupported data type. Supported types are: list, tuple, dict, NumPy array, JSON string, or pandas DataFrame.")







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

    # Handle DataFrame
    if isinstance(df, pd.DataFrame):
        for column in columns if isinstance(columns, list) else [columns]:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

            # Convert the column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                try:
                    df[column] = pd.to_datetime(df[column])
                except Exception as e:
                    raise ValueError(f"Failed to convert column '{column}' to datetime. Error: {e}")

            # Adding requested datetime features
            if day:
                df[f'{column}_day'] = df[column].dt.day
            if month:
                df[f'{column}_month'] = df[column].dt.month
            if year:
                df[f'{column}_year'] = df[column].dt.year
            if quarter:
                df[f'{column}_quarter'] = df[column].dt.quarter
            if hour:
                df[f'{column}_hour'] = df[column].dt.hour
            if minute:
                df[f'{column}_minute'] = df[column].dt.minute
            if day_of_week:
                df[f'{column}_day_of_week'] = df[column].dt.day_name()

            # Apply date and time format and timezone conversion
            if from_timezone and to_timezone:
                df[column] = (
                    df[column]
                    .dt.tz_localize(from_timezone, ambiguous='NaT', nonexistent='NaT')
                    .dt.tz_convert(to_timezone)
                )
            elif from_timezone or to_timezone:
                raise ValueError("Both from_timezone and to_timezone must be specified for timezone conversion.")

            # Apply date and time format
            df[column] = df[column].dt.strftime(f"{date_format} {time_format}")

        return df

   # Handle list
#changes by Uzair for List start
    elif isinstance(df, list):
      if not all(isinstance(item, dict) for item in df):
        raise ValueError("Each item in the list should be a dictionary if it's not a DataFrame.")

      for column in columns if isinstance(columns, list) else [columns]:
          for item in df:
              if column not in item:
                  raise ValueError(f"Column '{column}' does not exist in the list of dictionaries.")

            # Convert to datetime (handles iterable or single datetime string)
              values = pd.to_datetime(item[column])

              if isinstance(values, pd.DatetimeIndex):  # If values are iterable
                  if day:
                      item[f'{column}_day'] = values.day.tolist()
                  if month:
                      item[f'{column}_month'] = values.month.tolist()
                  if year:
                      item[f'{column}_year'] = values.year.tolist()
                  if quarter:
                      item[f'{column}_quarter'] = values.quarter.tolist()
                  if hour:
                      item[f'{column}_hour'] = values.hour.tolist()
                  if minute:
                      item[f'{column}_minute'] = values.minute.tolist()
                  if day_of_week:
                      item[f'{column}_day_of_week'] = [val.strftime('%A') for val in values]
              else:  # If it's a single datetime value
                  if day:
                      item[f'{column}_day'] = values.day
                  if month:
                      item[f'{column}_month'] = values.month
                  if year:
                      item[f'{column}_year'] = values.year
                  if quarter:
                      item[f'{column}_quarter'] = (values.month - 1) // 3 + 1
                  if hour:
                      item[f'{column}_hour'] = values.hour
                  if minute:
                      item[f'{column}_minute'] = values.minute
                  if day_of_week:
                      item[f'{column}_day_of_week'] = values.strftime('%A')

      return df
#changes for list part end

#changes by Uzair Start for dictionary part

    elif isinstance(df, dict):
      for column in columns if isinstance(columns, list) else [columns]:
        if column not in df:
          raise ValueError(f"Column '{column}' does not exist in the dictionary.")

        # Convert to datetime (assumes values are iterable, like a list, or single datetime string)
        values = pd.to_datetime(df[column])

        if isinstance(values, pd.DatetimeIndex):  # If values are iterable
            if day:
                df[f'{column}_day'] = values.day.tolist()
            if month:
                df[f'{column}_month'] = values.month.tolist()
            if year:
                df[f'{column}_year'] = values.year.tolist()
            if quarter:
                df[f'{column}_quarter'] = values.quarter.tolist()
            if hour:
                df[f'{column}_hour'] = values.hour.tolist()
            if minute:
                df[f'{column}_minute'] = values.minute.tolist()
            if day_of_week:
                df[f'{column}_day_of_week'] = [val.strftime('%A') for val in values]
        else:  # If it's a single datetime value
            if day:
                df[f'{column}_day'] = values.day
            if month:
                df[f'{column}_month'] = values.month
            if year:
                df[f'{column}_year'] = values.year
            if quarter:
                df[f'{column}_quarter'] = (values.month - 1) // 3 + 1
            if hour:
                df[f'{column}_hour'] = values.hour
            if minute:
                df[f'{column}_minute'] = values.minute
            if day_of_week:
                df[f'{column}_day_of_week'] = values.strftime('%A')

      return df
#changes by Uzair end for dictionary part


#changes by Uzair start for tuple part
    elif isinstance(df, tuple):
      df_as_list = list(df)  # Convert tuple to list for easier manipulation

      for column in columns if isinstance(columns, list) else [columns]:
          for item in df_as_list:
              if isinstance(item, dict) and column in item:
                # Convert the column to datetime
                  values = pd.to_datetime(item[column])

                  if isinstance(values, pd.DatetimeIndex):  # If values are iterable
                      if day:
                          item[f'{column}_day'] = values.day.tolist()
                      if month:
                          item[f'{column}_month'] = values.month.tolist()
                      if year:
                          item[f'{column}_year'] = values.year.tolist()
                      if quarter:
                          item[f'{column}_quarter'] = values.quarter.tolist()
                      if hour:
                          item[f'{column}_hour'] = values.hour.tolist()
                      if minute:
                          item[f'{column}_minute'] = values.minute.tolist()
                      if day_of_week:
                          item[f'{column}_day_of_week'] = [val.strftime('%A') for val in values]
                  else:  # If it's a single datetime value
                      if day:
                          item[f'{column}_day'] = values.day
                      if month:
                          item[f'{column}_month'] = values.month
                      if year:
                          item[f'{column}_year'] = values.year
                      if quarter:
                          item[f'{column}_quarter'] = (values.month - 1) // 3 + 1
                      if hour:
                          item[f'{column}_hour'] = values.hour
                      if minute:
                          item[f'{column}_minute'] = values.minute
                      if day_of_week:
                          item[f'{column}_day_of_week'] = values.strftime('%A')
              else:
                  raise ValueError(f"Column '{column}' does not exist in one of the dictionaries in the tuple.")

      return tuple(df_as_list)  # Convert back to tuple

#changes by Uzair end for tuple part

    # Handle np.ndarray
    elif isinstance(df, np.ndarray):
        raise ValueError("DataFrames are expected, but if working with numpy arrays, conversion is not straightforward. Convert them to DataFrames first.")

    # Handle str (JSON)
    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                return format_dt(parsed_data, columns, day, month, year, quarter, hour, minute, day_of_week, date_format, time_format, from_timezone, to_timezone)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")

    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, list, tuple, dict, or JSON string.")



    





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
    # Handle different input types without converting to DataFrame

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
            if isinstance(text, str):  # Ensure the value is a string

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
    # Handle the input type without converting to DataFrame

    errors = []

    # Default to American English dictionary
    spellcheck_dict = 'en_US'

    # Check for invalid dates if date_columns is provided
    if isinstance(df, pd.DataFrame):
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    invalid_dates = detect_invalid_dates(df[col], date_format=date_format)
                    for idx in invalid_dates[invalid_dates].index:
                        errors.append({'Column': col, 'Error Type': 'Invalid Date', 'Value': df.at[idx, col]})

        # Check for invalid numeric formats if numeric_columns is provided
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    non_numeric = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
                    for idx in non_numeric[non_numeric].index:
                        errors.append({'Column': col, 'Error Type': 'Invalid Numeric', 'Value': df.at[idx, col]})

        # Spell check on text columns if text_columns is provided
        if text_columns:
            misspelled_words = spell_check_dataframe(df, dictionary=spellcheck_dict, columns=text_columns)
            for col, words in misspelled_words.items():
                for word in words:
                    indices = df[col].apply(lambda x: word in x if isinstance(x, str) else False)
                    for idx in indices[indices].index:
                        errors.append({'Column': col, 'Error Type': 'Misspelled Word', 'Value': word})

    elif isinstance(df, (list, tuple, np.ndarray)):
        # Convert list, tuple, or ndarray to DataFrame for processing
        df_converted = pd.DataFrame(df)

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

    elif isinstance(df, dict):
        # Convert dict to DataFrame for processing
        df_converted = pd.DataFrame.from_dict(df)

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

    elif isinstance(df, str):
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                df_converted = pd.DataFrame.from_dict(parsed_data)
            else:
                raise ValueError("String input must be valid JSON representing an object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")

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

    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, list, tuple, dict, or JSON string.")

    # Convert result to a DataFrame
    errors_df = pd.DataFrame(errors)

    # Return the errors in DataFrame format
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
) -> Union[pd.DataFrame, pd.Series, list, dict, tuple, str]:
    """
    Recommend and apply data type conversions for a given DataFrame, Series, or other supported formats.

    Parameters:
    - df (Union[pd.DataFrame, pd.Series, list, dict, tuple, np.ndarray, str]): The input data to analyze.
    - columns (str, list of str, or None): The specific column(s) to analyze. If None, the function analyzes all columns.

    Returns:
    - pd.DataFrame, pd.Series, list, dict, tuple, or str: The data with applied type conversions.
    """
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

    # Conversion logic for DataFrame
    if isinstance(df, pd.DataFrame):
        recommendations = {}

        if columns:
            for column in columns:
                if column not in df.columns:
                    raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
            data_to_analyze = df[columns]
        else:
            data_to_analyze = df

        for col_name, col_data in data_to_analyze.items():
            suggestions = suggest_conversion(col_data)
            if suggestions:
                recommendations[col_name] = {
                    'current_dtype': col_data.dtype,
                    'suggestions': suggestions
                }

        if not recommendations:
            print("All types are fine. No conversion required.")
            return df

        for col_name, rec in recommendations.items():
            print(f"\nColumn: {col_name}")
            print(f"Current Data Type: {rec['current_dtype']}")
            print(f"Recommended Conversions: {', '.join(rec['suggestions'])}")
            for suggestion in rec['suggestions']:
                user_input = input(f"Apply {suggestion} to column '{col_name}'? (yes/no): ").strip().lower()
                if user_input == 'yes':
                    if suggestion == 'Convert to integer':
                        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).astype(int)
                    elif suggestion == 'Convert to numeric':
                        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    elif suggestion == 'Convert to category':
                        df[col_name] = df[col_name].astype('category')
                    elif suggestion == 'Convert to datetime':
                        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                    print(f"Column '{col_name}' converted to {df[col_name].dtype}.")
        return df

    # Conversion logic for Series
    elif isinstance(df, pd.Series):
        suggestions = suggest_conversion(df)

        if not suggestions:
            print("All types are fine. No conversion required.")
            return df

        for suggestion in suggestions:
            user_input = input(f"Apply {suggestion} to Series? (yes/no): ").strip().lower()
            if user_input == 'yes':
                if suggestion == 'Convert to integer':
                    df = pd.to_numeric(df, errors='coerce').fillna(0).astype(int)
                elif suggestion == 'Convert to numeric':
                    df = pd.to_numeric(df, errors='coerce')
                elif suggestion == 'Convert to category':
                    df = df.astype('category')
                elif suggestion == 'Convert to datetime':
                    df = pd.to_datetime(df, errors='coerce')
                print(f"Series converted to {df.dtype}.")
        return df

    # Conversion logic for lists, tuples, and arrays
    elif isinstance(df, (list, tuple, np.ndarray)):
        data = pd.Series(df)
        suggestions = suggest_conversion(data)

        if not suggestions:
            print("All types are fine. No conversion required.")
            return df

        for suggestion in suggestions:
            user_input = input(f"Apply {suggestion} to data? (yes/no): ").strip().lower()
            if user_input == 'yes':
                if suggestion == 'Convert to integer':
                    converted = pd.to_numeric(data, errors='coerce').fillna(0).astype(int)
                elif suggestion == 'Convert to numeric':
                    converted = pd.to_numeric(data, errors='coerce')
                elif suggestion == 'Convert to category':
                    converted = data.astype('category')
                elif suggestion == 'Convert to datetime':
                    converted = pd.to_datetime(data, errors='coerce')
                converted = converted.tolist() if isinstance(df, list) else tuple(converted) if isinstance(df, tuple) else np.array(converted)
                print(f"Data converted to {type(converted)}.")
                return converted
        return df

    # Conversion logic for dictionary
    elif isinstance(df, dict):
        try:
            data = pd.DataFrame.from_dict(df, orient='index').T
            return convert_type(data, columns)
        except Exception as e:
            raise ValueError("Unable to process dictionary. Ensure it represents tabular data.") from e

    # Conversion logic for JSON string
    elif isinstance(df, str):
        try:
            parsed = json.loads(df)
            return convert_type(parsed)
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")

    else:
        raise TypeError("Unsupported data type. Please provide a DataFrame, Series, list, dict, tuple, or JSON string.")







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

    #changes by Uzair
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










import pandas as pd
import numpy as np
import json
from typing import Union, List, Optional

@log_function_call
@track_changes
def scale_data(
    df: Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray],
    columns: Union[str, List[str], None] = None,
    method: str = 'min-max'
) -> Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]:
    """
    Scales the specified column(s) in the DataFrame (or equivalent) using the given scaling method.

    Parameters:
    - df (Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]): The input data.
    - columns (str or list of str, optional): The column name(s) to scale. If not specified, it will scale the entire content of lists or dict values.
    - method (str): The scaling method to use. Options are 'min-max', 'robust', and 'standard'.

    Returns:
    - Union[pd.DataFrame, pd.Series, list, dict, tuple, str, np.ndarray]: The scaled data.
    """
    # Detect original data format
    original_format = type(df)

    # Handle DataFrame input
    if isinstance(df, pd.DataFrame):
        # If a single column name is provided as a string, convert it to a list
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

            if method == 'min-max':
                X = df[column]
                X_min = X.min()
                X_max = X.max()
                df[column] = (X - X_min) / (X_max - X_min)

            elif method == 'robust':
                X = df[column]
                median = X.median()
                IQR = X.quantile(0.75) - X.quantile(0.25)
                df[column] = (X - median) / IQR

            elif method == 'standard':
                X = df[column]
                mean = X.mean()
                std = X.std()
                df[column] = (X - mean) / std

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
    # Handle the input type based on original format

    if isinstance(df, (list, tuple)):
        # Process list/tuple directly
        data_cleaned = list(df) if isinstance(df, list) else tuple(df)
    elif isinstance(df, dict):
        data_cleaned = df.copy()
    elif isinstance(df, np.ndarray):
        data_cleaned = df.copy()
    elif isinstance(df, str):
        # Parse JSON string
        try:
            parsed_data = json.loads(df)
            if isinstance(parsed_data, dict):
                data_cleaned = parsed_data
            else:
                raise ValueError("String input must represent a JSON object.")
        except json.JSONDecodeError:
            raise ValueError("String input must be valid JSON representing an object.")
    elif isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        # Directly use the DataFrame or Series as it is
        data_cleaned = df
    else:
        raise TypeError(f"Unsupported data type: {type(df)}. Please provide a DataFrame, Series, list, dict, tuple, numpy array, or JSON string.")

    # Ensure columns is a list, even if a single string is passed
    if isinstance(columns, str):
        columns = [columns]

    # Validate inputs
    for column in columns:
        if isinstance(data_cleaned, dict) and column not in data_cleaned:
            raise ValueError(f"Column '{column}' does not exist in the data structure.")
        if unit_category not in default_unit_conversion_factors:
            raise ValueError(f"Unit category '{unit_category}' is not defined.")
        if from_unit not in default_unit_conversion_factors.get(unit_category, {}):
            raise ValueError(f"Invalid 'from_unit': {from_unit} for unit category '{unit_category}'.")
        if to_unit not in default_unit_conversion_factors.get(unit_category, {}):
            raise ValueError(f"Invalid 'to_unit': {to_unit} for unit category '{unit_category}'.")

    # Apply conversion to the columns
    if isinstance(data_cleaned, (list, tuple)):
        for idx, value in enumerate(data_cleaned):
            if isinstance(value, (int, float)):
                try:
                    converted_value = convert_to_base_unit(value, from_unit, to_unit, unit_category)
                    data_cleaned[idx] = converted_value
                except ValueError as e:
                    print(f"Error converting value {value}: {e}")
            else:
                print(f"Skipping non-numeric value: {value}")
    elif isinstance(data_cleaned, dict):
        for column in columns:
            for idx, value in enumerate(data_cleaned[column]):
                if isinstance(value, (int, float)):
                    try:
                        converted_value = convert_to_base_unit(value, from_unit, to_unit, unit_category)
                        data_cleaned[column][idx] = converted_value
                    except ValueError as e:
                        print(f"Error converting value {value} in column '{column}': {e}")
                else:
                    print(f"Skipping non-numeric value in column '{column}': {value}")
    elif isinstance(data_cleaned, (pd.Series, pd.DataFrame)):
        for column in columns:
            for idx, value in data_cleaned[column].items():
                if isinstance(value, (int, float)):
                    try:
                        converted_value = convert_to_base_unit(value, from_unit, to_unit, unit_category)
                        data_cleaned.at[idx, column] = converted_value
                    except ValueError as e:
                        print(f"Error converting value {value} in column '{column}': {e}")
                else:
                    print(f"Skipping non-numeric value in column '{column}': {value}")

    # Return in the original format (if input was not DataFrame/Series)
    return data_cleaned

