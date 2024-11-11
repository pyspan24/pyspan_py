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
from typing import Optional, Union, List
@log_function_call
@track_changes

def handle_nulls(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    action: str = 'remove',
    with_val: Optional[Union[int, float, str]] = None,
    by: Optional[str] = None,
    inplace: bool = False,
    threshold: Optional[float] = None,
    axis: str = 'rows'
) -> Optional[pd.DataFrame]:
    """
    Handle null values in a DataFrame by removing, replacing, imputing, or dropping rows/columns
    with more than x% missing values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (Optional[Union[str, List[str]]]): Column(s) to check for null values.
      If None, apply to the entire DataFrame. Default is None.
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
    - Optional[pd.DataFrame]: DataFrame with null values handled according to specified action, or None if inplace=True.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'df' parameter must be a pandas DataFrame.")

    # Validate axis parameter
    if axis not in ['rows', 'columns']:
        raise ValueError("The 'axis' parameter must be either 'rows' or 'columns'.")

    # Validate threshold
    if threshold is not None:
        if not (0 <= threshold <= 100):
            raise ValueError("The 'threshold' parameter must be between 0 and 100.")

    # Handle missing columns parameter
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

    if inplace:
        df.update(df_cleaned)
        return None
    else:
        return df_cleaned



import pandas as pd
from typing import Optional, Union, List

@log_function_call
@track_changes

def remove(
    df: pd.DataFrame,
    operation: str,
    columns: Optional[Union[str, List[str]]] = None,
    keep: Optional[str] = 'first',
    consider_all: bool = True,
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """
    Remove duplicates or columns from a DataFrame based on the specified operation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
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
    - inplace (bool): If True, modify the DataFrame in place. If False, return a new DataFrame. Default is False.

    Returns:
    - Optional[pd.DataFrame]: DataFrame with duplicates or columns removed according to specified criteria, or None if inplace=True.

    Raises:
    - ValueError: If invalid columns are specified or operation is invalid.
    - TypeError: If input types are incorrect.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'df' parameter must be a pandas DataFrame.")

    if operation == 'duplicates':
        # Validate the keep parameter
        if keep not in ['first', 'last', 'none', None]:
            raise ValueError("keep must be one of ['first', 'last', 'none'] or None.")

        # Validate columns input
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]  # Convert single column name to a list

            # Check if all columns in columns exist in the DataFrame
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Columns {missing_columns} not found in DataFrame.")

        # Remove duplicates
        if keep == 'none':
            df_cleaned = df.drop_duplicates(subset=columns, keep=False)
        else:
            if consider_all:
                df_cleaned = df.drop_duplicates(subset=columns, keep=keep or 'first')
            else:
                df_cleaned = df.drop_duplicates(keep=keep or 'first')

    elif operation == 'columns':
        # Validate input
        if isinstance(columns, str):
            columns = [columns]  # Convert a single column name to a list
        elif not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise TypeError("columns must be a string or a list of column names.")

        # Check if columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in DataFrame.")

        # Remove specified columns
        df_cleaned = df.drop(columns=columns)

    else:
        raise ValueError("operation must be either 'duplicates' or 'columns'.")

    if inplace:
        df.update(df_cleaned)
        return None
    else:
        return df_cleaned






import pandas as pd
import re
from spellchecker import SpellChecker

def clean_text(text: str) -> str:
    """
    Cleans text by removing special characters, correcting spelling, and removing extra spaces.

    Parameters:
    - text (str): The original text to clean.

    Returns:
    - str: The cleaned text.
    """
    # Check if text is None and handle it
    if text is None:
        return None

    # Remove special characters and replace them with spaces
    clean_text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)

    # Join words with single spaces and strip leading/trailing spaces
    return ' '.join(clean_text.split()).strip()

def clean_column_name(column_name: str) -> str:
    """
    Cleans and recommends a new column name by correcting spelling and removing special characters.

    Parameters:
    - column_name (str): The original column name.

    Returns:
    - str or None: The recommended clean column name, or None if the column name is invalid.
    """
    # Check if text is None and handle it
    if column_name is None:
        return None

    # Initialize spell checker
    spell = SpellChecker()

    # Remove special characters for spelling check
    clean_text = re.sub(r'[^a-zA-Z0-9 ]', ' ', column_name)

    # Correct spelling of words
    words = clean_text.split()
    corrected_words = []
    for word in words:
        if word:
            corrected_word = spell.correction(word)
            if corrected_word:  # Only add valid corrected words
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)  # Keep the original word if no correction
        else:
            corrected_words.append(word)

    # Join words, capitalize each, and remove extra spaces
    cleaned_column_name = ' '.join(corrected_words).title()

    # Return None if the cleaned name is equivalent to the original (or is empty)
    if cleaned_column_name == column_name or cleaned_column_name.strip() == "":
        return None

    return cleaned_column_name


def clean_row_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the rows of a DataFrame by removing special characters and extra spaces.

    Parameters:
    - df (pd.DataFrame): The DataFrame to clean.

    Returns:
    - pd.DataFrame: The cleaned DataFrame with cleaned row data.
    """
    # Apply clean_text function to each cell in the DataFrame
    df_cleaned = df.map(lambda x: clean_text(str(x)) if isinstance(x, str) else x)
    return df_cleaned

def rename_columns(df: pd.DataFrame) -> dict:
    """
    Automatically recommend readable column names for a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame with columns to be analyzed.

    Returns:
    - dict: A dictionary where keys are current column names and values are recommended new names.
    """
    recommendations = {}

    for column in df.columns:
        recommended_name = clean_column_name(column)
        if recommended_name != column:
            recommendations[column] = recommended_name

    return recommendations

def apply_column_renames(df: pd.DataFrame, rename_map: dict) -> None:
    """
    Apply the recommended column name changes to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to rename columns in.
    - rename_map (dict): The dictionary of column renaming recommendations.
    """
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

# Refined function
@log_function_call
@track_changes
def refine(df: pd.DataFrame, clean_rows: bool = True) -> pd.DataFrame:
    """
    Refines the DataFrame by cleaning both column names and optionally row data.

    Parameters:
    - df (pd.DataFrame): The DataFrame to refine.
    - clean_rows (bool): Whether to clean the row data as well as the column names. Default is True.

    Returns:
    - pd.DataFrame: The refined DataFrame.
    """
    # Get column rename recommendations
    rename_recommendations = rename_columns(df)

    if not rename_recommendations:
        print("All column names are already clean and readable!")
    else:
        # Print recommendations
        print("\nRecommended Column Renames:")
        for original, recommended in rename_recommendations.items():
            print(f"Original: {original}, Recommended: {recommended}")

        # Ask the user if they want to apply the changes
        apply_changes = input("\nDo you want to apply these column name changes? (yes/no): ").strip().lower()
        if apply_changes == 'yes':
            apply_column_renames(df, rename_recommendations)
            print("\nRenamed DataFrame Column Names:")
            print(df.columns)
        else:
            print("\nNo changes were made to column names.")

    # Clean rows if specified
    if clean_rows:
        # Ask the user if they want to clean the row data
        clean_row_data_prompt = input("\nDo you want to clean row data (remove special characters and extra spaces)? (yes/no): ").strip().lower()
        if clean_row_data_prompt == 'yes':
            df = clean_row_data(df)
            print("\nRow data has been cleaned.")
        else:
            print("\nNo changes were made to row data.")

    return df





import pandas as pd
@log_function_call
@track_changes

def manual_rename_columns(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
    """
    Rename columns in a DataFrame using a provided dictionary mapping.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns are to be renamed.
    - rename_dict (dict): A dictionary mapping current column names (keys) to new column names (values).

    Returns:
    - pd.DataFrame: A new DataFrame with columns renamed according to the dictionary.

    Raises:
    - KeyError: If any of the specified columns in the dictionary keys do not exist in the DataFrame.
    - ValueError: If the rename_dict is empty.
    """
    if not isinstance(rename_dict, dict) or not rename_dict:
        raise ValueError("rename_dict must be a non-empty dictionary.")

    # Check for columns that are not in the DataFrame
    missing_columns = [col for col in rename_dict if col not in df.columns]
    if missing_columns:
        raise KeyError(f"These columns do not exist in the DataFrame: {missing_columns}")

    # Rename columns using the provided dictionary
    renamed_df = df.rename(columns=rename_dict)

    return renamed_df




import pandas as pd
import pytz
from typing import Union, List, Optional

@log_function_call
@track_changes
def format_dt(
    df: pd.DataFrame,
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
) -> pd.DataFrame:
    """
    Add additional date/time-based columns and format date/time columns in a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to which new date/time features will be added and formatted.
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
    - pd.DataFrame: The DataFrame with added date/time features and formatted date/time columns.
    
    Raises:
    - ValueError: If the specified column does not exist in the DataFrame or conversion fails.
    """
    
    # Ensure columns is a list to handle both single and multiple columns
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        # Check if the DataFrame contains the specified column
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
            # Convert timezone if both from and to timezones are specified
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




import pandas as pd
from collections import Counter

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

def split_column(df: pd.DataFrame, column: str, delimiter: str = None) -> pd.DataFrame:
    """
    Split a single column into multiple columns based on a delimiter.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column to split.
    - column (str): The name of the column to be split.
    - delimiter (str): The delimiter to use for splitting. If None, detect the most common delimiter.

    Returns:
    - pd.DataFrame: A new DataFrame with the specified column split into multiple columns.
    """
    # Validate input
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if delimiter is None:
        # Detect the delimiter if not provided
        delimiter = detect_delimiter(df[column])
        if delimiter is None:
            raise ValueError("No delimiter detected and none provided.")
    else:
        # If the user provides a delimiter, use it regardless of the predefined list
        if not isinstance(delimiter, str) or len(delimiter) == 0:
            raise ValueError("Provided delimiter must be a non-empty string.")

    # Split the column based on the delimiter
    expanded_columns = df[column].str.split(delimiter, expand=True)

    # Remove any columns that are entirely NaN (i.e., extra columns)
    expanded_columns = expanded_columns.dropna(how='all', axis=1)

    # Drop the original column and concatenate the new columns
    df_expanded = df.drop(columns=[column]).join(expanded_columns)

    # Rename new columns with a suffix to identify them
    df_expanded.columns = list(df_expanded.columns[:-len(expanded_columns.columns)]) + \
                          [f"{column}_{i+1}" for i in range(len(expanded_columns.columns))]

    return df_expanded




import pandas as pd
from spellchecker import SpellChecker
import re

# Initialize SpellCheckers for different dictionaries
spell_checker_dict = {
    'en_US': SpellChecker(language='en'),
    # You can add more dictionaries if needed
}

def spell_check_dataframe(data, dictionary='en_US', columns=None):
    """
    Perform spell check on the specified columns of a DataFrame using the specified dictionary.
    """
    if columns is None:
        columns = data.select_dtypes(include=['object']).columns

    if dictionary not in spell_checker_dict:
        raise ValueError("Dictionary must be one of 'en_US'")

    spell_checker = spell_checker_dict[dictionary]
    misspelled_words = {}

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column {column} is not in the DataFrame.")

        text_data = data[column].dropna()
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

def detect_errors(data, date_columns=None, numeric_columns=None, text_columns=None, date_format=None):
    """
    Detect and flag data entry errors in a DataFrame, including invalid dates and misspelled words.
    """
    errors = []

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")

    # Default to American English dictionary
    spellcheck_dict = 'en_US'

    # Check for invalid dates if date_columns is provided
    if date_columns:
        for col in date_columns:
            if col in data.columns:
                invalid_dates = detect_invalid_dates(data[col], date_format=date_format)
                for idx in invalid_dates[invalid_dates].index:
                    errors.append({'Column': col, 'Error Type': 'Invalid Date', 'Value': data.at[idx, col]})

    # Check for invalid numeric formats if numeric_columns is provided
    if numeric_columns:
        for col in numeric_columns:
            if col in data.columns:
                non_numeric = pd.to_numeric(data[col], errors='coerce').isna() & data[col].notna()
                for idx in non_numeric[non_numeric].index:
                    errors.append({'Column': col, 'Error Type': 'Invalid Numeric', 'Value': data.at[idx, col]})

    # Spell check on text columns if text_columns is provided
    if text_columns:
        misspelled_words = spell_check_dataframe(data, dictionary=spellcheck_dict, columns=text_columns)
        for col, words in misspelled_words.items():
            for word in words:
                indices = data[col].apply(lambda x: word in x if isinstance(x, str) else False)
                for idx in indices[indices].index:
                    errors.append({'Column': col, 'Error Type': 'Misspelled Word', 'Value': word})

    return pd.DataFrame(errors)





import pandas as pd
from typing import Union, Optional, List

@log_function_call
@track_changes
def convert_type(data: Union[pd.DataFrame, pd.Series], columns: Optional[Union[str, List[str]]] = None) -> Union[pd.DataFrame, pd.Series]:
    """
    Recommend and apply data type conversions for a given DataFrame or Series based on the analysis of each column's data.

    Parameters:
    - data (pd.DataFrame or pd.Series): The input data to analyze.
    - columns (str, list of str, or None): The specific column(s) to analyze. If None, the function analyzes all columns.

    Returns:
    - pd.DataFrame or pd.Series: The data with applied type conversions.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Data must be a pandas DataFrame or Series.")

    # If a single column is provided as a string, convert it to a list
    if isinstance(columns, str):
        columns = [columns]

    # If columns are provided, check if they exist in the DataFrame
    if columns:
        for column in columns:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        data_to_analyze = data[columns]
    else:
        data_to_analyze = data

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
    for col_name, col_data in data.items():
        current_dtype = col_data.dtype
        suggestions = suggest_conversion(col_data)

        if suggestions:
            recommendations[col_name] = {
                'current_dtype': current_dtype,
                'suggestions': suggestions
            }

    # Display recommendations and get user confirmation
    for col_name, rec in recommendations.items():
        print(f"\nColumn: {col_name}")
        print(f"Current Data Type: {rec['current_dtype']}")
        print(f"Recommended Conversions: {', '.join(rec['suggestions'])}")

        # Ask user to confirm each conversion
        for suggestion in rec['suggestions']:
            user_input = input(f"Do you want to {suggestion.lower()} for column '{col_name}'? (yes/no): ")
            if user_input.lower() == 'yes':
                if suggestion == 'Convert to integer':
                    data[col_name] = pd.to_numeric(data[col_name], errors='coerce').fillna(0).astype(int)
                elif suggestion == 'Convert to numeric':
                    data[col_name] = pd.to_numeric(data[col_name], errors='coerce')
                elif suggestion == 'Convert to category':
                    data[col_name] = data[col_name].astype('category')
                elif suggestion == 'Convert to datetime':
                    data[col_name] = pd.to_datetime(data[col_name], errors='coerce')
                print(f"Column '{col_name}' converted to {data[col_name].dtype}.")
            else:
                print(f"Column '{col_name}' not converted.")

    return data





import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
@log_function_call
@track_changes


def detect_outliers(
    data,
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
    - data (pd.DataFrame): The DataFrame containing the data.
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
    - pd.DataFrame: A DataFrame with outliers removed, and prints the data shape before and after removal.
    """
    
    # Convert single column to list for uniform processing
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

    if method not in ['iqr', 'z-score'] and anomaly_method not in ['isolation_forest', 'lof', 'dbscan', None]:
        raise ValueError("Method must be one of ['iqr', 'z-score'] or a valid anomaly method ['isolation_forest', 'lof', 'dbscan'].")

    # Handle missing values
    original_shape = data.shape
    if handle_missing:
        data = data.dropna(subset=columns)

    # Initialize mask to identify outliers
    combined_outliers = np.zeros(data.shape[0], dtype=bool)

    for column in columns:
        if anomaly_method is None:
            outlier_type = "outliers"
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (data[column] < lower_bound) | (data[column] > upper_bound)

            elif method == 'z-score':
                mean = data[column].mean()
                std = data[column].std()
                z_scores = (data[column] - mean) / std
                outliers = np.abs(z_scores) > threshold
        else:
            outlier_type = "anomalies"
            if anomaly_method == 'isolation_forest':
                model = IsolationForest(contamination=contamination, random_state=42)
                outliers = model.fit_predict(data[[column]]) == -1
            elif anomaly_method == 'lof':
                lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                outliers = lof_model.fit_predict(data[[column]]) == -1
            elif anomaly_method == 'dbscan':
                dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
                outliers = dbscan_model.fit_predict(data[[column]]) == -1
            else:
                raise ValueError(f"Anomaly method '{anomaly_method}' is not supported.")
        
        combined_outliers |= outliers

        num_outliers = np.sum(outliers)
        print(f"Total number of {outlier_type} detected in column '{column}': {num_outliers}")

    # Remove all detected outliers
    data_cleaned = data[~combined_outliers]

    # Print data shape before and after outlier removal
    print(f"Original data shape: {original_shape}")
    print(f"Data shape after removing {outlier_type}: {data_cleaned.shape}")

    return data_cleaned


import re
import pandas as pd
from typing import Union, List

@log_function_call
@track_changes

def remove_chars(df: pd.DataFrame, columns: Union[str, List[str]], strip_all=False, custom_characters=None) -> pd.DataFrame:
    """
    Cleans specified columns in a DataFrame by trimming spaces and optionally removing custom characters.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to be cleaned.
    - columns (str or list of str): A column name or a list of column names to apply the cleaning function to.
    - strip_all (bool): If True, all extra spaces will be removed;
      otherwise, only leading, trailing, and extra spaces between words will be reduced to one.
    - custom_characters (str or None): A string of characters to be removed from the text.
      If None, no custom characters will be removed.

    Returns:
    - pd.DataFrame: The DataFrame with the specified columns cleaned.
    """
    def clean_text(text):
        if isinstance(text, str):
            # Trim leading and trailing spaces
            trimmed_text = text.strip()

            if strip_all:
                # Remove all extra spaces
                cleaned_text = re.sub(r'\s+', '', trimmed_text)
            else:
                # Replace multiple spaces with a single space
                cleaned_text = re.sub(r'\s+', ' ', trimmed_text)

            if custom_characters:
                # Remove custom characters if specified
                cleaned_text = re.sub(f'[{re.escape(custom_characters)}]', '', cleaned_text)

            return cleaned_text
        else:
            return text  # Return as is if it's not a string

     # If a single column name is provided as a string, convert it to a list
    if isinstance(columns, str):
        columns = [columns]

    # Apply the cleaning function to each specified column
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(clean_text)
        else:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    return df





import pandas as pd
@log_function_call
@track_changes

def reformat(df, target_column, reference_column):
    """
    Applies the data type and formatting from a reference column to a target column in the same DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing both the target and reference columns.
    - target_column (str): The name of the column to format.
    - reference_column (str): The name of the column to borrow formatting from.
    
    Returns:
    - pd.DataFrame: The DataFrame with the target column formatted based on the reference column.
    
    Raises:
    - ValueError: If the target column or reference column does not exist in the DataFrame.
    - TypeError: If the reference column is not of a type that can be applied to the target column.
    """
    
    # Check if the target and reference columns exist in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")
    if reference_column not in df.columns:
        raise ValueError(f"Column '{reference_column}' does not exist in the DataFrame.")
    
    # Get the data type of the reference column
    ref_dtype = df[reference_column].dtype
    
    # Check and apply formatting based on data type
    if pd.api.types.is_datetime64_any_dtype(ref_dtype):
        # If the reference column is datetime, convert the target column to datetime
        try:
            df[target_column] = pd.to_datetime(df[target_column], errors='coerce')
        except Exception as e:
            raise TypeError(f"Error converting '{target_column}' to datetime: {e}")
    elif pd.api.types.is_numeric_dtype(ref_dtype):
        # If the reference column is numeric, convert the target column to numeric
        try:
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        except Exception as e:
            raise TypeError(f"Error converting '{target_column}' to numeric: {e}")
    elif pd.api.types.is_string_dtype(ref_dtype):
        # If the reference column is string, apply string formatting based on the reference column's format
        ref_sample = df[reference_column].dropna().astype(str).iloc[0]
        
        if ref_sample.isupper():
            # If reference column is uppercase, convert target column to uppercase
            df[target_column] = df[target_column].astype(str).str.upper()
        elif ref_sample.islower():
            # If reference column is lowercase, convert target column to lowercase
            df[target_column] = df[target_column].astype(str).str.lower()
        elif ref_sample.istitle():
            # If reference column is title case, convert target column to title case
            df[target_column] = df[target_column].astype(str).str.title()
        else:
            # For other string formats, simply convert to string
            df[target_column] = df[target_column].astype(str)
    else:
        # For other types, raise an error or handle accordingly
        raise TypeError(f"Data type '{ref_dtype}' of reference column is not supported for formatting.")
    
    return df





import pandas as pd
import numpy as np

@log_function_call
@track_changes
def scale_data(df, columns, method='min-max'):
    """
    Scales the specified column(s) in the DataFrame using the given scaling method.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to scale.
    - columns (str or list): The name(s) of the column(s) to scale.
    - method (str): The scaling method to use. Options are 'min-max', 'robust', and 'standard'.
    
    Returns:
    - pd.DataFrame: The DataFrame with the specified column(s) scaled.
    
    Raises:
    - ValueError: If the specified method is not recognized.
    """
    
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if method == 'min-max':
            # Min-Max Scaling
            X = df[column]
            X_min = X.min()
            X_max = X.max()
            df[column] = (X - X_min) / (X_max - X_min)

        elif method == 'robust':
            # Robust Scaling
            X = df[column]
            median = X.median()
            IQR = X.quantile(0.75) - X.quantile(0.25)
            df[column] = (X - median) / IQR

        elif method == 'standard':
            # Standard Scaling
            X = df[column]
            mean = X.mean()
            std = X.std()
            df[column] = (X - mean) / std

        else:
            raise ValueError(f"Scaling method '{method}' is not recognized. Choose from 'min-max', 'robust', or 'standard'.")
    
    return df




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

# Expanded unit conversion mappings
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

# Temperature conversion helper functions
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

# General unit conversion function for length, mass, etc.
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
def convert_unit(df, columns, unit_category, from_unit, to_unit):
    """
    Detects units in the specified columns and converts them to the target unit.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - columns (str or list): The column(s) to check for unit conversion.
    - unit_category (str): The category of units to convert (e.g., 'length', 'mass', 'volume').
    - from_unit (str): The unit to convert from.
    - to_unit (str): The unit to convert to.

    Returns:
    - pd.DataFrame: A new DataFrame with converted values.
    """
    # Ensure columns is a list, even if a single string is passed
    if isinstance(columns, str):
        columns = [columns]

    # Validate inputs
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        if unit_category not in default_unit_conversion_factors:
            raise ValueError(f"Unit category '{unit_category}' is not defined.")
        if from_unit not in default_unit_conversion_factors.get(unit_category, {}):
            raise ValueError(f"Invalid 'from_unit': {from_unit} for unit category '{unit_category}'.")
        if to_unit not in default_unit_conversion_factors.get(unit_category, {}):
            raise ValueError(f"Invalid 'to_unit': {to_unit} for unit category '{unit_category}'.")

    # Copy DataFrame to avoid modifying the original
    converted_data = df.copy()

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

    return converted_data

