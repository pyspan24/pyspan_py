import inspect
import logging

# Store logs in a list for retrieval
log_entries = []

def log_function_call(func):
    def wrapper(*args, **kwargs):
        # Get the name of the calling function
        caller = inspect.stack()[1]
        caller_function = caller.function
        caller_file = caller.filename

        # Create a log entry
        log_entry = f'Function "{func.__name__}" was called from "{caller_function}" in file "{caller_file}".'
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
from typing import Optional, Union, List
@log_function_call

def handle_nulls(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    action: str = 'remove',
    with_val: Optional[Union[int, float, str]] = None,
    by: Optional[str] = None,
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """
    Handle null values in a DataFrame by removing, replacing, or imputing them.

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

    Returns:
    - Optional[pd.DataFrame]: DataFrame with null values handled according to specified action, or None if inplace=True.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'df' parameter must be a pandas DataFrame.")

    # Check if 'with_val' is provided without 'replace' action
    if with_val is not None and action != 'replace':
        raise ValueError("The 'with_val' parameter can only be used when the 'action' is 'replace'.")

    # Check if 'by' is provided without 'impute' action
    if by is not None and action != 'impute':
        raise ValueError("The 'by' parameter can only be used when the 'action' is 'impute'.")

    # Check if 'impute' action is selected but 'by' is not provided
    if action == 'impute' and by is None:
        raise ValueError("An impute strategy must be provided when action is 'impute'.")

    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        subset_df = df[columns]
    else:
        subset_df = df

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

def clean_column_name(column_name: str) -> str:
    """
    Cleans and recommends a new column name by correcting spelling and removing special characters.

    Parameters:
    - column_name (str): The original column name.

    Returns:
    - str: The recommended clean column name.
    """
    # Initialize spell checker
    spell = SpellChecker()

    # Remove special characters and replace them with spaces
    clean_name = re.sub(r'[^a-zA-Z0-9 ]', ' ', column_name)

    # Correct spelling of words
    words = clean_name.split()
    corrected_words = [spell.correction(word) for word in words]

    # Join words with spaces and convert to title case
    recommended_name = ' '.join(corrected_words).title()
    return recommended_name

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

@log_function_call
def auto_rename_columns(df: pd.DataFrame) -> None:
    """
    Provide an interactive prompt for users to apply recommended column renaming.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns will be analyzed and potentially renamed.
    """
    # Get column rename recommendations
    rename_recommendations = rename_columns(df)

    if not rename_recommendations:
        print("All column names are already clean and readable!")
        return

    # Print recommendations
    print("\nRecommended Column Renames:")
    for original, recommended in rename_recommendations.items():
        print(f"Original: {original}, Recommended: {recommended}")

    # Ask the user if they want to apply the changes
    apply_changes = input("\nDo you want to apply these changes? (yes/no): ").strip().lower()
    if apply_changes == 'yes':
        apply_column_renames(df, rename_recommendations)
        print("\nRenamed DataFrame Column Names:")
        print(df.columns)
    else:
        print("\nNo changes were made.")






import pandas as pd
@log_function_call

def rename_dataframe_columns(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
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

def format_dt(
    df: pd.DataFrame,
    column_name: str,
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
    - column_name (str): The name of the column containing date/time data.
    - day (bool): If True, add a column with the day of the month.
    - month (bool): If True, add a column with the month.
    - year (bool): If True, add a column with the year.
    - quarter (bool): If True, add a column with the quarter of the year.
    - hour (bool): If True, add a column with the hour of the day.
    - minute (bool): If True, add a column with the minute of the hour.
    - day_of_week (bool): If True, add a column with the day of the week (0=Monday, 6=Sunday).
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
    # Check if the DataFrame contains the specified column
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Convert the column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        try:
            df[column_name] = pd.to_datetime(df[column_name])
        except Exception as e:
            raise ValueError(f"Failed to convert column '{column_name}' to datetime. Error: {e}")

    # Adding requested datetime features
    if day:
        df[f'{column_name}_day'] = df[column_name].dt.day
    if month:
        df[f'{column_name}_month'] = df[column_name].dt.month
    if year:
        df[f'{column_name}_year'] = df[column_name].dt.year
    if quarter:
        df[f'{column_name}_quarter'] = df[column_name].dt.quarter
    if hour:
        df[f'{column_name}_hour'] = df[column_name].dt.hour
    if minute:
        df[f'{column_name}_minute'] = df[column_name].dt.minute
    if day_of_week:
        df[f'{column_name}_day_of_week'] = df[column_name].dt.dayofweek

    # Apply date and time format and timezone conversion
    if from_timezone and to_timezone:
        # Convert timezone if both from and to timezones are specified
        df[column_name] = (
            df[column_name]
            .dt.tz_localize(from_timezone, ambiguous='NaT', nonexistent='NaT')
            .dt.tz_convert(to_timezone)
        )
    elif from_timezone or to_timezone:
        raise ValueError("Both from_timezone and to_timezone must be specified for timezone conversion.")

    # Apply date and time format
    df[column_name] = df[column_name].dt.strftime(f"{date_format} {time_format}")

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
@log_function_call
def convert_type(data, column=None):
    """
    Recommend and apply data type conversions for a given DataFrame or Series based on the analysis of each column's data.

    Parameters:
    data (pd.DataFrame or pd.Series): The input data to analyze.
    column (str or None): The specific column to analyze. If None, the function analyzes all columns.

    Returns:
    pd.DataFrame or pd.Series: The data with applied type conversions.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Data must be a pandas DataFrame or Series.")

    if column:
        if column not in data.columns:
            raise ValueError(f"Column {column} does not exist in the DataFrame.")
        data = data[[column]]

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
@log_function_call

def detect_outliers(data, method='iqr', threshold=1.5, columns=None, handle_missing=True):
    """
    Detect outliers in a dataset using the specified method and threshold.

    Parameters:
    data (Union[pd.Series, pd.DataFrame]): The input data to analyze.
    method (str, default='iqr'): The outlier detection method to use. Options include 'z-score' and 'iqr'.
    threshold (float, default=1.5): The threshold for outlier detection.
    columns (List[str], optional): List of columns to apply the outlier detection on. If None, applies to all numerical columns.
    handle_missing (bool, default=True): Whether to drop rows with outliers and print the shape of the data before and after.

    Returns:
    pd.DataFrame: A DataFrame with an additional boolean column 'is_outlier' or the cleaned DataFrame.
    """

    # Validate input
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Data must be a pandas DataFrame or Series.")

    if not isinstance(threshold, (int, float)):
        raise TypeError("Threshold must be a numeric value.")

    if method not in ['z-score', 'iqr']:
        raise ValueError("Method must be one of ['z-score', 'iqr'].")

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Handle missing values
    data_original_shape = data.shape
    if handle_missing:
        data = data.dropna()

    # Select numerical columns if columns is None
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns

    # Initialize a boolean mask for outliers
    outliers = pd.DataFrame(index=data.index, columns=columns, data=False)

    if method == 'z-score':
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column {col} is not in the DataFrame.")
            
            mean = data[col].mean()
            std = data[col].std()

            # Calculate Z-score
            z_scores = (data[col] - mean) / std
            # Mark outliers
            outliers[col] = np.abs(z_scores) > threshold

    elif method == 'iqr':
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column {col} is not in the DataFrame.")

            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Mark outliers
            outliers[col] = (data[col] < lower_bound) | (data[col] > upper_bound)

    # Count the total number of outliers
    total_outliers = outliers.sum().sum()
    print(f"Total number of outliers detected: {total_outliers}")

    # Combine the outlier information with the original data
    data_with_outliers = data.copy()
    data_with_outliers['is_outlier'] = outliers.any(axis=1)

    # If handle_missing is True, drop rows with any outliers and print shape before and after
    if handle_missing:
        data_cleaned = data_with_outliers[~data_with_outliers['is_outlier']].drop(columns='is_outlier')

        print(f"Original data shape: {data_original_shape}")
        print(f"Data shape after removing outliers: {data_cleaned.shape}")

        return data_cleaned
    else:
        return data_with_outliers


import re
import pandas as pd
@log_function_call

def remove_chars(df, columns, strip_all=False, custom_characters=None):
    """
    Cleans specified columns in a DataFrame by trimming spaces and optionally removing custom characters.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to be cleaned.
    - columns (list of str): A list of column names to apply the cleaning function to.
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

    # Apply the cleaning function to the specified columns
    df[columns] = df[columns].map(clean_text)

    return df





import pandas as pd
@log_function_call
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
def scale_data(df, column_name, method='min-max'):
    """
    Scales the specified column in the DataFrame using the given scaling method.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to scale.
    - column_name (str): The name of the column to scale.
    - method (str): The scaling method to use. Options are 'min-max', 'robust', and 'standard'.
    
    Returns:
    - pd.DataFrame: The DataFrame with the specified column scaled.
    
    Raises:
    - ValueError: If the specified method is not recognized.
    """
    
    if method == 'min-max':
        # Min-Max Scaling
        X = df[column_name]
        X_min = X.min()
        X_max = X.max()
        df[column_name] = (X - X_min) / (X_max - X_min)
        
    elif method == 'robust':
        # Robust Scaling
        X = df[column_name]
        median = X.median()
        IQR = X.quantile(0.75) - X.quantile(0.25)
        df[column_name] = (X - median) / IQR
        
    elif method == 'standard':
        # Standard Scaling
        X = df[column_name]
        mean = X.mean()
        std = X.std()
        df[column_name] = (X - mean) / std
        
    else:
        raise ValueError(f"Scaling method '{method}' is not recognized. Choose from 'min-max', 'robust', or 'standard'.")
    
    return df



