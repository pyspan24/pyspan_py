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
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """
    Handle rows with null values in a DataFrame by removing or replacing them.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (Optional[Union[str, List[str]]]): Column(s) to check for null values.
      If None, apply to the entire DataFrame. Default is None.
    - action (str): Action to perform on rows with null values.
      Options are 'remove' to drop rows or 'replace' to fill nulls with a custom value. Default is 'remove'.
    - with_val (Optional[Union[int, float, str]]): Custom value to replace nulls with, applicable if action is 'replace'.
    - inplace (bool): If True, modify the DataFrame in place. If False, return a new DataFrame. Default is False.

    Returns:
    - Optional[pd.DataFrame]: DataFrame with null values handled according to specified action, or None if inplace=True.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'df' parameter must be a pandas DataFrame.")

    # Validate the action parameter
    if action not in ['remove', 'replace']:
        raise ValueError("Action must be either 'remove' or 'replace'.")

    # Determine columns to process
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]  # Convert a single column name to a list
        subset_df = df[columns]
    else:
        subset_df = df

    # Remove or replace rows with nulls
    if action == 'remove':
        df_cleaned = df.dropna(subset=subset_df.columns)
    elif action == 'replace':
        if with_val is None:
            raise ValueError("with value must be provided when action is 'replace'.")
        df_cleaned = df.fillna({col: with_val for col in subset_df.columns})

    if inplace:
        df.update(df_cleaned)
        return None
    else:
        return df_cleaned




import pandas as pd
from typing import Optional, Union, List
@log_function_call

def remove_duplicates(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    keep: Optional[str] = 'first',
    consider_all: bool = True,
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """
    Remove duplicate rows from a DataFrame based on specified columns or the entire row.

    Parameters:
    - df (pd.DataFrame): The input DataFrame from which duplicates are to be removed.
    - columns (Optional[Union[str, List[str]]]): Specific column(s) to check for duplicates.
      If None, checks for duplicates in the entire DataFrame. Default is None.
    - keep (Optional[str]): Determines which duplicates to keep. Options are:
      'first': Keep the first occurrence of each duplicate.
      'last': Keep the last occurrence of each duplicate.
      'none': Remove all duplicates.
      Default is None, which keeps the first occurrence.
    - consider_all (bool): Whether to consider all columns in the DataFrame if duplicates are found in the specified columns.
      True means removing the entire row if any duplicates are found in the specified columns. Default is True.
    - inplace (bool): If True, modify the DataFrame in place. If False, return a new DataFrame. Default is False.

    Returns:
    - Optional[pd.DataFrame]: DataFrame with duplicates removed according to specified criteria, or None if inplace=True.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'df' parameter must be a pandas DataFrame.")

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

    if inplace:
        df.update(df_cleaned)
        return None
    else:
        return df_cleaned






import pandas as pd
from typing import Union, List
@log_function_call

def remove_columns(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Remove specified columns from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame from which columns are to be removed.
    - columns (Union[str, List[str]]): A column name or a list of column names to be removed.

    Returns:
    - pd.DataFrame: A DataFrame with the specified columns removed.

    Raises:
    - ValueError: If any of the specified columns do not exist in the DataFrame.
    - TypeError: If columns is not a string or a list of strings.
    """

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
    df_reduced = df.drop(columns=columns)

    return df_reduced








import pandas as pd
import re
from spellchecker import SpellChecker
@log_function_call

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

def auto_rename_columns(df: pd.DataFrame) -> dict:
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

def rename_columns(df: pd.DataFrame) -> None:
    """
    Provide an interactive prompt for users to apply recommended column renaming.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns will be analyzed and potentially renamed.
    """
    # Get column rename recommendations
    rename_recommendations = auto_rename_columns(df)

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

def change_dt(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    date_format: str = "%Y-%m-%d",
    time_format: str = "%H:%M:%S",
    from_timezone: Optional[str] = None,
    to_timezone: Optional[str] = None
) -> pd.DataFrame:
    """
    Change the format of date and time columns in a DataFrame and handle timezone conversion.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing date/time columns to reformat.
    - columns (Union[str, List[str]]): The name(s) of the column(s) to be reformatted.
    - date_format (str): The desired date format (default: "%Y-%m-%d").
    - time_format (str): The desired time format (default: "%H:%M:%S").
    - from_timezone (Optional[str]): The original timezone of the datetime column(s).
      If None, no timezone conversion will be applied (default: None).
    - to_timezone (Optional[str]): The desired timezone for the datetime column(s).
      If None, no timezone conversion will be applied (default: None).

    Returns:
    - pd.DataFrame: A new DataFrame with the date/time columns reformatted.
    """

    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]

    # Check if the provided columns exist in the DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"These columns do not exist in the DataFrame: {missing_columns}")

    # Clone the DataFrame to avoid modifying the original one
    formatted_df = df.copy()

    # Change the date and time format and timezone for each specified column
    for col in columns:
        # Convert to datetime if not already in datetime format
        formatted_df[col] = pd.to_datetime(formatted_df[col], errors='coerce')

        if from_timezone and to_timezone:
            # Convert timezone if both from and to timezones are specified
            formatted_df[col] = (
                formatted_df[col]
                .dt.tz_localize(from_timezone, ambiguous='NaT', nonexistent='NaT')
                .dt.tz_convert(to_timezone)
            )
        elif from_timezone or to_timezone:
            raise ValueError("Both from_timezone and to_timezone must be specified for timezone conversion.")

        # Apply date and time format
        formatted_df[col] = formatted_df[col].dt.strftime(f"{date_format} {time_format}")

    return formatted_df






import pandas as pd
from collections import Counter
@log_function_call

def detect_delimiter(series: pd.Series) -> str:
    """
    Detect the most common delimiter in a Series of strings.

    Parameters:
    - series (pd.Series): A Series containing strings to analyze.

    Returns:
    - str: The most common delimiter found.
    """
    # Define a list of potential delimiters
    potential_delimiters = [',', ';', '|', '\t', ':']

    # Collect all delimiters used in the series
    delimiters = [delimiter for text in series.dropna() for delimiter in potential_delimiters if delimiter in text]

    # If no delimiters are found, return None
    if not delimiters:
        return None

    # Return the most common delimiter
    most_common_delimiter, _ = Counter(delimiters).most_common(1)[0]
    return most_common_delimiter

def split_column(df: pd.DataFrame, column_name: str, delimiter: str = None) -> pd.DataFrame:
    """
    Split a single column into multiple columns based on a delimiter.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column to split.
    - column_name (str): The name of the column to be split.
    - delimiter (str): The delimiter to use for splitting. If None, detect the most common delimiter.

    Returns:
    - pd.DataFrame: A new DataFrame with the specified column split into multiple columns.
    """
    # Validate input
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    if delimiter is None:
        # Detect the delimiter if not provided
        delimiter = detect_delimiter(df[column_name])
        if delimiter is None:
            raise ValueError("No delimiter detected and none provided.")

    # Split the column based on the delimiter
    expanded_columns = df[column_name].str.split(delimiter, expand=True)

    # Drop the original column and concatenate the new columns
    df_expanded = df.drop(columns=[column_name]).join(expanded_columns)

    # Rename new columns with a suffix to identify them
    df_expanded.columns = [f"{column_name}_{i+1}" if i < len(expanded_columns.columns) else col
                           for i, col in enumerate(df_expanded.columns)]

    return df_expanded





import pandas as pd
import numpy as np
@log_function_call

def impute(data, by=None, value=None, columns=None):
    """
    Handle missing values in a DataFrame or Series using the specified strategy, custom value, and columns.

    Parameters:
    data (pd.DataFrame or pd.Series): The DataFrame or Series with missing values (NaN represents missing values).
    by (str or None): The strategy to use for imputing missing values.
                      Options: 'mean', 'median', 'mode', 'interpolate', 'forward_fill', 'backward_fill'.
                      If None, a value must be provided.
    value: A custom value to fill NaNs with. Supports various data types.
    columns (list of str or None): The list of column names to apply the fill operation.
                                   If None, applies to all columns.

    Returns:
    pd.DataFrame or pd.Series: A new DataFrame or Series with missing values handled.
    """
    # Validate input
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Data must be a pandas DataFrame or Series.")

    if isinstance(data, pd.Series):
        data = data.to_frame()

    if columns is None:
        # Apply to all columns if none specified
        columns = data.columns

    # Dictionary to map strategy to pandas methods
    strategies = {
        'mean': lambda col: col.fillna(col.mean()),
        'median': lambda col: col.fillna(col.median()),
        'mode': lambda col: col.fillna(col.mode().iloc[0]),
        'interpolate': lambda col: col.interpolate(),
        'forward_fill': lambda col: col.ffill(),
        'backward_fill': lambda col: col.bfill()
    }

    # Handle missing values
    if by is not None and by not in strategies:
        raise ValueError("Invalid strategy. Use one of ['mean', 'median', 'mode', 'interpolate', 'forward_fill', 'backward_fill'].")

    result = data.copy()

    for col in columns:
        if col not in data.columns:
            raise ValueError(f"Column {col} is not in the DataFrame.")

        if by:
            # Apply the selected strategy
            result[col] = strategies[by](result[col])
        elif value is not None:
            # Use the custom value for filling
            result[col].fillna(value, inplace=True)
        else:
            raise ValueError("Either a strategy ('by') or a value must be provided.")

    return result





import pandas as pd
from spellchecker import SpellChecker

# Initialize SpellCheckers for different dictionaries
spell_checker_dict = {
    'en_US': SpellChecker(language='en'),
    # 'en_GB': SpellChecker(language='en_GB'),
    # 'en_AU': SpellChecker(language='en_AU'),
    # 'en_IE': SpellChecker(language='en_IE')
}
@log_function_call

def spell_check_dataframe(data, dictionary='en_US', columns=None):
    """
    Perform spell check on the specified columns of a DataFrame using the specified dictionary.

    Parameters:
    data (pd.DataFrame): The DataFrame containing columns to spell check.
    dictionary (str): The dictionary to use for spell checking.
                      Options: 'en_US' (American English, default), 'en_GB' (British English),
                               'en_AU' (Australian English), 'en_IE' (Irish English).
    columns (list of str or None): The list of column names to perform spell check on.
                                    If None, performs spell check on all string columns.

    Returns:
    dict: A dictionary where keys are column names and values are lists of misspelled words.
    """

    # Validate input
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")

    if columns is None:
        # Select all object (string) columns if no specific columns are provided
        columns = data.select_dtypes(include=['object']).columns

    if dictionary not in spell_checker_dict:
        raise ValueError("Dictionary must be one of ['en_US', 'en_GB', 'en_AU', 'en_IE'].")

    # Initialize the spell checker with the specified language
    spell_checker = spell_checker_dict[dictionary]

    # Dictionary to store misspelled words per column
    misspelled_words = {}

    # Check spelling for specified columns
    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column {column} is not in the DataFrame.")

        # Extract text from the column
        text_data = data[column].dropna()

        # Find misspelled words
        misspelled = []
        for text in text_data:
            words = text.split()  # Split text into words
            misspelled.extend([word for word in words if word.lower() not in spell_checker])

        # Store misspelled words in the dictionary
        misspelled_words[column] = list(set(misspelled))  # Remove duplicates

    return misspelled_words

def detect_invalid_dates(series):
    """
    Detect invalid date values in a pandas Series.

    Parameters:
    series (pd.Series): The Series to check for invalid dates.

    Returns:
    pd.Series: A boolean Series where True indicates invalid date values.
    """
    try:
        pd.to_datetime(series, errors='raise')
        return pd.Series([False] * len(series), index=series.index)
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors='coerce').isna()

def detect_data_entry_errors(data, spellcheck_dict='en_US', date_columns=None, numeric_columns=None, text_columns=None):
    """
    Detect and flag data entry errors in a DataFrame, including invalid dates and misspelled words.

    Parameters:
    data (pd.DataFrame): The DataFrame to analyze.
    spellcheck_dict (str): The dictionary to use for spell checking.
                           Options: 'en_US' (American English, default), 'en_GB' (British English),
                                    'en_AU' (Australian English), 'en_IE' (Irish English).
    date_columns (list of str or None): The list of columns to check for invalid dates.
    numeric_columns (list of str or None): The list of columns to check for numeric format errors.
    text_columns (list of str or None): The list of text columns to perform spell checking on.

    Returns:
    pd.DataFrame: A DataFrame indicating detected errors with columns: 'Column', 'Error Type', 'Value'.
    """
    errors = []

    # Validate input
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")

    # Check for invalid dates
    if date_columns is None:
        date_columns = data.select_dtypes(include=['object']).columns  # Default to all object columns

    for col in date_columns:
        if col in data.columns:
            invalid_dates = detect_invalid_dates(data[col])
            for idx in invalid_dates[invalid_dates].index:
                errors.append({'Column': col, 'Error Type': 'Invalid Date', 'Value': data.at[idx, col]})

    # Check for invalid numeric formats
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=['object']).columns  # Default to all object columns

    for col in numeric_columns:
        if col in data.columns:
            non_numeric = pd.to_numeric(data[col], errors='coerce').isna() & data[col].notna()
            for idx in non_numeric[non_numeric].index:
                errors.append({'Column': col, 'Error Type': 'Invalid Numeric', 'Value': data.at[idx, col]})

    # Spell check on text columns
    misspelled_words = spell_check_dataframe(data, dictionary=spellcheck_dict, columns=text_columns)

    for col, words in misspelled_words.items():
        for word in words:
            indices = data[col].apply(lambda x: word in x if isinstance(x, str) else False)
            for idx in indices[indices].index:
                errors.append({'Column': col, 'Error Type': 'Misspelled Word', 'Value': word})

    return pd.DataFrame(errors)





import pandas as pd
@log_function_call

def data_type_conversions(data, column=None):
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
    Detect outliers in a dataset using specified method and threshold.

    Parameters:
    data (Union[pd.Series, pd.DataFrame]): The input data to analyze.
    method (str, default='iqr'): The outlier detection method to use. Options include 'z-score' and 'iqr'.
    threshold (float, default=1.5): The threshold for outlier detection.
    columns (List[str], optional): List of columns to apply the outlier detection on. If None, applies to all numerical columns.
    handle_missing (bool, default=True): Whether to handle missing values by dropping them or not.

    Returns:
    pd.DataFrame: A DataFrame where outliers are marked as True and non-outliers are marked as False.
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
    if handle_missing:
        data = data.dropna()

    # Select numerical columns if columns is None
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns

    # Initialize outliers as False for all values
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

    return outliers if handle_missing else data.isna() | outliers



