import pandas as pd
from spellchecker import SpellChecker
import re
import json
import numpy as np
from typing import Union, List, Optional
from .logging_utils import log_function_call
from .state_management import track_changes

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