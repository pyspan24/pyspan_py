import pandas as pd
import numpy as np
import re
import json
from typing import Union, List, Optional
from .logging_utils import log_function_call
from .state_management import track_changes

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