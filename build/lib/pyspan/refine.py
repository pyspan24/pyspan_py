import pandas as pd
import numpy as np
import re
import json
from spellchecker import SpellChecker
from typing import Union
from .logging_utils import log_function_call
from .state_management import track_changes

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
    - clean_rows (bool): Whether to clean row data as well as column names.

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