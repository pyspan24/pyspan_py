import pandas as pd
import numpy as np
import json
from typing import Union, Optional, List
from .logging_utils import log_function_call
from .state_management import track_changes

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