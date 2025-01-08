import pandas as pd
import numpy as np
import json
from typing import Union, List, Optional
from .logging_utils import log_function_call
from .state_management import track_changes

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