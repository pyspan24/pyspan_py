import numpy as np
import pandas as pd
import pytz
import json
from typing import Union, List, Optional
from .logging_utils import log_function_call
from .state_management import track_changes

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
