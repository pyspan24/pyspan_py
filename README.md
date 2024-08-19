# pyspan

## 'pyspan' is a Python package designed to facilitate data cleaning and preprocessing using Pandas. It provides various functions to handle missing values, detect outliers, spell check data, and more. Additionally, it includes a logging utility to keep track of function calls and their parameters.

Installation

To use pyspan, simply install the package using pip:

``bash
pip install pyspan 

# Functions

1. handle_nulls(data: pd.DataFrame, columns: List[str], method: str, value: Optional[Union[str, float]] = None) -> pd.DataFrame
Handles missing values in the specified columns of a DataFrame.
Parameters:
data: DataFrame with missing values.
columns: List of column names to apply the fill operation.
method: Strategy to use for imputing missing values ('mean', 'median', 'mode', 'interpolate', 'forward_fill', 'backward_fill').
value: Custom value to fill NaNs with (optional).

2. remove_duplicates(df: pd.DataFrame) -> pd.DataFrame
Removes duplicate rows from a DataFrame.
Parameters:
df: DataFrame to remove duplicates from.

3. remove_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame
Removes specified columns from a DataFrame.
Parameters:
df: DataFrame to remove columns from.
columns: List of column names to remove.

4. auto_rename_columns(df: pd.DataFrame) -> pd.DataFrame
Automatically renames columns to remove spaces and special characters.
Parameters:
df: DataFrame to rename columns in.
And call the this function like this rename_columns.

5. rename_dataframe_columns(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame
Renames columns in a DataFrame using a provided dictionary mapping.
Parameters:
df: DataFrame to rename columns in.
rename_dict: Dictionary mapping current column names to new column names.

6. change_dt(df: pd.DataFrame, columns: Union[str, List[str]], date_format: str = "%Y-%m-%d", time_format: str = "%H:%M:%S", from_timezone: Optional[str] = None, to_timezone: Optional[str] = None) -> pd.DataFrame
Changes the format of date and time columns and handles timezone conversion.
Parameters:
df: DataFrame containing date/time columns to reformat.
columns: Name(s) of the column(s) to be reformatted.
date_format: Desired date format.
time_format: Desired time format.
from_timezone: Original timezone of the datetime column(s).
to_timezone: Desired timezone for the datetime column(s).

7. detect_delimiter(series: pd.Series) -> str
Detects the most common delimiter in a Series of strings.
Parameters:
series: Series containing strings to analyze.

8. split_column(df: pd.DataFrame, column_name: str, delimiter: str = None) -> pd.DataFrame
Splits a single column into multiple columns based on a delimiter.
Parameters:
df: DataFrame containing the column to split.
column_name: Name of the column to be split.
delimiter: Delimiter to use for splitting (optional).

9. impute(data, by=None, value=None, columns=None) -> pd.DataFrame
Handles missing values using specified strategy or custom value.
Parameters:
data: DataFrame or Series with missing values.
by: Strategy for imputing missing values ('mean', 'median', 'mode', 'interpolate', 'forward_fill', 'backward_fill').
value: Custom value to fill NaNs with.
columns: List of column names to apply the fill operation.

10. spell_check_dataframe(data: pd.DataFrame, dictionary='en_US', columns=None) -> dict
Performs spell check on specified columns of a DataFrame.
Parameters:
data: DataFrame containing columns to spell check.
dictionary: Dictionary to use for spell checking ('en_US', 'en_GB', 'en_AU', 'en_IE').
columns: List of column names to perform spell check on.

11. detect_invalid_dates(series: pd.Series) -> pd.Series
Detects invalid date values in a Series.
Parameters:
series: Series to check for invalid dates.

12. detect_data_entry_errors(data, spellcheck_dict='en_US', date_columns=None, numeric_columns=None, text_columns=None) -> pd.DataFrame
Detects and flags data entry errors including invalid dates and misspelled words.
Parameters:
data: DataFrame to analyze.
spellcheck_dict: Dictionary to use for spell checking.
date_columns: List of columns to check for invalid dates.
numeric_columns: List of columns to check for numeric format errors.
text_columns: List of text columns to perform spell checking on.

13. data_type_conversions(data, column=None) -> pd.DataFrame or pd.Series
Recommends and applies data type conversions based on the analysis of each column's data.
Parameters:
data: DataFrame or Series to analyze.
column: Specific column to analyze (optional).

14. detect_outliers(data, method='iqr', threshold=1.5, columns=None, handle_missing=True) -> pd.DataFrame
Detects outliers in a dataset using specified method and threshold.
Parameters:
data: DataFrame or Series to analyze.
method: Outlier detection method ('z-score', 'iqr').
threshold: Threshold for outlier detection.
columns: List of columns to apply the outlier detection on (optional).
handle_missing: Whether to handle missing values by dropping them or not.

Logging Functions
1. display_logs()
Prints stored log entries.


## Example Usage
Here are some examples to illustrate the usage of the functions provided in pyspan:

import pandas as pd
from pyspan import handle_nulls, remove_duplicates, remove_columns, auto_rename_columns, rename_dataframe_columns
from pyspan import change_dt, detect_delimiter, split_column, impute, spell_check_dataframe
from pyspan import detect_invalid_dates, detect_data_entry_errors, data_type_conversions, detect_outliers
from pyspan import log_function_call, display_logs

# Load a dataset
df = pd.read_csv('/content/GlobalSharkAttacks.csv')

# Example usage of handle_nulls
df_filled = handle_nulls(df, columns=['Column1', 'Column2'], method='mean')

# Example usage of remove_duplicates
df_unique = remove_duplicates(df)

# Example usage of remove_columns
df_reduced = remove_columns(df, columns=['ColumnToRemove'])

# Example usage of auto_rename_columns
# Run the renaming function
rename_columns(df)

# Example usage of rename_dataframe_columns
rename_dict = {'OldName': 'NewName'}
df_renamed_dict = rename_dataframe_columns(df, rename_dict)

# Example usage of change_dt
df_formatted = change_dt(df, columns=['Date'], date_format='%d-%m-%Y', time_format='%I:%M %p')

# Example usage of detect_delimiter
delimiter = detect_delimiter(df['ColumnWithDelimiters'])

# Example usage of split_column
df_split = split_column(df, column_name='ColumnWithDelimiters')

# Example usage of impute
df_imputed = impute(df, by='mean', columns=['Column1'])

# Example usage of spell_check_dataframe
misspelled = spell_check_dataframe(df, dictionary='en_US', columns=['TextColumn'])

# Example usage of detect_invalid_dates
invalid_dates = detect_invalid_dates(df['DateColumn'])

# Example usage of detect_data_entry_errors
errors = detect_data_entry_errors(df, spellcheck_dict='en_US', date_columns=['DateColumn'], numeric_columns=['NumericColumn'])

# Example usage of data_type_conversions
df_converted = data_type_conversions(df)

# Example usage of detect_outliers
outliers = detect_outliers(df, method='iqr', threshold=1.5)

# Example usage of display_logs
display_logs()


# License
This package is licensed under the MIT License. See the LICENSE file for more details.

# Contact
For issues or questions, please contact [amynahreimoo@gmail.com].

