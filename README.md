# pyspan

## 'pyspan' is a Python package designed to facilitate data cleaning and preprocessing using Pandas. It provides various functions to handle missing values, detect outliers, spell check detect_errors, and more. Additionally, it includes a logging utility to keep track of function calls and their parameters.

Installation

To use pyspan, simply install the package using pip:

``bash
pip install pyspan 

# Functions

1. handle_nulls(data: pd.DataFrame,columns: Optional[Union[str, List[str]]] = None,action: str = 'remove',with_val: Optional[Union[int, float, str]] = None,by: Optional[str] = None,inplace: bool = False) -> Optional[pd.DataFrame]
Description
handle_nulls manages null values within the specified columns of a DataFrame using different strategies. You can choose to remove rows with nulls, replace nulls with a custom value, or impute nulls using a predefined strategy (e.g., mean, median, mode).
Parameters:
data: pd.DataFrame
The input DataFrame containing missing values.

columns: Optional[Union[str, List[str]]]
A column name or list of column names where the fill operation should be applied. If not provided, the function will apply to the entire DataFrame.

action: str
The strategy to use for handling missing values. Options include:

'remove': Drops rows containing null values in the specified columns.
'replace': Replaces null values with a specified custom value.
'impute': Imputes null values using a chosen strategy (e.g., mean, median, mode, interpolate, forward fill, backward fill).
with_val: Optional[Union[int, float, str]]
The custom value to replace NaNs with, applicable only if action='replace'.

by: Optional[str]
The strategy to use for imputing null values, applicable only if action='impute'. Available options are:

'mean': Impute with the mean of the column.
'median': Impute with the median of the column.
'mode': Impute with the most frequent value in the column.
'interpolate': Use linear interpolation to impute missing values.
'forward_fill': Propagate the last valid observation forward to the next.
'backward_fill': Use the next valid observation to fill the gaps backward.
inplace: bool
Whether to modify the DataFrame in place. If True, the operation modifies the DataFrame directly and returns None. If False, returns a new DataFrame with the null values handled according to the specified action.

Returns:
Optional[pd.DataFrame]: The DataFrame with handled null values, or None if inplace=True.

2. remove(df: pd.DataFrame, operation: str, columns: Optional[Union[str, List[str]]] = None, keep: Optional[str] = 'first', consider_all: bool = True, inplace: bool = False) -> Optional[pd.DataFrame]:

Description
The remove function handles two types of DataFrame modifications: removing duplicate rows or removing specific columns. The type of operation is specified by the operation parameter.

Parameters
df (pd.DataFrame): The input DataFrame on which the operation will be performed.

operation (str): Specifies the type of removal operation. Options are:

'duplicates': To remove duplicate rows.
'columns': To remove specified columns.
columns (Optional[Union[str, List[str]]]): Column(s) to consider for the operation:

For 'duplicates': Columns to check for duplicates.
For 'columns': Column(s) to be removed.
keep (Optional[str]): Determines which duplicates to keep. Options are:

'first': Keep the first occurrence of each duplicate.
'last': Keep the last occurrence of each duplicate.
'none': Remove all duplicates. Default is 'first'.
consider_all (bool): Relevant only when operation is 'duplicates'. If True, removes the entire row if any duplicates are found in the specified columns. Default is True.

inplace (bool): If True, modifies the DataFrame in place. If False, returns a new DataFrame. Default is False.

Returns
Optional[pd.DataFrame]: DataFrame with duplicates removed or columns removed according to the specified criteria. Returns None if inplace=True.
Raises
ValueError: If invalid columns are specified or operation is invalid.
TypeError: If input types are incorrect.

3. auto_rename_columns(df: pd.DataFrame) -> pd.DataFrame
Automatically renames columns to remove spaces and special characters.
Parameters:
df: DataFrame to rename columns in.

4. rename_dataframe_columns(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame
Renames columns in a DataFrame using a provided dictionary mapping.
Parameters:
df: DataFrame to rename columns in.
rename_dict: Dictionary mapping current column names to new column names.

5. format_dt(df: pd.DataFrame,column_name: str,add_day: bool = False,add_month: bool = False,add_year: bool = False,add_quarter: bool = False,add_hour: bool = False,add_minute: bool = False,add_day_of_week: bool = False,date_format: str = "%Y-%m-%d",time_format: str = "%H:%M:%S",from_timezone: Optional[str] = None,to_timezone: Optional[str] = None) -> pd.DataFrame:

Adds additional date/time-based columns to a DataFrame and formats date/time columns.

Parameters
df (pd.DataFrame): DataFrame to which new date/time features will be added.
column_name: The name of the column containing date/time data.
day: If True, adds a column with the day of the month.
month: If True, adds a column with the month.
year: If True, adds a column with the year.
quarter: If True, adds a column with the quarter of the year.
hour: If True, adds a column with the hour of the day.
minute: If True, adds a column with the minute of the hour.
day_of_week: If True, adds a column with the day of the week (0=Monday, 6=Sunday).
date_format (str): Desired date format (default: "%Y-%m-%d").
time_format (str): Desired time format (default: "%H:%M:%S").
from_timezone (str): Original timezone of the datetime column(s).
to_timezone (str): Desired timezone for the datetime column(s).
Returns
pd.DataFrame: DataFrame with added and formatted date/time features.

6. split_column(df: pd.DataFrame, column_name: str, delimiter: str = None) -> pd.DataFrame
Splits a single column into multiple columns based on a delimiter.
Parameters:
df: DataFrame containing the column to split.
column_name: Name of the column to be split.
delimiter: Delimiter to use for splitting (optional) e.g delimiter (, ;, |).

7. detect_errors(data, spellcheck_dict='en_US', date_columns=None, numeric_columns=None, text_columns=None) -> pd.DataFrame
Detects and flags data entry errors including invalid dates and misspelled words.
Parameters:
data: DataFrame to analyze.
spellcheck_dict: Dictionary to use for spell checking.
date_columns: List of columns to check for invalid dates.
numeric_columns: List of columns to check for numeric format errors.
text_columns: List of text columns to perform spell checking on.

8. convert_type(data, column=None) -> pd.DataFrame or pd.Series
Recommends and applies data type conversions based on the analysis of each column's data.
Parameters:
data: DataFrame or Series to analyze.
column: Specific column to analyze (optional).

9. detect_outliers(data, method='iqr', threshold=1.5, columns=None, handle_missing=True) -> pd.DataFrame
Detects outliers in a dataset using specified method and threshold.
Parameters:
data: DataFrame or Series to analyze.
method: Outlier detection method ('z-score', 'iqr').
threshold: Threshold for outlier detection.
columns: List of columns to apply the outlier detection on (optional).
handle_missing: Whether to handle missing values by dropping them or not.

10. Logging Functions
display_logs()
Prints stored log entries.

11. remove_chars(text, remove_multiple_spaces=False, custom_characters=None):
The remove_chars function is used to clean and format text by trimming leading and trailing spaces, handling multiple spaces within the text, and optionally removing custom characters.

Parameters
text (str): The input string to be cleaned.
remove_multiple_spaces (bool): If True, all extra spaces within the text will be removed. Otherwise, only leading, trailing, and extra spaces between words will be reduced to a single space. Default is False.
custom_characters (str or None): A string of characters to be removed from the text. If None, no custom characters will be removed. Default is None.
Returns
str: The cleaned text with appropriate spaces and optional custom characters removed.

12. reformat(df, target_column, reference_column)

The reformat function applies the data type and formatting from a reference column to a target column in the same DataFrame. It supports data types such as datetime, numeric, and string. For string columns, it also applies formatting such as uppercase, lowercase, or title case based on the reference column.
Parameters
df (pd.DataFrame): The DataFrame containing both the target and reference columns.
target_column (str): The name of the column to format.
reference_column (str): The name of the column to borrow formatting from.
Returns
pd.DataFrame: The DataFrame with the target column formatted based on the reference column.
Raises
ValueError: If the target column or reference column does not exist in the DataFrame.
TypeError: If the reference column is not of a type that can be applied to the target column or if the data type is unsupported for formatting.

13. scale_data(df, method='minmax', columns=None)
Scales the data in a DataFrame using one of the specified scaling methods: Min-Max Scaling, Robust Scaling, or Standard Scaling.

Parameters:
df (pd.DataFrame): The DataFrame containing the data to be scaled.
method (str): The scaling method to use. Options are:
'minmax': Min-Max Scaling (default)
'robust': Robust Scaling
'standard': Standard Scaling
columns (list, optional): List of column names to apply scaling on. If None, scales all numerical columns.
Returns:
pd.DataFrame: A DataFrame with the specified columns scaled according to the selected method.


## Example Usage
Here are some examples to illustrate the usage of the functions provided in pyspan:

import pandas as pd
from pyspan import handle_nulls, remove, auto_rename_columns, rename_dataframe_columns
from pyspan import change_dt, split_column,
from pyspan import detect_errors, convert_type, detect_outliers
from pyspan import display_logs, clean_spaces, remove_chars, reformat

# Load a dataset
df = pd.read_csv('/content/GlobalSharkAttacks.csv')

# Example usage of handle_nulls
# 1. Remove rows with null values in the column
df_cleaned_remove = handle_nulls(df, columns='column ', action='remove', inplace=False)

# 2. Replace null values in the column with 'Unknown'
df_cleaned_replace = handle_nulls(df, columns='column ', action='replace', with_val='Unknown', inplace=False)

# 3. Impute null values in the column using the mode
df_cleaned_impute_mode = handle_nulls(df, columns='column ', action='impute', impute_strategy='mode', inplace=False)

# Example usage of remove
# Remove duplicate rows based on column 
df_no_duplicates = remove(df, operation='duplicates', columns='Type', keep='first', inplace=False)

# Remove specified columns
df_no_columns = remove(df, operation='columns', columns=['Type','Date'], inplace=False)

# Example usage of auto_rename_columns
auto_rename_columns(df)

# Example usage of rename_dataframe_columns
rename_dict = {'OldName': 'NewName'}
df_renamed_dict = rename_dataframe_columns(df, rename_dict)

# Example usage of format_dt
# Apply the function to add features and format the 'timestamp' column
processed_df = format_dt(df,column_name='timestamp',day=True,month=True,year=True,quarter=True,hour=True,minute=True,day_of_week=True,
date_format="%d-%m-%Y",time_format="%I:%M %p",from_timezone='UTC',to_timezone='America/New_York')

# Example usage of split_column
df_split = split_column(df, column_name='ColumnName', delimiter=','--> Optional)

# Example usage of detect_errors
errors = detect_errors(df, spellcheck_dict='en_US', date_columns=['DateColumn'], numeric_columns=['NumericColumn'], text_columns=['TextColumn'])

# Example usage of convert_type
df_converted = convert_type(df)

# Example usage of detect_outliers
outliers = detect_outliers(df, method='iqr', threshold=1.5)

# Example usage of display_logs
display_logs()


# Example usage of remove_chars
## Apply the remove_chars function to the 'Name' column
df['Name'] = df['Name'].apply(remove_chars) 

## Apply the remove_chars function to all string columns in the DataFrame
df = df.applymap(lambda x: remove_chars(x) if isinstance(x, str) else x)

# Apply the remove_chars function with remove_multiple_spaces=True to all string columns
df = df.applymap(lambda x: remove_chars(x, remove_multiple_spaces=True) if isinstance(x, str) else x)

# Apply the remove_chars function with custom characters (e.g., spaces, !, : etc) to all string columns
df = df.applymap(lambda x: remove_chars(x, custom_characters=' ') if isinstance(x, str) else x)

# Explanation of applymap with lambda x:
applymap(lambda x: clean_spaces(x) if isinstance(x, str) else x): This command applies the clean_spaces function to every element in the DataFrame. It checks if each element x is a string (isinstance(x, str)), and if so, it cleans it. Non-string elements remain unchanged.

# Example usage of reformat
# Apply formatting from 'target_column' to 'reference_column' 
df = reformat(df, 'target_column', 'reference_column')

# Example usage of scale_data
# Apply Min-Max Scaling to columns 'A' and 'B'
df_scaled = scale_data(df, method='minmax', columns=['A', 'B'])

# License
This package is licensed under the MIT License. See the LICENSE file for more details.

# Contact
For issues or questions, please contact [amynahreimoo@gmail.com].

