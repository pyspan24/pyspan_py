# Pyspan Overview 

🎉🎉🎉 Welcome to **pyspan**! 🎉🎉🎉

**pyspan** is a Python package designed to simplify data cleaning and preprocessing tasks, especially for Pandas DataFrames. It offers a range of functions to enhance data handling, including:

- Handling missing values
- Detecting outliers and anomalies
- Performing spell checks and error identification
- Logging function calls and parameters efficiently
- Undo functionality for reverting recent changes made to the data

**Key Features:**

- **Pandas Support (Optional):** You can use **pyspan** with or without Pandas. It supports data passed in formats such as **list**, **string**, **tuple**, **dict**, or **JSON**. This makes it versatile for different data sources and formats.
- **Undo Changes:** Need to undo the last changes made to your data? **pyspan** includes an undo feature to help you revert the most recent modifications.

Whether you're working with a Pandas DataFrame or any other data structure, **pyspan** helps streamline your data preprocessing tasks, making it easier to clean, preprocess, and analyze your data.


🚀  *Installation Process*
To get started with **pyspan**, install the package using pip:

```bash
pip install pyspan 
```

## Functions

## 1. `handle_nulls`

**Description**  
The `handle_nulls` function handles missing values in a DataFrame by offering several options: removing rows or columns with nulls, replacing nulls with a custom value, or imputing them using various methods like mean, median, or mode. This function can also drop rows or columns exceeding a specified threshold of null values.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  The input DataFrame containing missing values.

- **`columns`** (`Optional[Union[str, List[str]]]`, default: `None`):  
  A single column name or a list of column names to apply the operation. If not provided, the function applies to the entire DataFrame.

- **`action`** (`str`, default: `'remove'`):  
  The strategy to handle missing values. Options include:
  - `'remove'`: Drops rows with null values in the specified columns.
  - `'replace'`: Replaces null values with a specified custom value.
  - `'impute'`: Imputes missing values using a chosen strategy (e.g., mean, median, mode, interpolate, forward fill, backward fill).

- **`with_val`** (`Optional[Union[int, float, str]]`, default: `None`):  
  The custom value to replace NaNs with, applicable only if `action='replace'`.

- **`by`** (`Optional[str]`, default: `None`):  
  The strategy for imputing missing values, applicable only if `action='impute'`. Available options are:
  - `'mean'`: Impute with the mean of the column.
  - `'median'`: Impute with the median of the column.
  - `'mode'`: Impute with the most frequent value in the column.
  - `'interpolate'`: Use linear interpolation to impute missing values.
  - `'forward_fill'`: Propagate the last valid observation forward.
  - `'backward_fill'`: Use the next valid observation to fill the gaps backward.

- **`inplace`** (`bool`, default: `False`):  
  If `True`, modifies the DataFrame in place and returns `None`. If `False`, returns a new DataFrame with null values handled according to the specified action.

- **`threshold`** (`Optional[float]`, default: `None`): 
  Sets a threshold (0-100) for the percentage of missing values. If a row or column exceeds this threshold, it will be dropped. Ignored if None.

- **`axis `** (`str`, default: `'rows'`):
  Specifies whether to apply the threshold to `'rows'` or `'columns'`. 

**Returns:**  
- `Optional[pd.DataFrame, List, dict, str]`: The Data with null values handled, or `None` if `inplace=True`.

---

## 2. `remove`

**Description**  
The `remove` function handles two types of DataFrame modifications: removing duplicate rows or removing specific columns. The type of operation is specified by the `operation` parameter.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  The input DataFrame on which the operation will be performed.

- **`operation`** (`str`):  
  Specifies the type of removal operation. Options are:
  - `'duplicates'`: To remove duplicate rows.
  - `'columns'`: To remove specified columns.

- **`columns`** (`Optional[Union[str, List[str]]]`, default: `None`):  
  Columns to consider for the operation:
  - For `'duplicates'`: Columns to check for duplicates.
  - For `'columns'`: Columns to be removed.

- **`keep`** (`Optional[str]`, default: `'first'`):  
  Determines which duplicates to keep. Options are:
  - `'first'`: Keep the first occurrence of each duplicate.
  - `'last'`: Keep the last occurrence of each duplicate.
  - `'none'`: Remove all duplicates.

- **`consider_all`** (`bool`, default: `True`):  
  Relevant only when `operation` is `'duplicates'`. If `True`, removes the entire row if any duplicates are found in the specified columns.

- **`inplace`** (`bool`, default: `False`):  
  If `True`, modifies the DataFrame in place. If `False`, returns a new DataFrame.

**Returns:**  
- `Optional[pd.DataFrame, List, dict, str]`: Data with duplicates removed or columns removed according to the specified criteria. Returns `None` if `inplace=True`.

**Raises:**
- `ValueError`: If invalid columns are specified or operation is invalid.
- `TypeError`: If input types are incorrect.

---

## 3. `refine`

**Description**  
The refine function is designed to clean and standardize a DataFrame by recommending readable column names and optionally cleaning row data. It suggests renaming columns to remove special characters and applies consistent formatting. Additionally, it can clean row data by removing unwanted characters and extra spaces.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  DataFrame to rename columns in.

- **`clean_rows`** (`bool`, default: `True`):
  If True, the function also cleans each row content by removing special characters and extra spaces.

**Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: Data with columns renamed.

---

## 4. `manual_rename_columns`

**Description**  
Renames columns in a DataFrame using a provided dictionary mapping.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  DataFrame to rename columns in.

- **`rename_dict`** (`dict`):  
  Dictionary mapping current column names to new column names.
  
  **Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: Data with columns renamed according to `rename_dict`.

---

## 5. `format_dt`

**Description**  
Adds additional date/time-based columns to a DataFrame and formats date/time columns.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  DataFrame to which new date/time features will be added.

- **`columns`** (`str`):  
  The name of the column containing date/time data.

- **`day`** (`bool`, default: `False`):  
  If `True`, adds a column with the day of the month.

- **`month`** (`bool`, default: `False`):  
  If `True`, adds a column with the month.

- **`year`** (`bool`, default: `False`):  
  If `True`, adds a column with the year.

- **`quarter`** (`bool`, default: `False`):  
  If `True`, adds a column with the quarter of the year.

- **`hour`** (`bool`, default: `False`):  
  If `True`, adds a column with the hour of the day.

- **`minute`** (`bool`, default: `False`):  
  If `True`, adds a column with the minute of the hour.

- **`day_of_week`** (`bool`, default: `False`):  
  If `True`, adds a column with the day of the week (i.e Monday, Sunday).

- **`date_format`** (`str`, default: `"%Y-%m-%d"`):  
  Desired date format.

- **`time_format`** (`str`, default: `"%H:%M:%S"`):  
  Desired time format.

- **`from_timezone`** (`Optional[str]`, default: `None`):  
  Original timezone of the datetime column(s).

- **`to_timezone`** (`Optional[str]`, default: `None`):  
  Desired timezone for the datetime column(s).

**Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: Data with added and formatted date/time features.

---

## 6. `split_column`

**Description**  
Splits a single column into multiple columns based on a delimiter.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  DataFrame containing the column to split.

- **`column`** (`str`):  
  Name of the column to be split.

- **`delimiter`** (`str`, default: `None`):  
  Delimiter to use for splitting (optional), e.g., `,`, `;`, `|`.

**Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: Data with the specified column split into multiple columns.

---

## 7. `detect_errors`

**Description**  
Detects and flags data entry errors including invalid dates and misspelled words.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  DataFrame to analyze.

- **`date_columns`** (`Optional[List[str]]`, default: `None`):  
  List of columns to check for invalid dates.

- **`numeric_columns`** (`Optional[List[str]]`, default: `None`):  
  List of columns to check for numeric format errors.

- **`text_columns`** (`Optional[List[str]]`, default: `None`):  
  List of text columns to perform spell checking on.

- **`date_format`** (`str`, default: `'%Y-%m-%d'`):  
  Expected date format for validation. 

**Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: Data with detected errors flagged.

---

## 8. `convert_type`

**Description**  
Recommends and applies data type conversions based on the analysis of each column's data.

**Parameters:**

- **`df`** (`pd.DataFrame` or `pd.Series`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  DataFrame or Series to analyze.

- **`columns`** (`Optional[List[str]]`, default: `None`):  
  Specific column to analyze (optional).

**Returns:**  
- `pd.DataFrame` or `pd.Series`, `List`, `dict`,`str`: Data or Series with recommended data type conversions applied.

---

## 9. `detect_outliers`

**Description**  
The `detect_outliers` function identifies outliers in a specified column of a DataFrame using traditional methods like Interquartile Range (IQR) and Z-Score, or advanced anomaly detection algorithms (Isolation Forest, Local Outlier Factor, and DBSCAN). The function removes outliers and prints the data's shape before and after outlier removal.

**Parameters:**

- **`df`** (`pd.DataFrame` or `pd.Series`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  DataFrame or Series to analyze.

- **`method`** (`str`, default: `'iqr'`):  
  Outlier detection method. Options are:
  - `'z-score'`: Using z-score.
  - `'iqr'`: Using interquartile range.

- **`threshold`** (`float`, default: `1.5`):  
  Threshold for outlier detection.

- **`columns`** (`Optional[List[str]]`, default: `None`):  
  List of columns or single column to apply outlier or anomalies detection on.

- **`handle_missing`** (`bool`, default: `True`):  
  Whether to handle missing values by dropping them.

- **`anomaly_method`** (`Optional[str]`, default: `None`): 
  Specifies an anomaly detection model. If provided, it overrides the method parameter. Options are:
  - `'isolation_forest'`: Uses the Isolation Forest model.
  - `'lof'`: Uses the Local Outlier Factor model.
  - `'dbscan'`: Uses the DBSCAN clustering algorithm.

- **`contamination`** (`float`, default: `0.05`):
  Proportion of outliers in the data for `'isolation_forest'` and `'lof'`.

- **`n_neighbors`** (`int`, default: `20`):  
  Number of neighbors for the Local Outlier Factor model.

- **`eps`** (`float`, default: `0.5`):
  Maximum distance for DBSCAN clustering to group points as neighbors.

- **`min_samples`** (`int`, default: `5`):
  Minimum samples for DBSCAN to define a cluster.


**Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: Data with `outliers` or `anomalies` detected.

---

## 10. `Logging Functions`

### `display_logs`

**Description**  
Display all used functions details log enteries.

**Returns:**  
- `None`

---

## 11. `remove_chars`

**Description**  
Cleans and formats text in specified columns of a DataFrame by trimming spaces and removing custom characters.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  The DataFrame containing the columns to be cleaned.

- **`columns`** (`Optional[List[str]]`):  
  List of column names to apply the cleaning function.

- **`strip_all`** (`bool`, default: `False`):  
  If `True`, removes all extra spaces within the text.

- **`custom_characters`** (`Optional[str]`, default: `None`):  
  String of custom characters to be removed.

**Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: Data with specified columns cleaned.

---

## 12. `reformat`

**Description**  
Applies the data type and formatting from a reference column to a target column in the same DataFrame.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  The DataFrame containing both the target and reference columns.

- **`target_column`** (`str`):  
  The name of the column to format.

- **`reference_column`** (`str`):  
  The name of the column to borrow formatting from.

**Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: Data with the target column formatted based on the reference column.

**Raises:**
- `ValueError`: If the target or reference column does not exist.
- `TypeError`: If the reference column's type cannot be applied to the target column or if the data type is unsupported.

---

## 13. `scale_data`

**Description**  
Scales data in a DataFrame using specified scaling methods: Min-Max Scaling, Robust Scaling, or Standard Scaling.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
  The DataFrame containing the data to be scaled.

- **`method`** (`str`, default: `'minmax'`):  
  Scaling method to use. Options are:
  - `'minmax'`: Min-Max Scaling.
  - `'robust'`: Robust Scaling.
  - `'standard'`: Standard Scaling.

- **`columns`** (`Optional[List[str]]`, default: `None`):  
  List of column names to apply scaling on. If `None`, scales all numerical columns.

**Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: Data with specified columns scaled according to the selected method.

---

## 14. `undo`

**Description**  
Reverts a DataFrame to the previous state by undoing the most recent modification. The undo function works in conjunction with the track_changes decorator, which saves the current DataFrame state before any changes are made.

**Parameters:**
No parameters are passed to the undo function, but it operates on a globally tracked DataFrame (_df) and its history.
Returns:

**Returns:**  
- `pd.DataFrame`, `List`, `dict`,`str`: The Data in its previous state before the most recent modification.

**Note:** 
If no changes have been made, undo will print a message saying there are no recent changes to undo.

---

## 15. `sample_data`

**Description**
Loads the sample_data, a dataset containing various customer sales records. This dataset includes columns with missing values, inconsistent formats, and outliers, providing an opportunity for data cleaning and manipulation.

**Parameters:**
This function does not take any parameters.

**Returns:**

-`pd.DataFrame`: A DataFrame containing the Customer Sales Data with the following columns:
-`CustomerID`: Unique identifier for each customer.
-`Age`: Age of the customer (may include missing values and inconsistent formats).
-`Gender`: Gender of the customer (may include missing values).
-`PurchaseHistory`: List of previous purchases made by the customer.
-`ProductCategory`: Category of the product purchased (e.g., "Electronics", "Apparel").
-`PurchaseDate`: Date of the purchase (may include inconsistent formats).
-`AmountSpent`: Total amount spent on the purchase, including outliers.
-`PaymentMethod`: Method of payment used (e.g., "Credit Card", "Cash"; includes mixed data types).
-`Country`: Country of the customer.
-`MembershipStatus`: Membership status of the customer (e.g., "Regular", "Gold", "Platinum"; may include missing values).
-`PhoneNumber`: Phone number of the customer, with various formats.
-`DiscountCode`: Discount code used in the transaction, with duplicates included.

**Note:**
This dataset contains a variety of messy data, such as inconsistent date formats, mixed data types, missing values, and outliers, making it suitable for testing data cleaning and preprocessing techniques.

---

## 16. `convert_unit`

**Description**
The convert_unit function is designed to detect and convert units within a specific column of a Pandas DataFrame. This function supports conversions across multiple categories like length, mass, volume, temperature, and more. It efficiently handles data conversions while accounting for units with inconsistent formats or missing values. This function is useful for data cleaning and preprocessing, ensuring consistency across datasets with mixed unit formats.

**Parameters:**

- **`df`** (`pd.DataFrame`, `List`, `dict`, `tuple`, `np.ndarray`, `str`):  
The DataFrame containing the data for unit conversion.

-**`columns`** (`Optional[List[str]]`):
The name of the columns where unit conversion should be applied.

-**`unit_category`** (`str`):
The category of units for conversion (e.g., 'length', 'mass', 'temperature').

-**`from_unit`** (`str`):
The unit of the values to be converted (e.g., 'cm', 'kg').

-**`to_unit`** (`str`):
The target unit for the conversion (e.g., 'm', 'g').

**Returns:**

-`pd.DataFrame`, `List`, `dict`,`str`: A new Data with the values in the specified column converted to the target unit. It also retains any non-numeric values in the original column.

**Supported Unit Categories and Examples:**

-`length`: 'mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi'
-`mass`: 'mg', 'g', 'kg', 'ton', 'lb', 'oz'
-`time`: 's', 'min', 'h', 'day', 'week', 'month', 'year'
-`volume`: 'ml', 'l', 'm3', 'ft3', 'gal'
-`temperature`: 'C', 'F', 'K'
-`speed`: 'm/s', 'km/h', 'mph'
-`energy`: 'J', 'kJ', 'cal'
-`area`: 'm2', 'km2', 'ft2'
-`pressure`: 'Pa', 'bar', 'atm', 'psi'

---

## Example Usage

Here are some examples to illustrate the usage of the functions provided in `pyspan`:

```python
import pandas as pd
from pyspan import handle_nulls, remove, refine, manual_rename_columns
from pyspan import format_dt, split_column, detect_errors, convert_type, detect_outliers
from pyspan import display_logs, remove_chars, reformat, scale_data, undo
from pyspan import sample_data, convert_unit

# Load a dataset
df = pd.read_csv('/content/GlobalSharkAttacks.csv')

# Example usage of handle_nulls
# 1. Remove rows with null values in the column
df_cleaned_remove = handle_nulls(df, columns='column ', action='remove', inplace=False)

# 2. Replace null values in the column with 'Unknown'
df_cleaned_replace = handle_nulls(df, columns='column ', action='replace', with_val='Unknown', inplace=False)

# 3. Impute null values in the column using the mode
df_cleaned_impute_mode = handle_nulls(df, columns='column ', action='impute', by='mode', inplace=False)

# 4. For rows
df_cleaned = handle_nulls(df, threshold=50, axis='rows')

# 5. For columns
df_cleaned = handle_nulls(df, threshold=30, axis='columns')

# Example usage of remove
# Remove duplicate rows based on column 
df_no_duplicates = remove(df, operation='duplicates', columns='Type', keep='first', inplace=False)

# Remove specified columns
df_no_columns = remove(df, operation='columns', columns=['Type', 'Date'], inplace=False)

# Example usage of refine
# Refine DataFrame columns
df_cleaned = refine(df)

# Refine DataFrame with column and row cleaning
df_cleaned = refine(df, clean_rows=True)

# Example usage of manual_rename_columns
rename_dict = {'OldName': 'NewName'}
df_renamed_dict = manual_rename_columns(df, rename_dict)

# Example usage of format_dt
# Apply the function to add features and format the 'timestamp' column
processed_df = format_dt(df, columns='timestamp', day=True, month=True, year=True, quarter=True,
                          hour=True, minute=True, day_of_week=True, date_format="%d-%m-%Y",
                          time_format="%I:%M %p", from_timezone='UTC', to_timezone='America/New_York')

# Example usage of split_column
df_split = split_column(df, column='ColumnName', delimiter=',')

# Example usage of detect_errors
errors = detect_errors(df, date_columns=['DateColumn'], numeric_columns=['NumericColumn'], text_columns=['TextColumn'])

# Example usage of convert_type
df_converted = convert_type(df)

# Example usage of detect_outliers
# Detect outliers using IQR for multiple columns
cleaned_data_iqr = detect_outliers(df, columns=['column', 'column'], method='iqr', threshold=1.5)

# Detect anomalies using Isolation Forest for multiple columns
cleaned_data_isolation_forest = detect_outliers(df, columns=['column', 'column'], anomaly_method='isolation_forest', contamination=0.05)

# Detect anomalies using Local Outlier Factor (LOF)
cleaned_data_lof = detect_outliers(df, columns='column', anomaly_method='lof', contamination=0.05, n_neighbors=20)

# Detect anomalies using DBSCAN
cleaned_data_dbscan = detect_outliers(df, columns='column', anomaly_method='dbscan', eps=0.5, min_samples=5)

# Example usage of display_logs
display_logs()

# Example usage of remove_chars
# Apply the remove_chars function to the 'Column' (keeping one space between words)
df_cleaned_spaces = remove_chars(df, columns=['Column'], strip_all=False)

# Apply the remove_chars function with strip_all=True to the 'Column' (removing all extra spaces)
df_cleaned_strip_all = remove_chars(df, columns=['Column'], strip_all=True)

# Apply the remove_chars function with custom characters (e.g., spaces) to the 'XYZ' columns
df_cleaned_custom_chars = remove_chars(df, columns=['Column'], custom_characters=' ')

# Example usage of reformat
# Apply formatting from 'reference_column' to 'target_column'
df_reformatted = reformat(df, target_column='target_column', reference_column='reference_column')

# Example usage of scale_data
# Apply Min-Max Scaling to columns 'A' and 'B'
df_scaled = scale_data(df, method='min-max', columns=['A', 'B'])

# Example usage of undo
# Undo the most recent change
df = undo()

# Example usage of Dataset 'sample_data'
# Load the Sample Data dataset
df = sample_data()

# Example usage of convert_unit
# Convert 'Distance' column from meters to nautical miles
converted_df = convert_unit(df, column='Distance', unit_category='length', from_unit='m', to_unit='nmi',)

# Convert 'Temperature' column from Fahrenheit to Kelvin
converted_df_temp = convert_unit(df, column='Temperature', unit_category='temperature', from_unit='F', to_unit='K')
```
---

## License

---

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact

For issues or questions, please contact [amynahreimoo@gmail.com](mailto:amynahreimoo@gmail.com).

---

