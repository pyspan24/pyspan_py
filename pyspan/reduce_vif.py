import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
from .logging_utils import log_function_call
from .state_management import track_changes

@log_function_call
@track_changes
def reduce_vif(data, threshold: float = 5.0, handle_na: str = 'drop', return_format: str = 'dataframe'):
    """
    Removes multicollinearity using VIF.

    Accepts: DataFrame, ndarray, list, dict, json.

    Parameters:
    ----------
    data : dataset
        Data in any format (DataFrame, numpy array, list, dict, json)

    threshold : float, default=5.0
        VIF threshold for dropping features.

    handle_na : 'drop' or 'fill', default='drop'
        How to handle NaN or inf.

    return_format : 'dataframe', 'dict', 'list', 'json'
        Format of output.

    Returns:
    -------
    Cleaned dataset in chosen format.
    """
    original_type = type(data)

    # Convert to DataFrame
    if isinstance(data, pd.DataFrame):
        df_clean = data.copy()
    elif isinstance(data, np.ndarray):
        df_clean = pd.DataFrame(data)
    elif isinstance(data, list):
        df_clean = pd.DataFrame(data)
    elif isinstance(data, dict):
        df_clean = pd.DataFrame(data)
    else:
        try:
            df_clean = pd.read_json(data)
        except Exception:
            raise ValueError("Unsupported data type provided.")

    # Handle NaN / inf
    if handle_na == 'drop':
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    elif handle_na == 'fill':
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        raise ValueError("handle_na should be either 'drop' or 'fill'")

    # VIF Loop
    while True:
        if df_clean.shape[1] <= 1:
            print("Warning: All columns removed due to high multicollinearity.")
            break

        vif_data = pd.DataFrame()
        vif_data['feature'] = df_clean.columns
        vif_data['VIF'] = [variance_inflation_factor(df_clean.values, i)
                           for i in range(df_clean.shape[1])]

        max_vif = vif_data['VIF'].max()
        feature_to_drop = vif_data.loc[vif_data['VIF'] == max_vif, 'feature'].values[0]

        if max_vif > threshold:
            print(f"Dropping '{feature_to_drop}' due to high VIF: {max_vif:.2f}")
            df_clean = df_clean.drop(columns=[feature_to_drop])
        else:
            break

    # Convert back to requested format
    if return_format == 'dataframe':
        return df_clean
    elif return_format == 'dict':
        return df_clean.to_dict(orient='list')
    elif return_format == 'list':
        return df_clean.values.tolist()
    elif return_format == 'json':
        return df_clean.to_json(orient='records')
    else:
        raise ValueError("return_format must be 'dataframe', 'dict', 'list' or 'json'")