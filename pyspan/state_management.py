import pandas as pd
import copy
import functools
from .logging_utils import log_function_call

# Global variables to track DataFrame and its history
_df = None
_history = []

def save_state(df):
    """
    Save the current state of the DataFrame before modification.
    """
    global _history
    _history.append(copy.deepcopy(df))

@log_function_call
def undo():
    """
    Undo the most recent change to the DataFrame.
    """
    global _df, _history
    if _history:
        _df = _history.pop()
    else:
        print("No recent changes to undo!")
    return _df

# Decorator to track changes and save the previous state
def track_changes(func):
    @functools.wraps(func)
    def wrapper(df, *args, **kwargs):
        save_state(df)
        result = func(df, *args, **kwargs)
        global _df
        _df = result
        return result
    return wrapper 