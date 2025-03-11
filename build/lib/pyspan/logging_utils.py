import inspect
import logging
import functools
import pandas as pd

# Store logs in a list for retrieval
log_entries = []

def log_function_call(func):
    @functools.wraps(func)  # This line preserves the original function's metadata
    def wrapper(*args, **kwargs):
        # Get the arguments passed to the decorated function (args and kwargs)
        arg_names = inspect.getfullargspec(func).args
        arg_vals = args[:len(arg_names)]  # Positional arguments
        kwarg_vals = kwargs  # Keyword arguments

        # Prepare a readable argument list, handling large objects (like DataFrames)
        def readable_repr(val):
            if isinstance(val, pd.DataFrame):
                return f'<DataFrame: {val.shape[0]} rows x {val.shape[1]} columns>'
            return val

        # Combine args and kwargs into a readable string
        arg_repr = ', '.join(f'{name}={readable_repr(val)}' for name, val in zip(arg_names, arg_vals))
        kwarg_repr = ', '.join(f'{key}={readable_repr(value)}' for key, value in kwarg_vals.items())

        # If no arguments are passed
        all_args_repr = f'{arg_repr}, {kwarg_repr}'.strip(', ')

        if not all_args_repr:
            all_args_repr = 'None'  # Show "None" if no arguments are passed
        
        # Get the calling line number
        caller_info = inspect.stack()[1]
        line_number = caller_info.lineno

        # Create a human-readable log entry with line number
        log_entry = f'Function "{func.__name__}" was called at line {line_number} with parameters: {all_args_repr}.'
        
        # Check for duplicates before appending
        if log_entry not in log_entries:
            log_entries.append(log_entry)

        # Optionally also log to a file
        logging.info(log_entry)
        
        return func(*args, **kwargs)
    return wrapper

def display_logs():
    if not log_entries:
        print("No logs available.")
    else:
        for entry in log_entries:
            print(entry) 