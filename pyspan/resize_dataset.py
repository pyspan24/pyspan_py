import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
import random
import string
from datetime import datetime, timedelta
import re
from .logging_utils import log_function_call
from .state_management import track_changes

class ResizeError(Exception):
    """Custom exception for resize_dataset function errors"""
    pass

def _generate_synthetic_numeric(column: pd.Series, n_samples: int) -> np.ndarray:
    """Generate synthetic numeric data based on the distribution of the original column"""
    mean = column.mean()
    std = column.std()
    if pd.isna(std) or std == 0:
        # If std is NA or 0, just use the mean with small random noise
        return np.random.normal(mean, 0.01 * abs(mean) if mean != 0 else 0.01, n_samples)
    # Generate data using normal distribution based on mean and std
    return np.random.normal(mean, std, n_samples)

def _generate_synthetic_categorical(column: pd.Series, n_samples: int) -> np.ndarray:
    """Generate synthetic categorical data based on the distribution of the original column"""
    # Get value counts and calculate probabilities
    value_counts = column.value_counts(normalize=True)
    # Sample from the original values using the calculated probabilities
    return np.random.choice(value_counts.index, size=n_samples, p=value_counts.values)

def _generate_synthetic_datetime(column: pd.Series, n_samples: int) -> List:
    """Generate synthetic datetime data based on the range of the original column"""
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(column):
        try:
            column = pd.to_datetime(column)
        except:
            # If conversion fails, treat as string
            return _generate_synthetic_categorical(column.astype(str), n_samples)
    
    # Get min and max dates
    min_date = column.min()
    max_date = column.max()
    
    if pd.isna(min_date) or pd.isna(max_date):
        # If min or max is NA, use a default range
        min_date = datetime.now() - timedelta(days=365)
        max_date = datetime.now()
    
    # Calculate date range in seconds
    date_range = (max_date - min_date).total_seconds()
    
    # Generate random offsets
    random_offsets = np.random.rand(n_samples) * date_range
    
    # Create new dates
    new_dates = [min_date + timedelta(seconds=offset) for offset in random_offsets]
    
    # Format according to the original format if it was a string
    if column.dtype == object:
        # Try to determine the format
        original_format = None
        sample = column.iloc[0]
        if isinstance(sample, str):
            if re.match(r'\d{4}-\d{2}-\d{2}', sample):
                original_format = '%Y-%m-%d'
            elif re.match(r'\d{2}/\d{2}/\d{4}', sample):
                original_format = '%m/%d/%Y'
            elif re.match(r'\d{2}-\d{2}-\d{4}', sample):
                original_format = '%m-%d-%Y'
        
        if original_format:
            return [date.strftime(original_format) for date in new_dates]
    
    return new_dates

def _generate_synthetic_text(column: pd.Series, n_samples: int) -> List[str]:
    """Generate synthetic text data based on patterns in the original column"""
    # Check if it's likely an email
    if column.str.contains('@', na=False).mean() > 0.5:
        domains = ['example.com', 'test.org', 'data.net', 'company.io']
        return [f"user_{i}@{random.choice(domains)}" for i in range(n_samples)]
    
    # Check if it's likely a name
    if column.str.contains(' ', na=False).mean() > 0.5:
        first_names = ['John', 'Jane', 'Alex', 'Maria', 'Michael', 'Sara', 'David', 'Lisa']
        last_names = ['Smith', 'Johnson', 'Brown', 'Garcia', 'Miller', 'Davis', 'Rodriguez']
        return [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_samples)]
    
    # Otherwise generate random text with similar length
    avg_length = int(column.str.len().mean())
    if np.isnan(avg_length):
        avg_length = 10  # Default length
    
    # Generate random strings
    return [''.join(random.choices(string.ascii_letters + string.digits, k=max(1, avg_length))) for _ in range(n_samples)]

def _determine_column_type(column: pd.Series) -> str:
    """Determine the data type of a column"""
    if pd.api.types.is_numeric_dtype(column):
        return 'numeric'
    elif pd.api.types.is_datetime64_any_dtype(column):
        return 'datetime'
    else:
        # Check if it's a string that can be converted to datetime
        if column.dtype == object:
            try:
                pd.to_datetime(column.iloc[0])
                return 'datetime'
            except:
                pass
            
            # Check if it's categorical
            if column.nunique() / len(column) < 0.2:  # If less than 20% unique values
                return 'categorical'
            else:
                return 'text'
        return 'categorical'

def _smart_sampling(df: pd.DataFrame, target_size: int, method: str = 'random') -> pd.DataFrame:
    """
    Perform smart sampling to reduce dataset size
    
    Args:
        df: DataFrame to sample
        target_size: Target size for the dataset
        method: Sampling method ('random', 'stratified', 'diversity', 'cluster')
        
    Returns:
        Sampled DataFrame
    """
    if target_size >= len(df):
        return df.copy()
    
    if method == 'random':
        return df.sample(target_size, random_state=42)
    
    elif method == 'stratified':
        # Try to find a categorical column for stratification
        categorical_cols = [col for col in df.columns if _determine_column_type(df[col]) == 'categorical']
        
        if categorical_cols:
            # Use the categorical column with the most balanced distribution
            best_col = categorical_cols[0]
            best_score = 0
            
            for col in categorical_cols:
                # Calculate entropy as a measure of balance
                value_counts = df[col].value_counts(normalize=True)
                entropy = -np.sum(value_counts * np.log2(value_counts))
                if entropy > best_score:
                    best_score = entropy
                    best_col = col
            
            # Perform stratified sampling
            strata = df[best_col].fillna('_missing_')
            
            # Calculate how many samples to take from each stratum
            n_strata = strata.nunique()
            sample_per_stratum = max(1, target_size // n_strata)
            
            result = pd.DataFrame()
            for value, group in df.groupby(best_col):
                if len(group) <= sample_per_stratum:
                    result = pd.concat([result, group])
                else:
                    result = pd.concat([result, group.sample(sample_per_stratum, random_state=42)])
            
            # If we don't have enough samples, add more randomly
            if len(result) < target_size:
                remaining = df[~df.index.isin(result.index)]
                additional = remaining.sample(min(target_size - len(result), len(remaining)), random_state=42)
                result = pd.concat([result, additional])
            
            # If we have too many samples, remove some randomly
            if len(result) > target_size:
                result = result.sample(target_size, random_state=42)
            
            return result
        else:
            # Fall back to random sampling
            return df.sample(target_size, random_state=42)
    
    elif method == 'diversity':
        # Try to select diverse samples using numeric columns
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols and len(numeric_cols) >= 2:
            # Normalize numeric columns
            scaler = MinMaxScaler()
            numeric_df = df[numeric_cols].copy()
            
            # Fill NAs with mean
            for col in numeric_cols:
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())
            
            numeric_scaled = scaler.fit_transform(numeric_df)
            
            # Use KDTree for diversity sampling
            tree = KDTree(numeric_scaled)
            
            # Start with a random point
            sampled_indices = [random.randint(0, len(df) - 1)]
            remaining_indices = set(range(len(df))) - set(sampled_indices)
            
            # Iteratively add the point farthest from the current set
            for _ in range(min(target_size - 1, len(df) - 1)):
                if not remaining_indices:
                    break
                    
                # Find distances to the closest already-sampled point
                distances, _ = tree.query(numeric_scaled[list(remaining_indices)])
                
                # Add the point with max min-distance
                farthest_idx = list(remaining_indices)[np.argmax(distances)]
                sampled_indices.append(farthest_idx)
                remaining_indices.remove(farthest_idx)
            
            return df.iloc[sampled_indices]
        else:
            # Fall back to random sampling
            return df.sample(target_size, random_state=42)
    
    elif method == 'cluster':
        # Try to select representative samples from clusters
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols and len(numeric_cols) >= 2:
            from sklearn.cluster import KMeans
            
            # Normalize numeric columns
            scaler = MinMaxScaler()
            numeric_df = df[numeric_cols].copy()
            
            # Fill NAs with mean
            for col in numeric_cols:
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())
            
            numeric_scaled = scaler.fit_transform(numeric_df)
            
            # Determine number of clusters
            n_clusters = min(target_size, len(df) // 2)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(numeric_scaled)
            
            # Sample from each cluster proportionally
            result = pd.DataFrame()
            for cluster_id in range(n_clusters):
                cluster_df = df[cluster_labels == cluster_id]
                cluster_size = int(np.ceil(len(cluster_df) / len(df) * target_size))
                cluster_size = min(cluster_size, len(cluster_df))
                
                if cluster_size > 0:
                    result = pd.concat([result, cluster_df.sample(cluster_size, random_state=42)])
            
            # Adjust to match target_size exactly
            if len(result) > target_size:
                result = result.sample(target_size, random_state=42)
            elif len(result) < target_size:
                remaining = df[~df.index.isin(result.index)]
                additional = remaining.sample(min(target_size - len(result), len(remaining)), random_state=42)
                result = pd.concat([result, additional])
            
            return result
        else:
            # Fall back to random sampling
            return df.sample(target_size, random_state=42)
    else:
        # Default to random sampling
        return df.sample(target_size, random_state=42)
@log_function_call
@track_changes
def resize_dataset(
    df: pd.DataFrame, 
    target_size: int,
    method: str = 'auto',
    balance_classes: bool = False,
    target_column: Optional[str] = None,
    preserve_correlation: bool = True,
    remove_outliers: bool = False,
    outlier_threshold: float = 2.0,
    synthetic_method: str = 'statistical',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Resize a dataset by expanding or reducing it to the target size.
    
    Args:
        df: Input DataFrame to resize
        target_size: Desired size of the output DataFrame
        method: Method to use for resizing:
            - 'auto': Automatically choose based on target_size
            - 'expand': Generate synthetic data to increase dataset size
            - 'reduce': Remove data to decrease dataset size
        balance_classes: If True, try to balance classes in the target_column (for classification tasks)
        target_column: Column name containing the target variable (for classification tasks)
        preserve_correlation: If True, try to preserve correlations when generating synthetic data
        remove_outliers: If True, remove outliers when reducing dataset
        outlier_threshold: Z-score threshold for outlier removal
        synthetic_method: Method for generating synthetic data:
            - 'statistical': Use statistical properties of original data
            - 'neighbor': Use nearest neighbor interpolation
        random_state: Random seed for reproducibility
        
    Returns:
        Resized DataFrame
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ResizeError("Input must be a pandas DataFrame")
    
    if target_size <= 0:
        raise ResizeError("Target size must be positive")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Automatically determine if we need to expand or reduce
    if method == 'auto':
        method = 'expand' if target_size > len(df) else 'reduce'
    
    # If target size equals current size, return a copy of the dataframe
    if target_size == len(df):
        return df.copy()
    
    # Handle balanced classes if requested
    if balance_classes and target_column:
        if target_column not in df.columns:
            raise ResizeError(f"Target column '{target_column}' not found in DataFrame")
        
        # Determine target distribution
        class_counts = df[target_column].value_counts()
        n_classes = len(class_counts)
        
        # Calculate target count per class
        target_per_class = target_size // n_classes
        
        result = pd.DataFrame()
        
        for class_value, count in class_counts.items():
            class_df = df[df[target_column] == class_value]
            
            if method == 'expand':
                if count < target_per_class:
                    # Need to generate more data for this class
                    class_result = _expand_data(class_df, target_per_class, synthetic_method, preserve_correlation)
                else:
                    # Sample down to target_per_class
                    class_result = class_df.sample(target_per_class, random_state=random_state)
            else:  # reduce
                if count > target_per_class:
                    # Reduce data for this class
                    class_result = _smart_sampling(class_df, target_per_class, 'diversity')
                else:
                    # Keep all samples from this class
                    class_result = class_df.copy()
            
            result = pd.concat([result, class_result])
        
        # Shuffle the final result
        return result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Standard resizing without class balancing
    if method == 'expand':
        return _expand_data(df, target_size, synthetic_method, preserve_correlation)
    else:  # reduce
        # Handle outlier removal if requested
        if remove_outliers:
            df_reduced = df.copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                mean = df_reduced[col].mean()
                std = df_reduced[col].std()
                if std > 0:  # Avoid division by zero
                    z_scores = np.abs((df_reduced[col] - mean) / std)
                    df_reduced = df_reduced[z_scores < outlier_threshold]
            
            # If we have enough data after outlier removal
            if len(df_reduced) >= target_size:
                return _smart_sampling(df_reduced, target_size, 'diversity')
        
        # Use smart sampling to reduce the dataset
        return _smart_sampling(df, target_size, 'diversity')

def _expand_data(
    df: pd.DataFrame, 
    target_size: int, 
    method: str = 'statistical',
    preserve_correlation: bool = True
) -> pd.DataFrame:
    """
    Expand a dataset by generating synthetic data
    
    Args:
        df: DataFrame to expand
        target_size: Target size for the expanded dataset
        method: Method for generating synthetic data
        preserve_correlation: Whether to preserve correlations in the data
        
    Returns:
        Expanded DataFrame
    """
    if target_size <= len(df):
        return df.copy()
    
    # Calculate how many new samples we need
    n_new_samples = target_size - len(df)
    
    if method == 'neighbor':
        from sklearn.neighbors import NearestNeighbors
        
        # Prepare data for nearest neighbors
        encoded_df = pd.DataFrame()
        encoders = {}
        
        for column in df.columns:
            col_type = _determine_column_type(df[column])
            
            if col_type == 'numeric':
                # For numeric columns, just copy
                encoded_df[column] = df[column]
            else:
                # For categorical/text columns, use ordinal encoding
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                encoded_df[column] = encoder.fit_transform(df[[column]]).flatten()
                encoders[column] = encoder
        
        # Fill NAs with mean/mode
        for column in encoded_df.columns:
            if encoded_df[column].isna().any():
                if pd.api.types.is_numeric_dtype(encoded_df[column]):
                    encoded_df[column] = encoded_df[column].fillna(encoded_df[column].mean())
                else:
                    encoded_df[column] = encoded_df[column].fillna(encoded_df[column].mode()[0])
        
        # Fit nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=min(5, len(df)))
        nn_model.fit(encoded_df)
        
        # Generate new samples
        synthetic_df = pd.DataFrame(columns=df.columns)
        
        for _ in range(n_new_samples):
            # Pick a random seed point
            seed_idx = random.randint(0, len(df) - 1)
            seed_point = encoded_df.iloc[seed_idx:seed_idx+1]
            
            # Find nearest neighbors
            distances, indices = nn_model.kneighbors(seed_point)
            
            # Pick a random neighbor (including the seed point)
            neighbor_idx = indices[0][random.randint(0, len(indices[0]) - 1)]
            
            # Create new sample with some random variation
            new_sample = {}
            
            for column in df.columns:
                col_type = _determine_column_type(df[column])
                
                if col_type == 'numeric':
                    # Add random noise to numeric columns
                    base_value = df.iloc[neighbor_idx][column]
                    if pd.isna(base_value):
                        base_value = df[column].mean()
                    
                    std = df[column].std()
                    if pd.isna(std) or std == 0:
                        std = abs(base_value) * 0.01 if base_value != 0 else 0.01
                    
                    new_sample[column] = base_value + np.random.normal(0, std * 0.5)
                else:
                    # For categorical/text, use the neighbor's value
                    new_sample[column] = df.iloc[neighbor_idx][column]
            
            synthetic_df = pd.concat([synthetic_df, pd.DataFrame([new_sample])], ignore_index=True)
        
        # Combine original and synthetic data
        return pd.concat([df, synthetic_df], ignore_index=True)
    
    else:  # statistical method
        # Create a DataFrame for synthetic data
        synthetic_df = pd.DataFrame(index=range(n_new_samples), columns=df.columns)
        
        if preserve_correlation and len(df.select_dtypes(include=[np.number]).columns) >= 2:
            # Generate correlated numeric data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr().fillna(0)
                
                # Generate multivariate normal distribution
                means = df[numeric_cols].mean().values
                samples = np.random.multivariate_normal(means, 
                                                       df[numeric_cols].cov().fillna(0).values, 
                                                       n_new_samples)
                
                # Add generated values to synthetic dataframe
                for i, col in enumerate(numeric_cols):
                    synthetic_df[col] = samples[:, i]
                
                # Generate other columns independently
                for column in df.columns:
                    if column not in numeric_cols:
                        col_type = _determine_column_type(df[column])
                        
                        if col_type == 'numeric':
                            synthetic_df[column] = _generate_synthetic_numeric(df[column], n_new_samples)
                        elif col_type == 'categorical':
                            synthetic_df[column] = _generate_synthetic_categorical(df[column], n_new_samples)
                        elif col_type == 'datetime':
                            synthetic_df[column] = _generate_synthetic_datetime(df[column], n_new_samples)
                        else:  # text
                            synthetic_df[column] = _generate_synthetic_text(df[column], n_new_samples)
            else:
                # Generate all columns independently
                for column in df.columns:
                    col_type = _determine_column_type(df[column])
                    
                    if col_type == 'numeric':
                        synthetic_df[column] = _generate_synthetic_numeric(df[column], n_new_samples)
                    elif col_type == 'categorical':
                        synthetic_df[column] = _generate_synthetic_categorical(df[column], n_new_samples)
                    elif col_type == 'datetime':
                        synthetic_df[column] = _generate_synthetic_datetime(df[column], n_new_samples)
                    else:  # text
                        synthetic_df[column] = _generate_synthetic_text(df[column], n_new_samples)
        else:
            # Generate all columns independently
            for column in df.columns:
                col_type = _determine_column_type(df[column])
                
                if col_type == 'numeric':
                    synthetic_df[column] = _generate_synthetic_numeric(df[column], n_new_samples)
                elif col_type == 'categorical':
                    synthetic_df[column] = _generate_synthetic_categorical(df[column], n_new_samples)
                elif col_type == 'datetime':
                    synthetic_df[column] = _generate_synthetic_datetime(df[column], n_new_samples)
                else:  # text
                    synthetic_df[column] = _generate_synthetic_text(df[column], n_new_samples)
        
        # Combine original and synthetic data
        return pd.concat([df, synthetic_df], ignore_index=True)

