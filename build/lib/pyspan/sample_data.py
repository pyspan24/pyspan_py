import pkg_resources
import pandas as pd
from .logging_utils import log_function_call

@log_function_call
def sample_data():
    """
    Load the Customer Sales Data dataset.

    This dataset includes simulated customer sales records with the following columns:
    - CustomerID: Unique identifier for each customer.
    - Age: Age of the customer (may include missing values and inconsistent formats).
    - Gender: Gender of the customer (may include missing values).
    - PurchaseHistory: List of previous purchases.
    - ProductCategory: Category of the purchased product.
    - PurchaseDate: Date of the purchase (may include inconsistent formats).
    - AmountSpent: Total amount spent on the purchase (includes outliers).
    - PaymentMethod: Method of payment used (includes mixed data types).
    - Country: Country of the customer.
    - MembershipStatus: Membership status (may include missing values).
    - PhoneNumber: Phone number of the customer (includes various formats).
    - DiscountCode: Discount code applied (includes duplicates).

    The dataset is stored in a CSV file located in the 'data' folder within the package.

    Returns:
        pandas.DataFrame: A DataFrame containing the Customer Sales Data.
    """
    try:
        # Use pkg_resources to access the file within the package
        data_path = pkg_resources.resource_filename('pyspan', 'data/customer_sales_data.csv')
        
        # Load the dataset using pandas
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        raise FileNotFoundError(f"Could not load sample data: {str(e)}. Please ensure the package is installed correctly.") 