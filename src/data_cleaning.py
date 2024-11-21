import pandas as pd

def clean_data(filepath):
    """
    Cleans the raw data by handling missing values and ensuring correct data types.
    Args:
        filepath (str): Path to the raw data CSV.
    Returns:
        DataFrame: Cleaned data.
    """
    # Load data
    data = pd.read_csv(filepath)
    
    # Handle missing values (if any)
    data = data.dropna()
    
    # Ensure correct data types
    data['Date'] = pd.to_datetime(data['Date'])
    
    return data

if __name__ == "__main__":
    cleaned_data = clean_data('../data/raw_data.csv')
    cleaned_data.to_csv('../data/cleaned_data.csv', index=False)
    print("Data cleaned and saved!")
