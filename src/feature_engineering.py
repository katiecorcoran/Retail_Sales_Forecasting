import pandas as pd

def create_features(data):
    """
    Adds time-based features to the data.
    Args:
        data (DataFrame): Cleaned data.
    Returns:
        DataFrame: Data with additional features.
    """
    # Extract time-based features
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    
    return data

if __name__ == "__main__":
    data = pd.read_csv('../data/cleaned_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    featured_data = create_features(data)
    featured_data.to_csv('../data/featured_data.csv', index=False)
    print("Features created and saved!")
