import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(data):
    """
    Trains a Random Forest model to predict sales.
    Args:
        data (DataFrame): Data with features and target variable.
    Returns:
        model: Trained Random Forest model.
        float: RMSE of the model on the test set.
    """
    # Features and target
    X = data[['Year', 'Month', 'Day']]
    y = data['Sales']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    return model, rmse

if __name__ == "__main__":
    data = pd.read_csv('../data/featured_data.csv')
    trained_model, rmse = train_model(data)
    print(f"Model trained with RMSE: {rmse}")
