import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from evaluation_metrics import evaluate_model  # Import the function

def train_model(data):
    """
    Trains a Random Forest model to predict sales.
    Args:
        data (DataFrame): Data with features and target variable.
    Returns:
        model: Trained Random Forest model.
        dict: Evaluation metrics (RMSE, MAE).
    """
    # Features and target
    X = data[['Year', 'Month', 'Day']]
    y = data['Sales']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate predictions
    metrics = evaluate_model(y_test, y_pred)
    
    return model, metrics

if __name__ == "__main__":
    data = pd.read_csv('../data/featured_data.csv')
    trained_model, metrics = train_model(data)
    print(f"Model trained. Evaluation Metrics: {metrics}")
