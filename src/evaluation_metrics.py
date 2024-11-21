from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(y_true, y_pred):
    """
    Evaluates a model's predictions using RMSE and MAE.
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    Returns:
        dict: Evaluation metrics.
    """
    metrics = {
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred)
    }
    return metrics
