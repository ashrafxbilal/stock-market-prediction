import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_stock_data

def load_trained_model():
    """
    Load trained LSTM model
    
    Returns:
    --------
    keras.models.Model
        Trained LSTM model
    """
    base_path = Path(__file__).parent.parent.parent
    model_path = base_path / 'models' / 'lstm_model.h5'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return load_model(model_path)

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : keras.models.Model
        Trained model
    X_test : numpy.ndarray
        Test data
    y_test : numpy.ndarray
        Test labels
    scaler : sklearn.preprocessing.MinMaxScaler
        Scaler used to normalize data
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_inv = scaler.inverse_transform(predictions).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, predictions_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    r2 = r2_score(y_test_inv, predictions_inv)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def plot_predictions(model, X_test, y_test, scaler, actual_dates=None):
    """
    Plot model predictions against actual values
    
    Parameters:
    -----------
    model : keras.models.Model
        Trained model
    X_test : numpy.ndarray
        Test data
    y_test : numpy.ndarray
        Test labels
    scaler : sklearn.preprocessing.MinMaxScaler
        Scaler used to normalize data
    actual_dates : pandas.Series
        Dates corresponding to test data (default: None)
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_inv = scaler.inverse_transform(predictions).flatten()
    
    # Create plot
    plt.figure(figsize=(16, 8))
    
    if actual_dates is not None and len(actual_dates) == len(y_test_inv):
        plt.plot(actual_dates, y_test_inv, label='Actual Price')
        plt.plot(actual_dates, predictions_inv, label='Predicted Price')
    else:
        plt.plot(y_test_inv, label='Actual Price')
        plt.plot(predictions_inv, label='Predicted Price')
    
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    
    # Save plot
    base_path = Path(__file__).parent.parent.parent
    plots_path = base_path / 'reports' / 'figures' / 'prediction_results.png'
    os.makedirs(os.path.dirname(plots_path), exist_ok=True)
    plt.savefig(plots_path)
    plt.show()

if __name__ == "__main__":
    # Load test data
    base_path = Path(__file__).parent.parent.parent
    X_test = np.load(base_path / 'data' / 'processed' / 'X_test.npy')
    y_test = np.load(base_path / 'data' / 'processed' / 'y_test.npy')
    
    # Load model
    model = load_trained_model()
    
    # Load scaler (this would need to be saved during training)
    from sklearn.preprocessing import MinMaxScaler
    raw_data = load_stock_data('AAPL')
    data = raw_data[['close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, scaler)
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Plot predictions
    plot_predictions(model, X_test, y_test, scaler)