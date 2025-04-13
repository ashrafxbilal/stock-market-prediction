import pandas as pd
import numpy as np
from pathlib import Path
import os
from data_loader import load_stock_data, process_stock_data

def create_train_test_data(df, test_size=0.2, sequence_length=60):
    """
    Create training and testing datasets for time series prediction
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed stock data
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    sequence_length : int
        Number of time steps to use for each sample (default: 60)
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, scaler)
    """
    from sklearn.preprocessing import MinMaxScaler
    
    # Use only 'close' price for prediction
    data = df[['close']].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Split into train and test sets
    train_size = int(len(scaled_data) * (1 - test_size))
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]
    
    # Create sequences
    X_train, y_train = [], []
    for i in range(sequence_length, len(train_data)):
        X_train.append(train_data[i - sequence_length:i, 0])
        y_train.append(train_data[i, 0])
    
    X_test, y_test = [], []
    for i in range(sequence_length, len(test_data)):
        X_test.append(test_data[i - sequence_length:i, 0])
        y_test.append(test_data[i, 0])
    
    # Convert to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    # Load raw data
    raw_data = load_stock_data('AAPL')
    
    # Process data
    processed_data = process_stock_data(raw_data)
    
    # Create train/test datasets
    X_train, y_train, X_test, y_test, scaler = create_train_test_data(processed_data)
    
    # Save datasets
    base_path = Path(__file__).parent.parent.parent
    np.save(base_path / 'data' / 'processed' / 'X_train.npy', X_train)
    np.save(base_path / 'data' / 'processed' / 'y_train.npy', y_train)
    np.save(base_path / 'data' / 'processed' / 'X_test.npy', X_test)
    np.save(base_path / 'data' / 'processed' / 'y_test.npy', y_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")