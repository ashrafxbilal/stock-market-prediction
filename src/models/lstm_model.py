import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """
    Create an LSTM model for stock price prediction
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (time_steps, features)
    units : int
        Number of LSTM units (default: 50)
    dropout_rate : float
        Dropout rate (default: 0.2)
        
    Returns:
    --------
    keras.models.Sequential
        LSTM model
    """
    model = Sequential()
    
    # First LSTM layer with dropout
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer with dropout
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Dense layers
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
    """
    Train LSTM model
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training data
    y_train : numpy.ndarray
        Training labels
    X_val : numpy.ndarray
        Validation data (default: None)
    y_val : numpy.ndarray
        Validation labels (default: None)
    epochs : int
        Number of epochs (default: 50)
    batch_size : int
        Batch size (default: 32)
        
    Returns:
    --------
    keras.models.Sequential
        Trained LSTM model
    """
    # Create model
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Set up callbacks
    base_path = Path(__file__).parent.parent.parent
    model_path = base_path / 'models' / 'lstm_model.h5'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=model_path, save_best_only=True)
    ]
    
    # Train model
    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Use a portion of training data for validation
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plot
    plots_path = base_path / 'reports' / 'figures' / 'training_history.png'
    os.makedirs(os.path.dirname(plots_path), exist_ok=True)
    plt.savefig(plots_path)
    
    return model

if __name__ == "__main__":
    # Load training and testing data
    base_path = Path(__file__).parent.parent.parent
    X_train = np.load(base_path / 'data' / 'processed' / 'X_train.npy')
    y_train = np.load(base_path / 'data' / 'processed' / 'y_train.npy')
    X_test = np.load(base_path / 'data' / 'processed' / 'X_test.npy')
    y_test = np.load(base_path / 'data' / 'processed' / 'y_test.npy')
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)
    
    # Evaluate model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")