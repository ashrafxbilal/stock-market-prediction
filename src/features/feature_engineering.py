import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_stock_data

def create_technical_indicators(df):
    """
    Create technical indicators for stock price prediction
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with OHLCV columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Moving Averages
    data['MA5'] = data['close'].rolling(window=5).mean()
    data['MA10'] = data['close'].rolling(window=10).mean()
    data['MA20'] = data['close'].rolling(window=20).mean()
    data['MA50'] = data['close'].rolling(window=50).mean()
    data['MA200'] = data['close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    # RSI (Relative Strength Index)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_middle'] = data['close'].rolling(window=20).mean()
    data['BB_std'] = data['close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
    data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
    
    # Stochastic Oscillator
    data['lowest_low'] = data['low'].rolling(window=14).min()
    data['highest_high'] = data['high'].rolling(window=14).max()
    data['%K'] = 100 * ((data['close'] - data['lowest_low']) / 
                        (data['highest_high'] - data['lowest_low']))
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    # Drop rows with NaN values
    data = data.dropna()
    
    return data

def save_features(df, symbol='AAPL'):
    """
    Save engineered features to CSV
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with technical indicators
    symbol : str
        Stock symbol (default: 'AAPL')
    """
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / 'data' / 'processed' / f'{symbol}_features.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

if __name__ == "__main__":
    # Load processed data
    data = load_stock_data('AAPL', processed=True)
    
    # Create technical indicators
    features_df = create_technical_indicators(data)
    
    # Save features
    save_features(features_df)
    
    print(f"Created {len(features_df.columns) - 5} new features")
    print(f"Final dataset shape: {features_df.shape}")