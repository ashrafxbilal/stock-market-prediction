import pandas as pd
import os
from pathlib import Path

def load_stock_data(symbol='AAPL', processed=False):
    """
    Load stock data from CSV files
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (default: 'AAPL')
    processed : bool
        Whether to load processed data or raw data (default: False)
        
    Returns:
    --------
    pandas.DataFrame
        Stock data
    """
    base_path = Path(__file__).parent.parent.parent
    
    if processed:
        file_path = base_path / 'data' / 'processed' / f'{symbol}_processed.csv'
    else:
        file_path = base_path / 'data' / 'raw' / f'{symbol}.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)

def process_stock_data(df, save=True, symbol='AAPL'):
    """
    Process raw stock data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw stock data
    save : bool
        Whether to save processed data (default: True)
    symbol : str
        Stock symbol (default: 'AAPL')
        
    Returns:
    --------
    pandas.DataFrame
        Processed stock data
    """
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Calculate additional features
    if 'close' in df.columns and 'open' in df.columns:
        df['daily_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate moving averages
    if 'close' in df.columns:
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    if save:
        base_path = Path(__file__).parent.parent.parent
        output_path = base_path / 'data' / 'processed' / f'{symbol}_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    # Example usage
    raw_data = load_stock_data('AAPL')
    processed_data = process_stock_data(raw_data)
    print(f"Processed data shape: {processed_data.shape}")