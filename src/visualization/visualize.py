import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_stock_data

def plot_stock_prices(df, symbol='AAPL', save=True):
    """
    Plot stock prices
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data
    symbol : str
        Stock symbol (default: 'AAPL')
    save : bool
        Whether to save the plot (default: True)
    """
    plt.figure(figsize=(16, 8))
    plt.title(f'{symbol} Stock Price History')
    plt.plot(df['date'], df['close'])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price USD ($)', fontsize=14)
    plt.grid(True)
    
    if save:
        base_path = Path(__file__).parent.parent.parent
        plots_path = base_path / 'reports' / 'figures' / f'{symbol}_price_history.png'
        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        plt.savefig(plots_path)
    
    plt.show()

def plot_technical_indicators(df, symbol='AAPL', save=True):
    """
    Plot technical indicators
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with technical indicators
    symbol : str
        Stock symbol (default: 'AAPL')
    save : bool
        Whether to save the plot (default: True)
    """
    # Plot moving averages
    plt.figure(figsize=(16, 8))
    plt.title(f'{symbol} Moving Averages')
    plt.plot(df['date'], df['close'], label='Close Price')
    plt.plot(df['date'], df['MA20'], label='20-day MA')
    plt.plot(df['date'], df['MA50'], label='50-day MA')
    plt.plot(df['date'], df['MA200'], label='200-day MA')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    if save:
        base_path = Path(__file__).parent.parent.parent
        plots_path = base_path / 'reports' / 'figures' / f'{symbol}_moving_averages.png'
        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        plt.savefig(plots_path)
    
    plt.show()
    
    # Plot MACD
    plt.figure(figsize=(16, 8))
    plt.title(f'{symbol} MACD')
    plt.plot(df['date'], df['MACD'], label='MACD')
    plt.plot(df['date'], df['MACD_signal'], label='Signal Line')
    plt.bar(df['date'], df['MACD_hist'], label='MACD Histogram')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('MACD', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    if save:
        base_path = Path(__file__).parent.parent.parent
        plots_path = base_path / 'reports' / 'figures' / f'{symbol}_macd.png'
        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        plt.savefig(plots_path)
    
    plt.show()
    
    # Plot RSI
    plt.figure(figsize=(16, 8))
    plt.title(f'{symbol} RSI')
    plt.plot(df['date'], df['RSI'])
    plt.axhline(y=70, color='r', linestyle='-', label='Overbought (70)')
    plt.axhline(y=30, color='g', linestyle='-', label='Oversold (30)')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('RSI', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    if save:
        base_path = Path(__file__).parent.parent.parent
        plots_path = base_path / 'reports' / 'figures' / f'{symbol}_rsi.png'
        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        plt.savefig(plots_path)
    
    plt.show()
    
    # Plot Bollinger Bands
    plt.figure(figsize=(16, 8))
    plt.title(f'{symbol} Bollinger Bands')
    plt.plot(df['date'], df['close'], label='Close Price')
    plt.plot(df['date'], df['BB_upper'], label='Upper Band')
    plt.plot(df['date'], df['BB_middle'], label='Middle Band')
    plt.plot(df['date'], df['BB_lower'], label='Lower Band')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    if save:
        base_path = Path(__file__).parent.parent.parent
        plots_path = base_path / 'reports' / 'figures' / f'{symbol}_bollinger_bands.png'
        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        plt.savefig(plots_path)
    
    plt.show()

def plot_correlation_matrix(df, save=True):
    """
    Plot correlation matrix of features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with features
    save : bool
        Whether to save the plot (default: True)
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    
    if save:
        base_path = Path(__file__).parent.parent.parent
        plots_path = base_path / 'reports' / 'figures' / 'correlation_matrix.png'
        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        plt.savefig(plots_path)
    
    plt.show()

if __name__ == "__main__":
    # Load processed data with features
    base_path = Path(__file__).parent.parent.parent
    features_path = base_path / 'data' / 'processed' / 'AAPL_features.csv'
    
    if features_path.exists():
        data = pd.read_csv(features_path)
        data['date'] = pd.to_datetime(data['date'])
    else:
        # If features file doesn't exist, load raw data
        data = load_stock_data('AAPL')
        data['date'] = pd.to_datetime(data['date'])
    
    # Plot stock prices
    plot_stock_prices(data)
    
    # If features exist, plot technical indicators and correlation matrix
    if 'RSI' in data.columns:
        plot_technical_indicators(data)
        plot_correlation_matrix(data)