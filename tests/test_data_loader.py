import unittest
import sys
import os
import pandas as pd
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')
from data.data_loader import load_stock_data, process_stock_data

class TestDataLoader(unittest.TestCase):
    def test_load_stock_data(self):
        """Test loading stock data"""
        # Test loading raw data
        df = load_stock_data('AAPL')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_cols = ['symbol', 'date', 'close', 'high', 'low', 'open', 'volume']
        for col in required_cols:
            self.assertIn(col.lower(), [c.lower() for c in df.columns])
    
    def test_process_stock_data(self):
        """Test processing stock data"""
        # Load raw data
        df = load_stock_data('AAPL')
        
        # Process data without saving
        processed_df = process_stock_data(df, save=False)
        
        # Check if processing added new columns
        self.assertIn('daily_return', processed_df.columns)
        self.assertIn('MA7', processed_df.columns)
        self.assertIn('MA20', processed_df.columns)
        self.assertIn('MA50', processed_df.columns)
        
        # Check if NaN values were dropped
        self.assertEqual(processed_df.isna().sum().sum(), 0)

if __name__ == '__main__':
    unittest.main()