import unittest
import sys
import os
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')
from models.lstm_model import create_lstm_model

class TestModel(unittest.TestCase):
    def test_create_lstm_model(self):
        """Test creating LSTM model"""
        # Create model with sample input shape
        input_shape = (60, 1)  # 60 time steps, 1 feature
        model = create_lstm_model(input_shape)
        
        # Check model structure
        self.assertEqual(len(model.layers), 6)  # 2 LSTM, 2 Dropout, 2 Dense layers
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 60, 1))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Test model with sample data
        X_sample = np.random.random((10, 60, 1))  # 10 samples
        predictions = model.predict(X_sample)
        
        # Check predictions shape
        self.assertEqual(predictions.shape, (10, 1))

if __name__ == '__main__':
    unittest.main()