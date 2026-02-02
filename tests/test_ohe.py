import pytest
import numpy as np
from easyprep.preprocess import OneHotEncoder


class TestOneHotEncoder:
    """Test suite for OneHotEncoder functionality."""
    
    @pytest.fixture
    def encoder(self):
        """Provide a fresh encoder instance for each test."""
        return OneHotEncoder()
    
    @pytest.fixture
    def simple_data(self):
        """Basic categorical data for standard tests."""
        return np.array([['apple'], ['banana'], ['apple']])
    
    @pytest.fixture
    def expected_simple_output(self):
        """Expected output for simple_data."""
        return np.array([[1, 0], [0, 1], [1, 0]])
    
    def test_fit_transform_basic(self, encoder, simple_data, expected_simple_output):
        """Test basic fit_transform produces correct one-hot encoding."""
        result = encoder.fit_transform(simple_data)
        assert np.array_equal(result, expected_simple_output)
    
    def test_fit_then_transform(self, encoder, simple_data, expected_simple_output):
        """Test separate fit and transform calls."""
        encoder.fit(simple_data)
        result = encoder.transform(simple_data)
        assert np.array_equal(result, expected_simple_output)
    
    def test_transform_unseen_category(self, encoder, simple_data):
        """Test behavior when transforming data with unseen categories."""
        encoder.fit(simple_data)
        unseen_data = np.array([['orange']])
        
        with pytest.raises((ValueError, KeyError)):
            encoder.transform(unseen_data)
    
    def test_multiple_categories(self, encoder):
        """Test encoding with more than two categories."""
        data = np.array([['red'], ['green'], ['blue'], ['red']])
        result = encoder.fit_transform(data)
        
        assert result.shape == (4, 3)
        assert np.sum(result, axis=1).tolist() == [1, 1, 1, 1]  
    
    def test_single_category(self, encoder):
        """Test encoding with only one unique category."""
        data = np.array([['apple'], ['apple'], ['apple']])
        result = encoder.fit_transform(data)
        
        assert result.shape == (3, 1)
        assert np.all(result == 1)
    
    def test_output_dtype(self, encoder, simple_data):
        """Test that output has correct data type."""
        result = encoder.fit_transform(simple_data)
        assert result.dtype in [np.int32, np.int64, np.float32, np.float64]
    
    def test_empty_input(self, encoder):
        """Test handling of empty input."""
        empty_data = np.array([]).reshape(0, 1)
        
        with pytest.raises((ValueError, IndexError)):
            encoder.fit_transform(empty_data)
    
    @pytest.mark.parametrize("data,expected_shape", [
        (np.array([['a'], ['b']]), (2, 2)),
        (np.array([['a'], ['a'], ['a']]), (3, 1)),
        (np.array([['x'], ['y'], ['z'], ['x'], ['y']]), (5, 3)),
    ])
    def test_output_shape(self, encoder, data, expected_shape):
        """Test output shape for various inputs."""
        result = encoder.fit_transform(data)
        assert result.shape == expected_shape