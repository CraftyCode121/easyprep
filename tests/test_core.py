import pytest
import numpy as np
from easyprep.core import Easyprep

@pytest.fixture
def messy_data():
    """Data with NaNs and some extreme values."""
    return np.array([
        [100, 90, 200],
        [6, 103, 3],
        [np.nan, 101, 7],
        [4, 105, 0.7],
        [3, 122, np.nan]
    ], dtype=np.float64)


@pytest.fixture
def clean_data():
    """Data without NaNs or extreme values."""
    return np.array([
        [50, 40, 60],
        [60, 45, 50],
        [65, 55, 55],
        [55, 50, 58],
        [53, 48, 56]
    ], dtype=np.float64)


@pytest.fixture
def single_feature_data():
    """Single feature 1D data for reshaping tests."""
    return np.array([1, 2, 3], dtype=np.float64)

@pytest.mark.parametrize("imputer", ["mean", "median", "most_frequent"])
def test_fit_transform_with_imputer(messy_data, imputer):
    prep = Easyprep(imputer=imputer)
    X_transformed = prep.fit_transform(messy_data)
    assert X_transformed.shape == messy_data.shape
    assert not np.isnan(X_transformed).any()

def test_fit_transform_with_imputer_none_raises(messy_data):
    """Easyprep with imputer=None should fail on NaNs when scaling."""
    prep = Easyprep(imputer=None)
    with pytest.raises(ValueError):
        prep.fit_transform(messy_data)

@pytest.mark.parametrize("handler", ["clip", "replace_mean", "replace_median", "replace_nan", None])
def test_outlier_handlers(clean_data, handler):
    """Test each outlier handler with an injected extreme value."""
    X = clean_data.copy()
    X[0, 0] = 1000  

    # If handler is 'replace_nan', skip scaling because StandardScaler cannot handle NaNs
    scaler = None if handler == "replace_nan" else "standard-scaler"
    prep = Easyprep(outlier_handler=handler, scaler=scaler)
    X_transformed = prep.fit_transform(X)
    
    assert X_transformed.shape == X.shape

    if handler == "replace_nan":
        assert np.isnan(X_transformed[0, 0])
    elif handler in ["clip", "replace_mean", "replace_median"]:
        # Extreme value should be reduced
        assert X_transformed[0, 0] < 1000
    else:
        assert np.isfinite(X_transformed[0, 0])


@pytest.mark.parametrize("skew_handler", ["log", "sqrt", "yeo-johnson", "auto", None])
def test_skew_handlers(clean_data, skew_handler):
    prep = Easyprep(skew_handler=skew_handler)
    X_transformed = prep.fit_transform(clean_data)
    assert X_transformed.shape == clean_data.shape
    assert np.all(np.isfinite(X_transformed))

def test_standard_scaler(clean_data):
    prep = Easyprep(scaler="standard-scaler")
    X_scaled = prep.fit_transform(clean_data)
    mean = X_scaled.mean(axis=0)
    std = X_scaled.std(axis=0)
    assert np.allclose(mean, 0, atol=1e-6)
    assert np.allclose(std, 1, atol=1e-6)

def test_minmax_scaler(clean_data):
    prep = Easyprep(scaler="minmax-scaler")
    X_scaled = prep.fit_transform(clean_data)
    assert X_scaled.min() >= 0
    assert X_scaled.max() <= 1

def test_fit_transform_consistency(clean_data):
    prep = Easyprep()
    X_fit = prep.fit_transform(clean_data)
    X_transform = prep.transform(clean_data)
    assert np.allclose(X_fit, X_transform)

def test_auto_reshape(single_feature_data):
    prep = Easyprep()
    X_transformed = prep.fit_transform(single_feature_data)
    assert X_transformed.shape == (3, 1)

def test_feature_count_mismatch(clean_data):
    prep = Easyprep()
    prep.fit(clean_data)
    X_new = np.array([[1, 2]])  
    with pytest.raises(ValueError):
        prep.transform(X_new)
        
def test_empty_input_raises():
    prep = Easyprep()
    X_empty = np.empty((0, 3))
    with pytest.raises(ValueError):
        prep.fit_transform(X_empty)

def test_all_nan_column():
    X_nan = np.array([[np.nan, 1], [np.nan, 2]])
    prep = Easyprep()
    X_transformed = prep.fit_transform(X_nan)
    assert not np.isnan(X_transformed).any()
