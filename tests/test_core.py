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


@pytest.fixture
def categorical_data():
    """Mixed data with categorical columns."""
    return np.array([
        ['A', 10, 'X'],
        ['B', 20, 'Y'],
        ['A', 15, 'X'],
        ['C', 25, 'Z'],
        ['B', 30, 'Y']
    ], dtype=object)


@pytest.fixture
def mixed_data_with_nans():
    """Mixed categorical and numeric data with NaNs."""
    return np.array([
        ['A', 10.5, 'X'],
        ['B', np.nan, 'Y'],
        ['A', 15.0, 'X'],
        ['C', 25.5, np.nan],
        ['B', 30.0, 'Y']
    ], dtype=object)


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


# ===== OneHotEncoder Tests =====

def test_encoder_basic_categorical(categorical_data):
    """Test basic one-hot encoding of categorical columns."""
    prep = Easyprep(encoder='ohe', ohe_indices=[0, 2], imputer=None, scaler=None)
    X_transformed = prep.fit_transform(categorical_data)
    
    # Should have more columns after encoding (original had 3, encoding col 0 and 2)
    # Col 0: A, B, C (3 values) -> 3 columns
    # Col 1: numeric (1 column) 
    # Col 2: X, Y, Z (3 values) -> 3 columns
    # Total: 1 + 3 + 3 = 7 columns
    assert X_transformed.shape[1] > categorical_data.shape[1]
    assert X_transformed.shape[0] == categorical_data.shape[0]
    
    # All values should be numeric
    assert X_transformed.dtype in [np.float64, np.float32, float]


def test_encoder_all_columns(categorical_data):
    """Test encoding all columns."""
    prep = Easyprep(encoder='ohe', ohe_indices=[0, 1, 2], imputer=None, scaler=None)
    X_transformed = prep.fit_transform(categorical_data)
    
    assert X_transformed.shape[0] == categorical_data.shape[0]
    # Should have expanded columns
    assert X_transformed.shape[1] > categorical_data.shape[1]


def test_encoder_with_scaling(categorical_data):
    """Test that encoder works with subsequent scaling."""
    prep = Easyprep(
        encoder='ohe', 
        ohe_indices=[0, 2], 
        imputer=None,
        scaler='standard-scaler'
    )
    X_transformed = prep.fit_transform(categorical_data)
    
    # Check output is numeric and finite
    assert np.all(np.isfinite(X_transformed))
    assert X_transformed.shape[0] == categorical_data.shape[0]


def test_encoder_transform_consistency(categorical_data):
    """Test that fit_transform and separate fit/transform give same results."""
    prep = Easyprep(encoder='ohe', ohe_indices=[0, 2], imputer=None, scaler=None)
    
    X_fit_transform = prep.fit_transform(categorical_data)
    
    prep2 = Easyprep(encoder='ohe', ohe_indices=[0, 2], imputer=None, scaler=None)
    prep2.fit(categorical_data)
    X_transform = prep2.transform(categorical_data)
    
    assert np.allclose(X_fit_transform, X_transform)


def test_encoder_unknown_category_raises():
    """Test that unknown categories during transform raise an error."""
    X_train = np.array([
        ['A', 10],
        ['B', 20],
        ['A', 15]
    ], dtype=object)
    
    X_test = np.array([
        ['A', 12],
        ['C', 25],  # 'C' is unknown
    ], dtype=object)
    
    prep = Easyprep(encoder='ohe', ohe_indices=[0], imputer=None, scaler=None)
    prep.fit(X_train)
    
    with pytest.raises(ValueError, match="Unknown categories"):
        prep.transform(X_test)


def test_encoder_with_full_pipeline(categorical_data):
    """Test encoder with complete preprocessing pipeline."""
    prep = Easyprep(
        encoder='ohe',
        ohe_indices=[0, 2],
        imputer='mean',
        outlier_handler='clip',
        skew_handler=None,
        scaler='minmax-scaler'
    )
    
    X_transformed = prep.fit_transform(categorical_data)
    
    # Check all values are finite and scaled
    assert np.all(np.isfinite(X_transformed))
    assert X_transformed.min() >= 0
    assert X_transformed.max() <= 1


def test_encoder_none_disabled():
    """Test that encoder=None doesn't apply encoding."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float64)
    
    prep = Easyprep(encoder=None, imputer=None, scaler=None)
    X_transformed = prep.fit_transform(X)
    
    # Shape should remain the same
    assert X_transformed.shape == X.shape
    assert np.allclose(X_transformed, X)


def test_encoder_requires_indices():
    """Test that OneHotEncoder requires indices parameter."""
    with pytest.raises(ValueError, match="Must specify indices"):
        Easyprep(encoder='ohe', ohe_indices=None)


def test_encoder_invalid_option_raises():
    """Test that invalid encoder option raises error."""
    with pytest.raises(ValueError, match="Invalid encoder"):
        Easyprep(encoder='invalid_encoder')


def test_encoder_with_mixed_data_and_nans(mixed_data_with_nans):
    """Test encoder handles mixed data with NaNs after imputation."""
    prep = Easyprep(
        encoder='ohe',
        ohe_indices=[0, 2],
        cat_imputer='most_frequent',  # Handle categorical NaNs BEFORE encoding
        imputer='mean',  # Handle numeric NaNs AFTER encoding
        scaler='standard-scaler'
    )
    
    X_transformed = prep.fit_transform(mixed_data_with_nans)
    
    # No NaNs should remain
    assert not np.isnan(X_transformed).any()
    assert X_transformed.shape[0] == mixed_data_with_nans.shape[0]
    

def test_encoder_single_category_column():
    """Test encoding a column with only one unique category."""
    X = np.array([
        ['A', 10],
        ['A', 20],
        ['A', 15]
    ], dtype=object)
    
    prep = Easyprep(encoder='ohe', ohe_indices=[0], imputer=None, scaler=None)
    X_transformed = prep.fit_transform(X)
    
    # Should still work, just creates one column for the single category
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == 2  # numeric col + 1 encoded col


def test_multiple_preprocessing_steps_order():
    """Test that preprocessing steps execute in correct order."""
    X = np.array([
        ['A', 100, 'X'],
        ['B', np.nan, 'Y'],
        ['A', 50, 'X'],
        ['C', 1000, 'Z'],  # Outlier
    ], dtype=object)
    
    prep = Easyprep(
        encoder='ohe',
        ohe_indices=[0, 2],
        imputer='median',
        outlier_handler='clip',
        scaler='standard-scaler'
    )
    
    X_transformed = prep.fit_transform(X)
    
    # Should successfully process: encode -> impute -> clip outliers -> scale
    assert np.all(np.isfinite(X_transformed))
    assert X_transformed.shape[0] == X.shape[0]
    # Should have expanded columns from encoding
    assert X_transformed.shape[1] > X.shape[1]


def test_encoder_empty_string_categories():
    """Test encoder handles empty strings as categories."""
    X = np.array([
        ['A', 10],
        ['', 20],
        ['A', 15],
        ['', 25]
    ], dtype=object)
    
    prep = Easyprep(encoder='ohe', ohe_indices=[0], imputer=None, scaler=None)
    X_transformed = prep.fit_transform(X)
    
    # Empty string is a valid category
    assert X_transformed.shape[0] == X.shape[0]
    assert np.all(np.isfinite(X_transformed))