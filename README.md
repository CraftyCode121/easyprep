![CI – Multi-OS Python Tests](https://github.com/CraftyCode121/easyprep/actions/workflows/ci.yml/badge.svg)
# EasyPrep

**EasyPrep** is a lightweight, NumPy-based preprocessing library that dramatically reduces boilerplate code in machine learning pipelines. It provides a unified interface for common data preprocessing tasks including missing value imputation, outlier handling, skewness correction, and feature scaling.

Built with pure Python and NumPy, EasyPrep offers efficient preprocessing without heavy dependencies, making it ideal for projects where you need scikit-learn-like functionality with minimal overhead.

## Why EasyPrep?

- **Unified Interface**: One object (`Easyprep`) handles multiple preprocessing steps
- **Minimal Dependencies**: Built with pure NumPy and Python
- **Sklearn-Compatible API**: Familiar `fit()` and `transform()` methods
- **Lightweight**: Significantly smaller footprint than full ML frameworks
- **Flexible**: Use the main class or individual preprocessing components
- **Type-Safe**: Full type hints for better IDE support

## Installation

```bash
pip install ez-easyprep
```

**From source:**

```bash
git clone https://github.com/CraftyCode121/easyprep.git
cd easyprep
pip install -e .
```

## Quick Start

```python
import numpy as np
from easyprep import Easyprep

# Sample data with missing values and outliers
X = np.array([
    [1.0, 2.0, 100.0],
    [2.0, np.nan, 3.0],
    [3.0, 4.0, 5.0],
    [np.nan, 5.0, 1000.0],  # outlier
    [5.0, 6.0, 7.0]
])

# Initialize with default settings
prep = Easyprep(
    imputer='mean',
    outlier_handler='replace_mean',
    skew_handler='auto',
    scaler='standard-scaler'
)

# Fit and transform
X_clean = prep.fit_transform(X)

print(X_clean)
# Output: Preprocessed array with imputed values, handled outliers, 
# corrected skewness, and scaled features
```

## Core Concepts

EasyPrep follows the **fit-transform** pattern common in scikit-learn:

1. **Fit**: Learn parameters from training data (means, quantiles, scale factors, etc.)
2. **Transform**: Apply learned transformations to data
3. **Fit-Transform**: Convenience method combining both steps

The library processes data in a specific order:
1. **Missing Value Imputation** → Fill NaN values
2. **Outlier Handling** → Detect and handle outliers using IQR method
3. **Skewness Correction** → Apply transformations to reduce skewness
4. **Scaling** → Normalize or standardize features

## API Reference

### Main Class: `Easyprep`

The primary interface for streamlined preprocessing.

```python
from easyprep import Easyprep

prep = Easyprep(
    imputer='mean',              # How to fill missing values
    outlier_handler='clip',       # How to handle outliers
    skew_handler='auto',          # How to handle skewness
    scaler='standard-scaler'      # How to scale features
)
```

**Parameters:**

- **`imputer`** *(str, optional)*: Strategy for handling missing values
  - Options: `None`, `'mean'`, `'median'`, `'most_frequent'`, `'constant'`
  - Default: `'mean'`

- **`outlier_handler`** *(str, optional)*: Strategy for handling outliers
  - Options: `None`, `'clip'`, `'remove'`, `'replace_mean'`, `'replace_median'`, `'replace_nan'`
  - Default: `'replace_mean'`

- **`skew_handler`** *(str, optional)*: Strategy for correcting skewness
  - Options: `None`, `'log'`, `'sqrt'`, `'boxcox'`, `'yeo-johnson'`, `'auto'`
  - Default: `'auto'`

- **`scaler`** *(str, optional)*: Strategy for feature scaling
  - Options: `None`, `'standard-scaler'`, `'minmax-scaler'`
  - Default: `'standard-scaler'`

**Methods:**

- **`fit(X)`**: Learn preprocessing parameters from data
- **`transform(X)`**: Apply preprocessing transformations
- **`fit_transform(X)`**: Fit and transform in one step

### Individual Preprocessing Components

For fine-grained control, use individual preprocessing classes:

#### `SimpleImputer`

Fill missing values using various strategies.

```python
from easyprep.preprocess import SimpleImputer

imputer = SimpleImputer(strategy='mean', fill_value=None)
X_imputed = imputer.fit_transform(X)
```

**Parameters:**
- `strategy`: `'mean'`, `'median'`, `'most_frequent'`, or `'constant'`
- `fill_value`: Value to use when `strategy='constant'`

#### `IQROutlierHandler`

Detect and handle outliers using the Interquartile Range (IQR) method.

```python
from easyprep.preprocess import IQROutlierHandler

outlier_handler = IQROutlierHandler(method='clip', k=1.5)
X_clean = outlier_handler.fit_transform(X)
```

**Parameters:**
- `method`: `'clip'`, `'remove'`, `'replace_mean'`, `'replace_median'`, or `'replace_nan'`
- `k`: IQR multiplier for outlier detection (default: 1.5)

#### `SkewHandler`

Reduce feature skewness using various transformations.

```python
from easyprep.preprocess import SkewHandler

skew_handler = SkewHandler(method='auto', threshold=0.5)
X_transformed = skew_handler.fit_transform(X)
```

**Parameters:**
- `method`: `'log'`, `'sqrt'`, `'boxcox'`, `'yeo-johnson'`, or `'auto'`
- `threshold`: Skewness threshold for automatic transformation (default: 0.5)

#### `StandardScaler`

Standardize features by removing mean and scaling to unit variance.

```python
from easyprep.preprocess import StandardScaler

scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
X_scaled = scaler.fit_transform(X)
```

**Parameters:**
- `with_mean`: Center data before scaling
- `with_std`: Scale data to unit variance
- `copy`: Create a copy of input data

#### `MinMaxScaler`

Scale features to a specified range.

```python
from easyprep.preprocess import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
```

**Parameters:**
- `feature_range`: Tuple specifying the desired range (default: (0, 1))

## Usage Examples

### Example 1: Basic Pipeline

```python
import numpy as np
from easyprep import Easyprep

# Create sample data
X_train = np.array([
    [1, 2, 3],
    [4, np.nan, 6],
    [7, 8, 9],
    [10, 11, 1000]  # outlier
])

X_test = np.array([
    [2, 3, 4],
    [5, np.nan, 7]
])

# Initialize preprocessor
prep = Easyprep()

# Fit on training data
prep.fit(X_train)

# Transform both training and test data
X_train_clean = prep.transform(X_train)
X_test_clean = prep.transform(X_test)
```

### Example 2: Custom Configuration

```python
from easyprep import Easyprep

# Configure for a specific use case
prep = Easyprep(
    imputer='median',           # Use median for robustness
    outlier_handler='clip',      # Clip outliers to IQR bounds
    skew_handler='yeo-johnson',  # Handle negative values
    scaler='minmax-scaler'       # Scale to [0, 1] range
)

X_processed = prep.fit_transform(X)
```

### Example 3: Disable Specific Steps

```python
from easyprep import Easyprep

# Only impute and scale, skip outlier and skew handling
prep = Easyprep(
    imputer='mean',
    outlier_handler=None,  # Skip outlier handling
    skew_handler=None,     # Skip skewness correction
    scaler='standard-scaler'
)

X_processed = prep.fit_transform(X)
```

### Example 4: Using Individual Components

```python
from easyprep.preprocess import SimpleImputer, StandardScaler

# Build custom pipeline
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Apply sequentially
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
```

### Example 5: Handling Different Data Types

```python
import numpy as np
from easyprep import Easyprep

# Mixed data with various issues
X = np.array([
    [1.5, 2.0, 100],
    [2.5, np.nan, 200],
    [np.nan, 4.0, 150],
    [4.5, 5.0, 99999],  # extreme outlier
    [5.5, 6.0, 175]
])

prep = Easyprep(
    imputer='mean',
    outlier_handler='replace_median',
    skew_handler='auto',
    scaler='standard-scaler'
)

X_clean = prep.fit_transform(X)
```

## Project Structure

```
easyprep/
├── easyprep/
│   ├── __init__.py           # Package initialization
│   ├── core.py               # Main Easyprep class
│   └── preprocess/           # Preprocessing components
│       ├── __init__.py
│       ├── nan_handler.py    # SimpleImputer
│       ├── outlier_handler.py # IQROutlierHandler
│       ├── scaler.py         # StandardScaler, MinMaxScaler
│       └── skew_handler.py   # SkewHandler
├── tests/
│   ├── __init__.py
│   └── test_core.py          # Unit tests
├── pyproject.toml            # Project metadata
└── README.md                 # This file
```

## Requirements

- **Python**: 3.7+
- **NumPy**: 2.4+

## Error Handling

EasyPrep validates all parameters at initialization:

```python
from easyprep import Easyprep

# This will raise ValueError
try:
    prep = Easyprep(imputer='invalid_strategy')
except ValueError as e:
    print(e)  # "Invalid imputer. Must be one of [None, 'mean', 'median', ...]"
```

## Testing

EasyPrep uses `pytest` for testing. To run tests:

```bash
# Install development dependencies
pip install pytest

# Run tests
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by scikit-learn's preprocessing module
- Built for developers who need lightweight, efficient preprocessing
- Designed with simplicity and performance in mind

## Support

- **Issues**: [GitHub Issues](https://github.com/CraftyCode121/easyprep/issues)
---

**Made with LO❤️E by Hassan Rasheed, for developers**
