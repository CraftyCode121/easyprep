"""
Individual transformers:
>>> from preprocess import SimpleImputer, IQROutlierHandler, SkewHandler, MinMaxScaler
>>> imputer = SimpleImputer(strategy='mean')
>>> X_imputed = imputer.fit_transform(X)
"""

from .nan_handler import SimpleImputer
from .outlier_handler import IQROutlierHandler
from .skew_handler import SkewHandler
from .scaler import MinMaxScaler, StandardScaler
from .encoder import OneHotEncoder

__version__ = '1.1.0'

__all__ = [
    'SimpleImputer',
    'IQROutlierHandler',
    'SkewHandler',
    'MinMaxScaler',
    'StandardScaler',
    'OneHotEncoder'
]
