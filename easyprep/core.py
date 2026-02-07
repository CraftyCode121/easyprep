from .preprocess.nan_handler import SimpleImputer
from .preprocess.outlier_handler import IQROutlierHandler
from .preprocess.skew_handler import SkewHandler
from .preprocess.scaler import StandardScaler, MinMaxScaler
from .preprocess.encoder import OneHotEncoder
import numpy as np
from typing import Optional, List


class Easyprep:
    """
    All-in-one preprocessing pipeline for tabular data.
    
    Automatically handles common preprocessing steps in the optimal order:
    1. Categorical missing value imputation (optional)
    2. One-hot encoding (optional)
    3. Numeric missing value imputation
    4. Skewness correction
    5. Outlier handling
    6. Feature scaling
    
    Parameters
    ----------
    imputer : str or None, default='mean'
        Strategy for imputing missing numeric values. Options: 'mean', 'median', 
        'most_frequent', 'constant', None
    outlier_handler : str or None, default='replace_mean'
        Method for handling outliers. Options: 'clip', 'remove', 'replace_mean',
        'replace_median', 'replace_nan', None
    skew_handler : str or None, default='auto'
        Method for handling skewed distributions. Options: 'log', 'sqrt', 
        'boxcox', 'yeo-johnson', 'auto', None
    scaler : str or None, default='standard-scaler'
        Scaling method. Options: 'standard-scaler', 'minmax-scaler', None
    encoder : str or None, default=None
        Encoding method for categorical features. Options: 'ohe', None
    ohe_indices : list of int or None, default=None
        Indices of columns to one-hot encode. Required if encoder='ohe'.
    cat_imputer : str or None, default=None
        Strategy for imputing missing categorical values (before encoding).
        Options: 'most_frequent', 'constant', None
    """
    
    def __init__(
        self,
        imputer: str = 'mean',
        outlier_handler: str = 'replace_mean',
        skew_handler: str = 'auto',
        scaler: str = 'standard-scaler',
        encoder: Optional[str] = None,
        ohe_indices: Optional[List[int]] = None,
        cat_imputer: Optional[str] = None
    ):
        
        imputers = [None, 'mean', 'median', 'most_frequent', 'constant']
        cat_imputers = [None, 'most_frequent', 'constant']
        outlier_handlers = [None, 'clip', 'remove', 'replace_mean', 'replace_median', 'replace_nan']
        skew_handlers = [None, 'log', 'sqrt', 'boxcox', 'yeo-johnson', 'auto']
        scalers = [None, 'standard-scaler', 'minmax-scaler']
        encoders = [None, 'ohe']
        
        if cat_imputer not in cat_imputers:
            raise ValueError(f"Invalid cat_imputer. Must be one of {cat_imputers}")
        
        if cat_imputer is None:
            self.cat_imputer = None
        else:
            self.cat_imputer = SimpleImputer(strategy=cat_imputer)
        
        if encoder is None:
            self.encoder = None
        elif encoder not in encoders:
            raise ValueError(f"Invalid encoder. Must be one of {encoders}")
        elif encoder == "ohe":
            if ohe_indices is None:
                raise ValueError("Must specify indices for the OneHotEncoder to work")
            self.encoder = OneHotEncoder(indices=ohe_indices)
        else:
            self.encoder = None
        
        if imputer not in imputers:
            raise ValueError(f"Invalid imputer. Must be one of {imputers}")
        
        if imputer is None:
            self.imputer = None
        else:
            self.imputer = SimpleImputer(strategy=imputer)
        
        if outlier_handler not in outlier_handlers:
            raise ValueError(f"Invalid outlier_handler. Must be one of {outlier_handlers}")
        
        if outlier_handler is None:
            self.outlier_handler = None
        else:
            self.outlier_handler = IQROutlierHandler(method=outlier_handler)
        
        if skew_handler not in skew_handlers:
            raise ValueError(f"Invalid skew_handler. Must be one of {skew_handlers}")
        
        if skew_handler is None:
            self.skew_handler = None
        else:
            self.skew_handler = SkewHandler(method=skew_handler)
        
        if scaler not in scalers:
            raise ValueError(f"Invalid scaler. Must be one of {scalers}")
        
        if scaler is None:
            self.scaler = None
        elif scaler == 'standard-scaler':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.n_features_in_ = None
    
    def fit(self, X: np.ndarray) -> 'Easyprep':
        """
        Fit all preprocessing steps.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        
        Returns
        -------
        self : Easyprep
            Fitted pipeline.
        """
        X = self._validate_data(X)
        self.n_features_in_ = X.shape[1]
        
        X_current = X.copy()
        
        if self.cat_imputer is not None:
            self.cat_imputer.fit(X_current)
            X_current = self.cat_imputer.transform(X_current)
        
        if self.encoder is not None:
            self.encoder.fit(X_current)
            X_current = self.encoder.transform(X_current)
        
        if self.imputer is not None:
            self.imputer.fit(X_current)
            X_current = self.imputer.transform(X_current)
        
        if self.skew_handler is not None:
            self.skew_handler.fit(X_current)
            X_current = self.skew_handler.transform(X_current)
    
        if self.outlier_handler is not None:
            self.outlier_handler.fit(X_current)
            X_current = self.outlier_handler.transform(X_current)
        
        if self.scaler is not None:
            self.scaler.fit(X_current)
            X_current = self.scaler.transform(X_current)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply all preprocessing steps.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples_out, n_features)
            Transformed data.
        """
        X = self._validate_data(X, reset=False)
        
        X_current = X.copy()
        
        # Applying same order as fit
        if self.cat_imputer is not None:
            X_current = self.cat_imputer.transform(X_current)
        
        if self.encoder is not None:
            X_current = self.encoder.transform(X_current)
        
        if self.imputer is not None:
            X_current = self.imputer.transform(X_current)
        
        if self.skew_handler is not None:
            X_current = self.skew_handler.transform(X_current)
        
        if self.outlier_handler is not None:
            X_current = self.outlier_handler.transform(X_current)
        
        if self.scaler is not None:
            X_current = self.scaler.transform(X_current)
        
        return X_current
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to fit and transform.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples_out, n_features)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)
    
    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Validate input data."""
        X = np.asarray(X, dtype=object)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        
        if X.shape[0] == 0:
            raise ValueError("Found array with 0 samples")
        
        if not reset and self.n_features_in_ is not None:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but Easyprep is expecting "
                    f"{self.n_features_in_} features as input"
                )
        
        return X