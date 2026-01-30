"""Efficient IQR-based outlier handler using NumPy."""

import numpy as np
from typing import Literal, Optional, Union


class IQROutlierHandler:
    """
    Detect and handle outliers using the Interquartile Range (IQR) method.
    
    Outliers are defined as values that fall below Q1 - k*IQR or above Q3 + k*IQR,
    where Q1 and Q3 are the first and third quartiles, and k is a multiplier
    (typically 1.5 for outliers, 3.0 for extreme outliers).
    
    Parameters
    ----------
    method : {'clip', 'remove', 'replace_mean', 'replace_median', 'replace_nan'}, default='clip'
        The method to handle outliers:
        - 'clip': Cap outliers to the lower/upper bounds
        - 'remove': Remove rows containing outliers
        - 'replace_mean': Replace outliers with column mean (excluding outliers)
        - 'replace_median': Replace outliers with column median (excluding outliers)
        - 'replace_nan': Replace outliers with NaN
    k : float, default=1.5
        IQR multiplier. Common values: 1.5 (mild outliers), 3.0 (extreme outliers)
    
    Attributes
    ----------
    lower_bounds_ : ndarray of shape (n_features,)
        Lower bounds (Q1 - k*IQR) for each feature.
    upper_bounds_ : ndarray of shape (n_features,)
        Upper bounds (Q3 + k*IQR) for each feature.
    q1_ : ndarray of shape (n_features,)
        First quartile (25th percentile) for each feature.
    q3_ : ndarray of shape (n_features,)
        Third quartile (75th percentile) for each feature.
    iqr_ : ndarray of shape (n_features,)
        Interquartile range (Q3 - Q1) for each feature.
    n_features_in_ : int
        Number of features seen during fit.
    n_outliers_ : int
        Total number of outlier values detected during fit.
    outlier_mask_ : ndarray of shape (n_samples, n_features)
        Boolean mask indicating outlier positions from fit data.
    """
    
    def __init__(
        self,
        method: Literal['clip', 'remove', 'replace_mean', 'replace_median', 'replace_nan'] = 'clip',
        k: float = 1.5
    ) -> None:
        """Initialize the IQROutlierHandler."""
        valid_methods = ('clip', 'remove', 'replace_mean', 'replace_median', 'replace_nan')
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}")
        
        if k <= 0:
            raise ValueError("k must be positive")
        
        self.method: str = method
        self.k: float = k
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        self.q1_: Optional[np.ndarray] = None
        self.q3_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_outliers_: Optional[int] = None
        self.outlier_mask_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> 'IQROutlierHandler':
        """
        Compute IQR bounds from X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data to compute outlier bounds.
        
        Returns
        -------
        self : IQROutlierHandler
            Fitted handler.
        """
        X = self._validate_data(X)
        self.n_features_in_ = X.shape[1]
        
        self.q1_ = np.nanpercentile(X, 25, axis=0)
        self.q3_ = np.nanpercentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        
        self.lower_bounds_ = self.q1_ - self.k * self.iqr_
        self.upper_bounds_ = self.q3_ + self.k * self.iqr_
        
        self.outlier_mask_ = self._get_outlier_mask(X)
        self.n_outliers_ = np.sum(self.outlier_mask_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Handle outliers in X according to the specified method.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples_out, n_features)
            Transformed data. If method='remove', n_samples_out may be less than n_samples.
        """
        self._check_is_fitted()
        X = self._validate_data(X, reset=False)
        
        outlier_mask = self._get_outlier_mask(X)
        
        if self.method == 'clip':
            return self._clip_outliers(X)
        
        elif self.method == 'remove':
            return self._remove_outliers(X, outlier_mask)
        
        elif self.method == 'replace_mean':
            return self._replace_outliers(X, outlier_mask, strategy='mean')
        
        elif self.method == 'replace_median':
            return self._replace_outliers(X, outlier_mask, strategy='median')
        
        elif self.method == 'replace_nan':
            return self._replace_with_nan(X, outlier_mask)
        
        return X
    
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
        return self.fit(X).transform(X)
    
    def detect_outliers(self, X: np.ndarray) -> np.ndarray:
        """
        Detect outliers without transforming.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to check for outliers.
        
        Returns
        -------
        outlier_mask : ndarray of shape (n_samples, n_features)
            Boolean mask where True indicates an outlier.
        """
        self._check_is_fitted()
        X = self._validate_data(X, reset=False)
        return self._get_outlier_mask(X)
    
    def _get_outlier_mask(self, X: np.ndarray) -> np.ndarray:
        """Get boolean mask of outlier positions."""
        lower_outliers = X < self.lower_bounds_
        upper_outliers = X > self.upper_bounds_
        return lower_outliers | upper_outliers
    
    def _clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """Clip outliers to bounds."""
        X_clipped = X.copy()
        np.clip(X_clipped, self.lower_bounds_, self.upper_bounds_, out=X_clipped)
        return X_clipped
    
    def _remove_outliers(self, X: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
        """Remove rows containing any outliers."""
        rows_with_outliers = np.any(outlier_mask, axis=1)
        X_clean = X[~rows_with_outliers]
        return X_clean
    
    def _replace_outliers(
        self,
        X: np.ndarray,
        outlier_mask: np.ndarray,
        strategy: Literal['mean', 'median']
    ) -> np.ndarray:
        """Replace outliers with mean or median of non-outliers."""
        X_replaced = X.copy()
        
        for i in range(X.shape[1]):
            col_outlier_mask = outlier_mask[:, i]
            
            if np.any(col_outlier_mask):
                col_clean = X[~col_outlier_mask, i]
                
                if len(col_clean) > 0:
                    if strategy == 'mean':
                        replacement = np.nanmean(col_clean)
                    else:  
                        replacement = np.nanmedian(col_clean)
                    
                    X_replaced[col_outlier_mask, i] = replacement
                else:
                    if strategy == 'mean':
                        X_replaced[col_outlier_mask, i] = np.nanmean(X[:, i])
                    else:
                        X_replaced[col_outlier_mask, i] = np.nanmedian(X[:, i])
        
        return X_replaced
    
    def _replace_with_nan(self, X: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
        """Replace outliers with NaN."""
        X_replaced = X.copy()
        X_replaced[outlier_mask] = np.nan
        return X_replaced
    
    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Validate input data."""
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        
        if X.shape[0] == 0:
            raise ValueError("Found array with 0 samples")
        
        if not reset and self.n_features_in_ is not None:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but IQROutlierHandler is expecting "
                    f"{self.n_features_in_} features as input"
                )
        
        return X
    
    def _check_is_fitted(self) -> None:
        if self.lower_bounds_ is None:
            raise RuntimeError(
                "This IQROutlierHandler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )