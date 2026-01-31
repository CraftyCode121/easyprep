import numpy as np
from typing import Union, Optional, Literal


class SimpleImputer:
    """
    Impute missing values using various strategies.
    
    Replace missing values (NaN) using a descriptive statistic along each column
    or with a constant value.
    
    Parameters
    ----------
    strategy : {'mean', 'median', 'most_frequent', 'constant'}, default='mean'
        The imputation strategy:
        - 'mean': Replace using mean of each column
        - 'median': Replace using median of each column
        - 'most_frequent': Replace using most frequent value of each column
        - 'constant': Replace using fill_value
    fill_value : scalar or array-like, default=None
        Value to use when strategy='constant'. If None, defaults to 0 for numeric.
    
    Attributes
    ----------
    statistics_ : ndarray of shape (n_features,)
        The imputation fill value for each feature.
    n_features_in_ : int
        Number of features seen during fit.
    """
    
    def __init__(
        self,
        strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'mean',
        fill_value: Optional[Union[float, int, str]] = None
    ) -> None:
        """Initialize the SimpleImputer."""
        valid_strategies = ('mean', 'median', 'most_frequent', 'constant')
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of {valid_strategies}")
        
        if strategy == 'constant' and fill_value is None:
            fill_value = 0
        
        self.strategy: str = strategy
        self.fill_value: Optional[Union[float, int, str]] = fill_value
        self.statistics_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
    
    def fit(self, X: np.ndarray) -> 'SimpleImputer':
        """
        Fit the imputer on X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to compute statistics.
        
        Returns
        -------
        self : SimpleImputer
            Fitted imputer.
        """
        X = self._validate_data(X)
        self.n_features_in_ = X.shape[1]
        
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        
        elif self.strategy == 'most_frequent':
            self.statistics_ = self._compute_most_frequent(X)
        
        elif self.strategy == 'constant':
            self.statistics_ = np.full(X.shape[1], self.fill_value, dtype=X.dtype)
        
        if self.strategy != 'constant':
            all_nan_mask = np.isnan(self.statistics_)
            if np.any(all_nan_mask):
                fill = self.fill_value if self.fill_value is not None else 0
                self.statistics_[all_nan_mask] = fill
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute all missing values in X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to impute.
        
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            The imputed data.
        """
        self._check_is_fitted()
        X = self._validate_data(X, reset=False)
        
        X_imputed = X.copy()
        
        nan_mask = np.isnan(X_imputed)
        
        for i in range(X_imputed.shape[1]):
            if np.any(nan_mask[:, i]):
                X_imputed[nan_mask[:, i], i] = self.statistics_[i]
        
        return X_imputed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit and transform.
        
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            The imputed data.
        """
        return self.fit(X).transform(X)
    
    def _compute_most_frequent(self, X: np.ndarray) -> np.ndarray:
        """Compute most frequent value for each column."""
        most_frequent = np.empty(X.shape[1], dtype=X.dtype)
        
        for i in range(X.shape[1]):
            col = X[:, i]
            col_no_nan = col[~np.isnan(col)]
            
            if len(col_no_nan) == 0:
                most_frequent[i] = 0
            else:
                unique_vals, counts = np.unique(col_no_nan, return_counts=True)
                most_frequent[i] = unique_vals[np.argmax(counts)]
        
        return most_frequent
    
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
                    f"X has {X.shape[1]} features, but SimpleImputer is expecting "
                    f"{self.n_features_in_} features as input"
                )
        
        return X
    
    def _check_is_fitted(self) -> None:
        if self.statistics_ is None:
            raise RuntimeError(
                "This SimpleImputer instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )