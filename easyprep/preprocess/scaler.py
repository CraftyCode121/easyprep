import numpy as np
from typing import Optional, Union, Tuple


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - u) / s
    
    where u is the mean of the training samples and s is the standard deviation.
    
    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        The mean value for each feature in the training set.
        
    scale_ : np.ndarray of shape (n_features,)
        The standard deviation for each feature in the training set.
        Equal to None when a feature has zero variance.
        
    var_ : np.ndarray of shape (n_features,)
        The variance for each feature in the training set.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    n_samples_seen_ : int
        Number of samples processed during fit.
        
    with_mean : bool
        If True, center the data before scaling.
        
    with_std : bool
        If True, scale the data to unit variance.
        
    Examples
    --------
    >>> import numpy as np
    >>> from standard_scaler import StandardScaler
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = StandardScaler()
    >>> scaler.fit(X)
    >>> X_scaled = scaler.transform(X)
    >>> print(X_scaled)
    [[-1.22474487 -1.22474487]
     [ 0.          0.        ]
     [ 1.22474487  1.22474487]]
    """
    
    def __init__(
        self,
        *,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True
    ) -> None:
        """
        Initialize the StandardScaler.
        
        Parameters
        ----------
        copy : bool, default=True
            If False, try to avoid a copy and do inplace scaling instead.
            This is not guaranteed to always work inplace.
            
        with_mean : bool, default=True
            If True, center the data before scaling.
            
        with_std : bool, default=True
            If True, scale the data to unit variance (or equivalently,
            unit standard deviation).
        """
        self.copy: bool = copy
        self.with_mean: bool = with_mean
        self.with_std: bool = with_std
        
        # Statistics computed during fit
        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_samples_seen_: Optional[int] = None
    
    def _validate_data(
        self,
        X: np.ndarray,
        reset: bool = True
    ) -> np.ndarray:
        """
        Validate input data and convert to appropriate format.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        reset : bool, default=True
            Whether to reset n_features_in_ attribute.
            
        Returns
        -------
        X_converted : np.ndarray
            Validated and converted input data.
            
        Raises
        ------
        ValueError
            If input data is invalid.
        """
        X = np.asarray(X)
        
        if X.size == 0:
            raise ValueError("Input array is empty.")
        
        # Ensure 2D
        if X.ndim == 1:
            raise ValueError(
                f"Expected 2D array, got 1D array instead. "
                f"Reshape your data using X.reshape(-1, 1) if your data has "
                f"a single feature or X.reshape(1, -1) if it contains a single sample."
            )
        elif X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        
        # Check for finite values
        if not np.all(np.isfinite(X)):
            raise ValueError("Input contains NaN, infinity or a value too large.")
        
        n_samples: int = X.shape[0]
        n_features: int = X.shape[1]
        
        # Validate number of features
        if reset:
            self.n_features_in_ = n_features
        else:
            if self.n_features_in_ is not None and n_features != self.n_features_in_:
                raise ValueError(
                    f"X has {n_features} features, but StandardScaler is expecting "
                    f"{self.n_features_in_} features as input."
                )
        
        return X
    
    def _reset(self) -> None:
        """Reset internal data-dependent state."""
        if hasattr(self, 'mean_'):
            del self.mean_
            del self.var_
            del self.scale_
            del self.n_samples_seen_
            del self.n_features_in_
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StandardScaler':
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
            
        y : None
            Ignored. Present for API consistency.
            
        Returns
        -------
        self : StandardScaler
            Fitted scaler.
            
        Raises
        ------
        ValueError
            If input data is invalid.
        """
        # Reset internal state
        self._reset()
        
        # Validate input
        X = self._validate_data(X, reset=True)
        
        n_samples: int = X.shape[0]
        n_features: int = X.shape[1]
        
        self.n_samples_seen_ = n_samples
        
        # Compute mean
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0, dtype=np.float64)
        else:
            self.mean_ = None
        
        # Compute variance and scale
        if self.with_std:
        
            self.var_ = np.var(X, axis=0, dtype=np.float64, ddof=0)
            
            self.scale_ = np.sqrt(self.var_)
            self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        else:
            self.var_ = None
            self.scale_ = None
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to center and scale.
            
        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Transformed array.
            
        Raises
        ------
        ValueError
            If the scaler has not been fitted yet or if input is invalid.
        """
        
        if self.mean_ is None and self.with_mean:
            raise ValueError(
                "This StandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'transform'."
            )
        
        if self.scale_ is None and self.with_std:
            raise ValueError(
                "This StandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'transform'."
            )
        
        X = self._validate_data(X, reset=False)
        
        
        if self.copy:
            X = X.copy()
        
        if self.with_mean:
            X -= self.mean_
        
        if self.with_std:
            X /= self.scale_
        
        return X
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        This is more efficient than calling fit and then transform separately.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
            
        y : None
            Ignored. Present for API consistency.
            
        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Transformed array.
            
        Raises
        ------
        ValueError
            If input data is invalid.
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data to the original representation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be transformed back.
            
        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            Transformed array.
            
        Raises
        ------
        ValueError
            If the scaler has not been fitted yet or if input is invalid.
        """

        if self.scale_ is None and self.with_std:
            raise ValueError(
                "This StandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'inverse_transform'."
            )
        
        X = self._validate_data(X, reset=False)
        
        if self.copy:
            X = X.copy()
        
        if self.with_std:
            X *= self.scale_
        
        if self.with_mean:
            X += self.mean_
        
        return X
    
    def __repr__(self) -> str:
        return (
            f"StandardScaler(copy={self.copy}, with_mean={self.with_mean}, "
            f"with_std={self.with_std})"
        )
    
    def __str__(self) -> str:
        fitted_status: str = "fitted" if self.mean_ is not None or self.scale_ is not None else "not fitted"
        return f"StandardScaler({fitted_status})"


"""
    MinMaxScaler
"""


class MinMaxScaler:
    """
    Scale features to a given range using min-max normalization.
    
    Transforms features by scaling each feature to a given range (default [0, 1]).
    Formula: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    
    Parameters
    ----------
    feature_range : tuple of float, default=(0, 1)
        Desired range of transformed data.
    
    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data.
    max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data.
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling factor.
    data_min_ : ndarray of shape (n_features,)
        Per feature minimum (same as min_).
    data_max_ : ndarray of shape (n_features,)
        Per feature maximum (same as max_).
    data_range_ : ndarray of shape (n_features,)
        Per feature range (data_max_ - data_min_).
    n_features_in_ : int
        Number of features seen during fit.
    n_samples_seen_ : int
        Number of samples seen during fit.
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)) -> None:
        """Initialize the MinMaxScaler."""
        if len(feature_range) != 2:
            raise ValueError("feature_range must be a tuple of two values")
        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range min must be less than max")
        
        self.feature_range: Tuple[float, float] = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_samples_seen_: Optional[int] = None
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute the minimum and maximum to be used for scaling.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum.
        
        Returns
        -------
        self : MinMaxScaler
            Fitted scaler.
        """
        X = self._validate_data(X)
        
        self.n_samples_seen_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Handle constant features (where min == max)
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        
        # Compute scale and min for transformation
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features of X according to feature_range.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to be transformed.
        
        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        self._check_is_fitted()
        X = self._validate_data(X, reset=False)
        
        # Efficient vectorized scaling: X * scale + min
        X_tr = X * self.scale_ + self.min_
        
        # Clip to feature range to handle numerical errors
        feature_min, feature_max = self.feature_range
        np.clip(X_tr, feature_min, feature_max, out=X_tr)
        
        return X_tr
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to be fitted and transformed.
        
        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Undo the scaling of X according to feature_range.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to be inverse transformed.
        
        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Inverse transformed data.
        """
        self._check_is_fitted()
        X = self._validate_data(X, reset=False)
        
        X_tr = (X - self.min_) / self.scale_
        
        return X_tr
    
    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
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
                    f"X has {X.shape[1]} features, but MinMaxScaler is expecting "
                    f"{self.n_features_in_} features as input"
                )
        
        return X
    
    def _check_is_fitted(self) -> None:
        if self.scale_ is None:
            raise RuntimeError(
                "This MinMaxScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )




