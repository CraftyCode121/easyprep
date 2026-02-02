import numpy as np
from typing import Optional, List


class OneHotEncoder:
    """
    Encode all categorical columns as one-hot numeric arrays.
    
    Attributes
    ----------
    categories_ : list of np.ndarray
        Unique categories for each column.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([['A', 'X'], ['B', 'Y'], ['A', 'X']])
    >>> enc = OneHotEncoder()
    >>> enc.fit(X)
    >>> X_encoded = enc.transform(X)
    """
    
    def __init__(self) -> None:
        self.categories_: Optional[List[np.ndarray]] = None
        self.n_features_in_: Optional[int] = None
    
    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """
        Validate input data.
        
        Parameters
        ----------
        X : array-like
            Input data.
        reset : bool
            Whether to reset n_features_in_.
            
        Returns
        -------
        X : np.ndarray
            Validated data.
        """
        X = np.asarray(X, dtype=object)
        
        if X.size == 0:
            raise ValueError("Input array is empty.")
        
        if X.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead. "
                "Reshape your data using X.reshape(-1, 1)."
            )
        elif X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        
        n_features = X.shape[1]
        
        if reset:
            self.n_features_in_ = n_features
        else:
            if self.n_features_in_ is not None and n_features != self.n_features_in_:
                raise ValueError(
                    f"X has {n_features} features, but encoder expects "
                    f"{self.n_features_in_} features."
                )
        
        return X
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OneHotEncoder':
        """
        Learn the categories from the data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored. Present for API consistency.
            
        Returns
        -------
        self : OneHotEncoder
        """
        X = self._validate_data(X, reset=True)
        
        self.categories_ = []
        for col_idx in range(X.shape[1]):
            unique_vals = np.unique(X[:, col_idx])
            self.categories_.append(unique_vals)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X by one-hot encoding all columns.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        X_out : np.ndarray
            One-hot encoded data.
        """
        if self.categories_ is None:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        X = self._validate_data(X, reset=False)
        n_samples = X.shape[0]
        
        parts = []
        
        for col_idx in range(X.shape[1]):
            cats = self.categories_[col_idx]
            n_cats = len(cats)
            
            col = X[:, col_idx]
            unknown_mask = ~np.isin(col, cats)

            if np.any(unknown_mask):
                unknown_values = np.unique(col[unknown_mask])
                raise ValueError(
                    f"Unknown categories in column {col_idx}: {unknown_values}"
                )
            
            encoded = np.zeros((n_samples, n_cats), dtype=float)
            
            for j, cat in enumerate(cats):
                mask = X[:, col_idx] == cat
                encoded[mask, j] = 1.0
            
            parts.append(encoded)
        
        return np.hstack(parts)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored. Present for API consistency.
            
        Returns
        -------
        X_out : np.ndarray
            One-hot encoded data.
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names.
        
        Parameters
        ----------
        input_features : list of str, optional
            Names of input features. If None, uses x0, x1, ...
            
        Returns
        -------
        feature_names : list of str
            Output feature names.
        """
        if self.categories_ is None:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        elif len(input_features) != self.n_features_in_:
            raise ValueError(
                f"input_features has {len(input_features)} elements, "
                f"but encoder expects {self.n_features_in_} features."
            )
        
        feature_names = []
        for i, cats in enumerate(self.categories_):
            for cat in cats:
                feature_names.append(f"{input_features[i]}_{cat}")
        
        return feature_names