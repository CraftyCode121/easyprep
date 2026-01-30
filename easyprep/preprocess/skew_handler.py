import numpy as np
from typing import Literal, Optional, Union


class SkewHandler:
    """
    Transform skewed features to approximate normal distribution.
    
    Applies various transformations to reduce skewness in data, making it more
    suitable for algorithms that assume normality.
    
    Parameters
    ----------
    method : {'log', 'sqrt', 'boxcox', 'yeo-johnson', 'auto'}, default='auto'
        Transformation method:
        - 'log': Natural logarithm (requires positive values)
        - 'sqrt': Square root (requires non-negative values)
        - 'boxcox': Box-Cox transformation (requires positive values)
        - 'yeo-johnson': Yeo-Johnson transformation (works with any values)
        - 'auto': Automatically select best method per feature
    threshold : float, default=0.5
        Absolute skewness threshold. Features with |skewness| > threshold are transformed.
    
    Attributes
    ----------
    skewness_before_ : ndarray of shape (n_features,)
        Skewness of each feature before transformation.
    skewness_after_ : ndarray of shape (n_features,)
        Skewness of each feature after transformation.
    methods_used_ : ndarray of shape (n_features,)
        Method used for each feature ('none' if not transformed).
    lambda_boxcox_ : ndarray of shape (n_features,)
        Lambda parameters for Box-Cox transformation (None for other methods).
    lambda_yeojohnson_ : ndarray of shape (n_features,)
        Lambda parameters for Yeo-Johnson transformation (None for other methods).
    n_features_in_ : int
        Number of features seen during fit.
    transformed_features_ : ndarray of bool, shape (n_features,)
        Boolean mask indicating which features were transformed.
    """
    
    def __init__(
        self,
        method: Literal['log', 'sqrt', 'boxcox', 'yeo-johnson', 'auto'] = 'auto',
        threshold: float = 0.5
    ) -> None:
        """Initialize the SkewHandler."""
        valid_methods = ('log', 'sqrt', 'boxcox', 'yeo-johnson', 'auto')
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}")
        
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        
        self.method: str = method
        self.threshold: float = threshold
        self.skewness_before_: Optional[np.ndarray] = None
        self.skewness_after_: Optional[np.ndarray] = None
        self.methods_used_: Optional[np.ndarray] = None
        self.lambda_boxcox_: Optional[np.ndarray] = None
        self.lambda_yeojohnson_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.transformed_features_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> 'SkewHandler':
        """
        Compute skewness and determine transformation parameters.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data to analyze skewness.
        
        Returns
        -------
        self : SkewHandler
            Fitted handler.
        """
        X = self._validate_data(X)
        self.n_features_in_ = X.shape[1]
        
        # Compute initial skewness
        self.skewness_before_ = self._compute_skewness(X)
        
        # Initialize arrays
        self.methods_used_ = np.array(['none'] * X.shape[1], dtype=object)
        self.lambda_boxcox_ = np.full(X.shape[1], np.nan)
        self.lambda_yeojohnson_ = np.full(X.shape[1], np.nan)
        self.transformed_features_ = np.abs(self.skewness_before_) > self.threshold
        
        # Determine transformation for each feature
        for i in range(X.shape[1]):
            if not self.transformed_features_[i]:
                continue
            
            col = X[:, i]
            
            if self.method == 'auto':
                # Try different methods and pick best
                self.methods_used_[i] = self._select_best_method(col)
            else:
                self.methods_used_[i] = self.method
            
            # Compute lambda for Box-Cox or Yeo-Johnson
            if self.methods_used_[i] == 'boxcox':
                self.lambda_boxcox_[i] = self._estimate_boxcox_lambda(col)
            elif self.methods_used_[i] == 'yeo-johnson':
                self.lambda_yeojohnson_[i] = self._estimate_yeojohnson_lambda(col)
        
        # Compute skewness after transformation
        X_transformed = self.transform(X)
        self.skewness_after_ = self._compute_skewness(X_transformed)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply skewness transformation.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        self._check_is_fitted()
        X = self._validate_data(X, reset=False)
        
        X_transformed = X.copy()
        
        for i in range(X.shape[1]):
            if self.methods_used_[i] == 'none':
                continue
            
            col = X[:, i]
            method = self.methods_used_[i]
            
            if method == 'log':
                X_transformed[:, i] = self._log_transform(col)
            elif method == 'sqrt':
                X_transformed[:, i] = self._sqrt_transform(col)
            elif method == 'boxcox':
                X_transformed[:, i] = self._boxcox_transform(col, self.lambda_boxcox_[i])
            elif method == 'yeo-johnson':
                X_transformed[:, i] = self._yeojohnson_transform(col, self.lambda_yeojohnson_[i])
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to fit and transform.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply inverse transformation.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Transformed data to invert.
        
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original scale.
        """
        self._check_is_fitted()
        X = self._validate_data(X, reset=False)
        
        X_original = X.copy()
        
        for i in range(X.shape[1]):
            if self.methods_used_[i] == 'none':
                continue
            
            col = X[:, i]
            method = self.methods_used_[i]
            
            if method == 'log':
                X_original[:, i] = np.exp(col)
            elif method == 'sqrt':
                X_original[:, i] = col ** 2
            elif method == 'boxcox':
                X_original[:, i] = self._boxcox_inverse(col, self.lambda_boxcox_[i])
            elif method == 'yeo-johnson':
                X_original[:, i] = self._yeojohnson_inverse(col, self.lambda_yeojohnson_[i])
        
        return X_original
    
    def _compute_skewness(self, X: np.ndarray) -> np.ndarray:
        """Compute skewness for each column using scipy's definition."""
        n = X.shape[0]
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)
        
        std = np.where(std == 0, 1, std)
        
        skewness = np.mean(((X - mean) / std) ** 3, axis=0)
        
        # Bias correction for sample skewness
        skewness = skewness * np.sqrt(n * (n - 1)) / (n - 2) if n > 2 else skewness
        
        return skewness
    
    def _select_best_method(self, col: np.ndarray) -> str:
        """Select best transformation method based on resulting skewness."""
        methods_to_try = []
        
        if np.all(col > 0):
            methods_to_try = ['log', 'sqrt', 'boxcox']
        elif np.all(col >= 0):
            methods_to_try = ['sqrt', 'yeo-johnson']
        else:
            methods_to_try = ['yeo-johnson']
        
        best_method = 'none'
        best_skewness = np.abs(self._compute_skewness(col.reshape(-1, 1))[0])
        
        for method in methods_to_try:
            try:
                if method == 'log':
                    transformed = self._log_transform(col)
                elif method == 'sqrt':
                    transformed = self._sqrt_transform(col)
                elif method == 'boxcox':
                    lam = self._estimate_boxcox_lambda(col)
                    transformed = self._boxcox_transform(col, lam)
                elif method == 'yeo-johnson':
                    lam = self._estimate_yeojohnson_lambda(col)
                    transformed = self._yeojohnson_transform(col, lam)
                
                skew = np.abs(self._compute_skewness(transformed.reshape(-1, 1))[0])
                
                if skew < best_skewness:
                    best_skewness = skew
                    best_method = method
            except:
                continue
        
        return best_method
    
    def _log_transform(self, col: np.ndarray) -> np.ndarray:
        """Apply log transformation."""
        if np.any(col <= 0):
            raise ValueError("Log transformation requires positive values")
        return np.log(col)
    
    def _sqrt_transform(self, col: np.ndarray) -> np.ndarray:
        """Apply square root transformation."""
        if np.any(col < 0):
            raise ValueError("Square root transformation requires non-negative values")
        return np.sqrt(col)
    
    def _estimate_boxcox_lambda(self, col: np.ndarray) -> float:
        """Estimate optimal lambda for Box-Cox transformation."""
        if np.any(col <= 0):
            raise ValueError("Box-Cox requires positive values")
        
        # Grid search for optimal lambda
        lambdas = np.linspace(-2, 2, 41)
        best_lambda = 0
        best_skewness = float('inf')
        
        for lam in lambdas:
            try:
                transformed = self._boxcox_transform(col, lam)
                skew = np.abs(self._compute_skewness(transformed.reshape(-1, 1))[0])
                if skew < best_skewness:
                    best_skewness = skew
                    best_lambda = lam
            except:
                continue
        
        return best_lambda
    
    def _boxcox_transform(self, col: np.ndarray, lam: float) -> np.ndarray:
        """Apply Box-Cox transformation."""
        if np.any(col <= 0):
            raise ValueError("Box-Cox requires positive values")
        
        if np.abs(lam) < 1e-10:
            return np.log(col)
        else:
            return (col ** lam - 1) / lam
    
    def _boxcox_inverse(self, col: np.ndarray, lam: float) -> np.ndarray:
        """Apply inverse Box-Cox transformation."""
        if np.abs(lam) < 1e-10:
            return np.exp(col)
        else:
            return (col * lam + 1) ** (1 / lam)
    
    def _estimate_yeojohnson_lambda(self, col: np.ndarray) -> float:
        """Estimate optimal lambda for Yeo-Johnson transformation."""
        lambdas = np.linspace(-2, 2, 41)
        best_lambda = 0
        best_skewness = float('inf')
        
        for lam in lambdas:
            try:
                transformed = self._yeojohnson_transform(col, lam)
                skew = np.abs(self._compute_skewness(transformed.reshape(-1, 1))[0])
                if skew < best_skewness:
                    best_skewness = skew
                    best_lambda = lam
            except:
                continue
        
        return best_lambda
    
    def _yeojohnson_transform(self, col: np.ndarray, lam: float) -> np.ndarray:
        """Apply Yeo-Johnson transformation (works with negative values)."""
        transformed = np.zeros_like(col)
        
        # For non-negative values
        pos_mask = col >= 0
        if np.any(pos_mask):
            if np.abs(lam) < 1e-10:
                transformed[pos_mask] = np.log(col[pos_mask] + 1)
            else:
                transformed[pos_mask] = ((col[pos_mask] + 1) ** lam - 1) / lam
        
        # For negative values
        neg_mask = col < 0
        if np.any(neg_mask):
            if np.abs(lam - 2) < 1e-10:
                transformed[neg_mask] = -np.log(-col[neg_mask] + 1)
            else:
                transformed[neg_mask] = -((-col[neg_mask] + 1) ** (2 - lam) - 1) / (2 - lam)
        
        return transformed
    
    def _yeojohnson_inverse(self, col: np.ndarray, lam: float) -> np.ndarray:
        """Apply inverse Yeo-Johnson transformation."""
        original = np.zeros_like(col)
        
        # Approximate threshold for determining which formula was used
        # This is a simplification; exact inverse requires knowing original signs
        pos_mask = col >= 0
        
        if np.any(pos_mask):
            if np.abs(lam) < 1e-10:
                original[pos_mask] = np.exp(col[pos_mask]) - 1
            else:
                original[pos_mask] = (col[pos_mask] * lam + 1) ** (1 / lam) - 1
        
        neg_mask = col < 0
        if np.any(neg_mask):
            if np.abs(lam - 2) < 1e-10:
                original[neg_mask] = -np.exp(-col[neg_mask]) + 1
            else:
                original[neg_mask] = -((-col[neg_mask] * (2 - lam) + 1) ** (1 / (2 - lam)) - 1)
        
        return original
    
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
                    f"X has {X.shape[1]} features, but SkewHandler is expecting "
                    f"{self.n_features_in_} features as input"
                )
        
        return X
    
    def _check_is_fitted(self) -> None:
        if self.methods_used_ is None:
            raise RuntimeError(
                "This SkewHandler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )