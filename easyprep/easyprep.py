from preprocess.nan_handler import SimpleImputer
from preprocess.outlier_handler import IQROutlierHandler
from preprocess.skew_handler import SkewHandler
from preprocess.scaler import StandardScaler, MinMaxScaler
import numpy as np

class Easyprep:
    def __init__(self,
                 imputer:str='mean',
                 outlier_handler:str='replace_mean',
                 skew_handler:str='auto',
                 scaler:str='standard-scaler',
                 ):
        
        imputers = [None, 'mean', 'median', 'most_frequent', 'constant']
        outlier_handlers = [None, 'clip', 'remove', 'replace_mean', 'replace_median', 'replace_nan']
        skew_handlers = [None, 'log', 'sqrt', 'boxcox', 'yeo-johnson', 'auto']
        scalers = [None, 'standard-scaler', 'minmax-scaler']
        
        if imputer in imputers:
            if imputer == None:
                self.scaler = None
            else:
                self.imputer = SimpleImputer(strategy=imputer)
                
        if outlier_handler in outlier_handlers:
            if imputer == None:
                self.outlier_handler = None
            else:
                self.outlier_handler = IQROutlierHandler(method=outlier_handler)
                
        if skew_handler in skew_handlers:
            if skew_handler == None:
                self.skew_handler = None
            else:
                self.skew_handler = SkewHandler(method=skew_handler)
                
        if scaler in scalers:
            if scaler == None:
                self.scaler = None
            elif scaler == 'standard-scaler':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid string parameters")
        
    def fit(self, X):
        X = self._validate_data(X)
        
        self.imputer.fit(X)
        self.skew_handler.fit(X)
        self.outlier_handler.fit(X)
        self.scaler.fit(X)
        
        return self
    
    def transform(self, X):
        X = self._validate_data(X)
        
        X = self.imputer.transform(X)
        X = self.skew_handler.transform(X)
        X = self.outlier_handler.transform(X)
        X = self.scaler.transform(X)
        
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
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