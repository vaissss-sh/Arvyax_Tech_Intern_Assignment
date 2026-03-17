import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

class EmotionModel:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        self.label_encoder = LabelEncoder()
        
        if self.model_type == "baseline":
            self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        else:
            self.model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                use_label_encoder=False,
                eval_metric="mlogloss"
            )
            
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the emotion classifier.
        """
        # Encode string labels (calm, restless, etc.) to ints
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts string labels.
        """
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class IntensityModel:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        
        if self.model_type == "baseline":
            self.model = Ridge(alpha=1.0)
        else:
            self.model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                objective="reg:squarederror"
            )
            
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the intensity regressor.
        """
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts intensity and clips/rounds to [1, 5].
        """
        preds = self.model.predict(X)
        # Round and clip to strict bounds
        preds = np.round(preds)
        preds = np.clip(preds, 1, 5)
        return preds.astype(int)
