import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

class RegressionEngine:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=degree)
        self.is_fitted = False

    def find_best_degree(self, X, y, max_degree=5):
        best_degree = 1
        best_r2 = -float('inf')
        
        for d in range(1, max_degree + 1):
            poly = PolynomialFeatures(degree=d)
            X_poly = poly.fit_transform(X.reshape(-1, 1))
            
            # Simple train-test split to avoid overfitting in selection
            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            if score > best_r2:
                best_r2 = score
                best_degree = d
        
        self.degree = best_degree
        self.poly_features = PolynomialFeatures(degree=best_degree)
        return best_degree

    def fit(self, X, y):
        X_poly = self.poly_features.fit_transform(X.reshape(-1, 1))
        self.model.fit(X_poly, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise Exception("Model not fitted yet")
        X_poly = self.poly_features.transform(X.reshape(-1, 1))
        return self.model.predict(X_poly)

    def get_line_data(self, min_x, max_x, points=100):
        X_line = np.linspace(min_x, max_x, points).reshape(-1, 1)
        y_line = self.predict(X_line)
        return X_line.flatten().tolist(), y_line.tolist()

class StatsEngine:
    @staticmethod
    def compute_all(y_true, y_pred, X):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Adjusted R2
        n = len(y_true)
        p = X.shape[1] if len(X.shape) > 1 else 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) != 0 else r2
        
        residuals = y_true - y_pred
        
        return {
            "r2": float(r2),
            "adj_r2": float(adj_r2),
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "residuals": residuals.tolist()
        }
