import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np


class ModelTrainerForEachMachineType:
    """
    Time-series aware model training pipeline using Random Forest + GridSearchCV
    """

    def __init__(self, target_col="HOURLY_KWH", random_state=42):
        self.target_col = target_col
        self.random_state = random_state

    def prepare_target_and_input_feature(self, df):
        df = df.sort_index()

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        return X, y

    def train_rf_with_gridsearch(self, X_train, y_train):
        rf = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1
        )

        param_grid = {
            "n_estimators": [100, 150, 175, 200, 250],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"]
        }

        tscv = TimeSeriesSplit(n_splits=5)

        grid = GridSearchCV(
            rf,
            param_grid,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_

    def train_xgb_with_gridsearch(self, X, y):
        xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )

        param_grid = {
            "n_estimators": [450, 500, 550, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.07],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "min_child_weight": [1, 3, 5]
        }

        tscv = TimeSeriesSplit(n_splits=5)

        grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X, y)
        
        return grid.best_estimator_, grid.best_params_

    def train_pipeline_for_dataset(self, df, name="Dataset"):
        print(f"\n🚀 Training model for: {name}")

        X, y = self.prepare_target_and_input_feature(df)

        model, best_params = self.train_xgb_with_gridsearch(X, y)
        
        print(f"✅ Best Params: {best_params}")

        return {
            "model": model,
            "best_params": best_params
        }
