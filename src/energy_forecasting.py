import pandas as pd
import numpy as np
from typing import List, Dict

class EnergyForecaster:
    """
    Multi-step iterative forecaster for energy consumption.
    Optimized to match FeatureEngineer exactly.
    """
    
    def __init__(self, model, feature_columns: List[str]):
        """
        Parameters:
        -----------
        model : trained sklearn model (e.g., RandomForest, XGBoost)
        feature_columns : list of feature names in exact order expected by model
        """
        self.model = model
        self.feature_columns = feature_columns
        self.predictions = []
        
    def prepare_historical_stats(self, historical_data: pd.DataFrame):
        """
        Compute statistics from historical data for electrical features
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Must contain: HOURLY_KWH, AVG_CURRENT, AVG_V_LN
        """
        hist_df = historical_data.copy()
        hist_df['hour'] = hist_df.index.hour
        hist_df['weekday'] = hist_df.index.weekday
        
        # Electrical features - use weekday-hour averages
        electrical_cols = ['AVG_CURRENT', 'AVG_V_LN']
        
        self.stats = {
            # Primary: weekday-hour specific
            'weekday_hour_electrical': (
                hist_df.groupby(['weekday', 'hour'])[electrical_cols].mean()
            ),
            # Fallback 1: hour specific
            'hour_electrical': (
                hist_df.groupby('hour')[electrical_cols].mean()
            ),
            # Fallback 2: global mean
            'global_electrical': hist_df[electrical_cols].mean(),
        }
        
        # Store full history for lag/rolling calculations
        self.history_kwh = historical_data['HOURLY_KWH'].values
        self.history_index = historical_data.index
        
    def _get_electrical_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Get electrical measurements from historical averages"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Try weekday-hour specific
        if (weekday, hour) in self.stats['weekday_hour_electrical'].index:
            elec = self.stats['weekday_hour_electrical'].loc[(weekday, hour)]
        # Fallback to hour average
        elif hour in self.stats['hour_electrical'].index:
            elec = self.stats['hour_electrical'].loc[hour]
        # Final fallback to global mean
        else:
            elec = self.stats['global_electrical']
        
        return {
            'AVG_CURRENT': elec['AVG_CURRENT'],
            'AVG_V_LN': elec['AVG_V_LN']
        }
    
    def _get_lag_value(self, current_idx: int, lag_hours: int) -> float:
        """
        Get lagged value from history or predictions
        
        Parameters:
        -----------
        current_idx : int
            Index in predictions array (0 = first forecast hour)
        lag_hours : int
            How many hours to look back
        """
        target_idx = current_idx - lag_hours
        
        if target_idx < 0:
            # Need historical data
            hist_idx = len(self.history_kwh) + target_idx
            if hist_idx >= 0:
                return self.history_kwh[hist_idx]
            else:
                return np.mean(self.history_kwh[-168:])  # Last week average
        else:
            # Use our predictions
            return self.predictions[target_idx]
    
    def _get_rolling_stat(self, current_idx: int, window_size: int, stat: str = 'mean') -> float:
        """
        Calculate rolling statistic from history + predictions
        
        Parameters:
        -----------
        current_idx : int
            Index in predictions array
        window_size : int
            Rolling window size in hours
        stat : str
            'mean', 'std', 'min', 'max'
        """
        # Combine history and predictions
        all_values = np.concatenate([
            self.history_kwh,
            self.predictions[:current_idx]
        ])
        
        # Take last window_size values
        window = all_values[-window_size:]
        
        if len(window) == 0:
            return np.mean(self.history_kwh[-24:])  # Fallback
        
        if stat == 'mean':
            return np.mean(window)
        elif stat == 'std':
            return np.std(window) if len(window) > 1 else 0.0
        elif stat == 'min':
            return np.min(window)
        elif stat == 'max':
            return np.max(window)
    
    def create_features_for_timestamp(self, timestamp: pd.Timestamp, pred_idx: int) -> Dict[str, float]:
        """
        Create all features for a single future timestamp
        Matches FeatureEngineer exactly
        
        Parameters:
        -----------
        timestamp : pd.Timestamp
            Future timestamp to create features for
        pred_idx : int
            Index in forecast sequence (0 = first hour)
        """
        features = {}
        
        # Extract time components
        hour = timestamp.hour
        weekday = timestamp.weekday()
        day_of_month = timestamp.day
        month = timestamp.month
        week_of_year = timestamp.isocalendar()[1]
        
        # ============= ELECTRICAL FEATURES =============
        electrical = self._get_electrical_features(timestamp)
        features['AVG_CURRENT'] = electrical['AVG_CURRENT']
        features['AVG_V_LN'] = electrical['AVG_V_LN']
        features['power_proxy'] = electrical['AVG_CURRENT'] * electrical['AVG_V_LN']
        
        # ============= TEMPORAL FEATURES =============
        features['hour'] = hour
        features['weekday'] = weekday
        features['day_of_month'] = day_of_month
        features['month'] = month
        features['week_of_year'] = week_of_year
        
        # Cyclic encoding
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # ============= LAG FEATURES (DYNAMIC) =============
        features['kwh_lag_1'] = self._get_lag_value(pred_idx, 1)
        features['kwh_lag_2'] = self._get_lag_value(pred_idx, 2)
        features['kwh_lag_24'] = self._get_lag_value(pred_idx, 24)
        features['kwh_lag_168'] = self._get_lag_value(pred_idx, 168)
        
        # ============= ROLLING FEATURES (DYNAMIC) =============
        features['kwh_roll_3h_mean'] = self._get_rolling_stat(pred_idx, 3, 'mean')
        features['kwh_roll_24h_mean'] = self._get_rolling_stat(pred_idx, 24, 'mean')
        features['kwh_roll_24h_std'] = self._get_rolling_stat(pred_idx, 24, 'std')
        features['kwh_roll_24h_min'] = self._get_rolling_stat(pred_idx, 24, 'min')
        features['kwh_roll_24h_max'] = self._get_rolling_stat(pred_idx, 24, 'max')
        features['kwh_roll_168h_mean'] = self._get_rolling_stat(pred_idx, 168, 'mean')
        features['kwh_roll_168h_std'] = self._get_rolling_stat(pred_idx, 168, 'std')
        
        # ============= RATIO FEATURES (DYNAMIC) =============
        # Use lag_1 as proxy (consistent with how model was trained)
        roll_24h = features['kwh_roll_24h_mean']
        roll_168h = features['kwh_roll_168h_mean']
        current_proxy = features['kwh_lag_1']
        
        features['kwh_ratio_to_24h_avg'] = current_proxy / (roll_24h + 1e-6)
        features['kwh_ratio_to_168h_avg'] = current_proxy / (roll_168h + 1e-6)
        
        return features
    
    def forecast(self, last_timestamp: pd.Timestamp, historical_data: pd.DataFrame, 
                 horizon: int = 168) -> pd.DataFrame:
        """
        Perform iterative multi-step forecasting
        
        Parameters:
        -----------
        last_timestamp : pd.Timestamp
            Last timestamp in training data
        historical_data : pd.DataFrame
            Historical data with HOURLY_KWH and all features
        horizon : int
            Number of hours to forecast (default: 168 = 7 days)
            
        Returns:
        --------
        pd.DataFrame with columns: ['Predicted'] + all features
        """
        # Prepare historical statistics
        self.prepare_historical_stats(historical_data)
        
        # Create future timestamps
        future_times = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=horizon,
            freq='H'
        )
        
        # Initialize predictions list
        self.predictions = []
        
        # Store results
        results = []
        
        print(f"Forecasting {horizon} hours starting from {future_times[0]}")
        
        # Iterative prediction
        for i, timestamp in enumerate(future_times):
            # Create features for this timestamp
            features = self.create_features_for_timestamp(timestamp, i)
            
            # Convert to DataFrame with correct column order
            X = pd.DataFrame([features])[self.feature_columns]
            
            # Make prediction
            pred = self.model.predict(X)[0]
            
            # Ensure non-negative prediction
            pred = max(0, pred)
            
            # Store prediction (must happen BEFORE next iteration uses it)
            self.predictions.append(pred)
            
            # Store all features + prediction
            features['Predicted'] = pred
            results.append(features)
            
            # Progress update
            if (i + 1) % 24 == 0:
                print(f"  Completed {i + 1}/{horizon} hours ({((i+1)/horizon)*100:.1f}%)")
        
        # Create result DataFrame
        forecast_df = pd.DataFrame(results, index=future_times)
        
        print(f"✓ Forecast complete: {horizon} hours")
        print(f"  Mean predicted consumption: {forecast_df['Predicted'].mean():.2f} kWh")
        print(f"  Min: {forecast_df['Predicted'].min():.2f} kWh")
        print(f"  Max: {forecast_df['Predicted'].max():.2f} kWh")
        
        return forecast_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def generate_forecast(model, feature_columns, training_data, target_series, horizon=168):
    """
    Convenience function to generate forecast
    
    Parameters:
    -----------
    model : trained model (e.g., rf_model)
    feature_columns : list of feature names (e.g., model.feature_names_in_)
    training_data : pd.DataFrame with features
    target_series : pd.Series with target values (HOURLY_KWH)
    horizon : int, hours to forecast
    
    Returns:
    --------
    pd.DataFrame with forecast
    """
    # Combine features and target
    historical_data = training_data.copy()
    historical_data['HOURLY_KWH'] = target_series
    
    # Get last timestamp
    last_timestamp = training_data.index.max()
    
    # Create forecaster
    forecaster = EnergyForecaster(model, feature_columns)
    
    # Generate forecast
    forecast_df = forecaster.forecast(last_timestamp, historical_data, horizon)
    
    return forecast_df


# # ============================================================================
# # USAGE EXAMPLE
# # ============================================================================

# def generate_forecast(model, feature_columns, training_data, target_series, horizon=168):
#     """
#     Convenience function to generate forecast
    
#     Parameters:
#     -----------
#     model : trained model (e.g., rf_random)
#     feature_columns : list of feature names (e.g., rf_feature_cols)
#     training_data : pd.DataFrame with features (e.g., rf_X)
#     target_series : pd.Series with target values (e.g., y)
#     horizon : int, hours to forecast
    
#     Returns:
#     --------
#     pd.DataFrame with forecast
#     """
#     # Combine features and target
#     historical_data = training_data.copy()
#     historical_data['HOURLY_KWH'] = target_series
    
#     # Get last timestamp
#     last_timestamp = training_data.index.max()
    
#     # Create forecaster
#     forecaster = EnergyForecaster(model, feature_columns)
    
#     # Generate forecast
#     forecast_df = forecaster.forecast(last_timestamp, historical_data, horizon)
    
#     return forecast_df


# Example usage:
# forecast_df = generate_forecast(
#     model=rf_random,
#     feature_columns=rf_feature_cols,
#     training_data=rf_X,
#     target_series=y,
#     horizon=24 * 7  # 7 days
# )