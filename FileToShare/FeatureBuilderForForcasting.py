import numpy as np
import pandas as pd


class WeeklyProfileRecursiveForecaster:
    """
    Builds inference features using:
    - Historical (weekday, hour) mean profiles
    - Time / cyclic / shift / off features
    - Lag + rolling HOURLY_KWH features

    Performs recursive multi-step forecasting.
    """

    def __init__(self, feature_columns: list):
        """
        feature_columns:
        - Columns used during model training (exact order)
        """
        self.feature_columns = feature_columns
        self.df = None
        self.profile = None

    # ------------------------------------------------------------------
    # 1) Prepare history + profile
    # ------------------------------------------------------------------
    def _prepare_history(self, df_history: pd.DataFrame):
        self.df = df_history.copy().sort_index()

        # Time parts
        self.df["hour"] = self.df.index.hour
        self.df["weekday"] = self.df.index.weekday
        self.df["month"] = self.df.index.month

        self.profile = self._build_profile()

    def _build_profile(self):
        agg = self.df.groupby(["weekday", "hour"]).agg({
            "AVG_CURRENT": "mean",
            "AVG_V_LN": "mean",
            "power_proxy": "mean"
        }).reset_index()

        return agg.rename(columns={
            "AVG_CURRENT": "AVG_CURRENT_mean",
            "AVG_V_LN": "AVG_V_LN_mean",
            "power_proxy": "power_proxy_mean"
        })

    # ------------------------------------------------------------------
    # 2) Weekly window helper
    # ------------------------------------------------------------------
    def _get_week_window(self):
        now = pd.Timestamp.now().floor("h")
        today_07 = now.normalize() + pd.Timedelta(hours=7)
        nextweek_thisday_06  = today_07 + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        return today_07, nextweek_thisday_06

    # ------------------------------------------------------------------
    # 3) Recursive Forecast
    # ------------------------------------------------------------------
    def recursive_forecast(self, historical_df, model, start_time, end_time, target_col="HOURLY_KWH"):

        df = historical_df.copy().sort_index()
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        history = df.loc[:start_time - pd.Timedelta(hours=1)].copy()
        if len(history) < 168:
            raise ValueError("Need at least 168 hours of history before start_time")

        preds = []
        current_time = start_time

        while current_time <= end_time:
            next_row = pd.DataFrame(index=[current_time])

            # ---- Time Features
            next_row["hour"] = current_time.hour
            next_row["weekday"] = current_time.weekday()
            next_row["month"] = current_time.month
            next_row["week_of_year"] = int(current_time.isocalendar().week)

            # ---- Cyclic Encoding
            next_row["hour_sin"] = np.sin(2 * np.pi * next_row["hour"] / 24)
            next_row["hour_cos"] = np.cos(2 * np.pi * next_row["hour"] / 24)
            next_row["weekday_sin"] = np.sin(2 * np.pi * next_row["weekday"] / 7)
            next_row["weekday_cos"] = np.cos(2 * np.pi * next_row["weekday"] / 7)
            next_row["month_sin"] = np.sin(2 * np.pi * next_row["month"] / 12)
            next_row["month_cos"] = np.cos(2 * np.pi * next_row["month"] / 12)

            # ---- Profile Means
            prof = self.profile.set_index(["weekday", "hour"])
            p = prof.loc[(next_row["weekday"].iloc[0], next_row["hour"].iloc[0])]
            next_row["AVG_CURRENT"] = p["AVG_CURRENT_mean"]
            next_row["AVG_V_LN"] = p["AVG_V_LN_mean"]
            next_row["power_proxy"] = p["power_proxy_mean"]

            # ---- Shift + Off
            h = current_time.hour
            next_row["Shift_A"] = (7 <= h < 15)
            next_row["Shift_B"] = (15 <= h < 22)
            next_row["Shift_C"] = (h >= 22 or h < 7)
            next_row["Off"] = False
            
            # ---- Original vs Predicted Flag
            next_row["original_data"] = False

            # ---- Lag Features
            for lag in [1, 2, 24, 168]:
                next_row[f"kwh_lag_{lag}"] = history[target_col].iloc[-lag]

            # ---- Rolling Features
            last_vals = history[target_col]
            l1 = last_vals.iloc[-1]

            next_row["kwh_roll_3h_mean"] = last_vals.iloc[-3:].mean()
            next_row["kwh_roll_24h_mean"] = last_vals.iloc[-24:].mean()
            next_row["kwh_roll_24h_std"] = last_vals.iloc[-24:].std()
            next_row["kwh_roll_24h_min"] = last_vals.iloc[-24:].min()
            next_row["kwh_roll_24h_max"] = last_vals.iloc[-24:].max()
            next_row["kwh_roll_168h_mean"] = last_vals.iloc[-168:].mean()
            next_row["kwh_roll_168h_std"] = last_vals.iloc[-168:].std()

            next_row["kwh_ratio_to_24h_avg"] = l1 / (next_row["kwh_roll_24h_mean"] + 1e-6)
            next_row["kwh_ratio_to_168h_avg"] = l1 / (next_row["kwh_roll_168h_mean"] + 1e-6)

            # ---- Final Feature Matrix
            X = next_row[self.feature_columns]

            y_pred = model.predict(X)[0]
            preds.append((current_time, y_pred))

            history.loc[current_time, target_col] = y_pred
            current_time += pd.Timedelta(hours=1)
            
        # ================================
        # Build Output DataFrames
        # ================================
        hourly_df = pd.DataFrame(preds, columns=["Time", "Predicted_KWH"]).set_index("Time")

        # 1️⃣ Today's Hourly (first 24 hours from start_time)
        today_prediction = hourly_df.loc[start_time : start_time + pd.Timedelta(hours=23)]

        # 2️⃣ Weekly Daily (custom day: 07:00 → next day 06:00)
        weekly_prediction = (
            hourly_df
            .copy()
            .assign(shifted_time=lambda x: x.index - pd.Timedelta(hours=7))
            .set_index("shifted_time")
            .resample("D")["Predicted_KWH"]
            .sum()
            .to_frame(name="Predicted_KWH")
        )

        return today_prediction, weekly_prediction

    # ------------------------------------------------------------------
    # 4) Public API
    # ------------------------------------------------------------------
    def build(self, df_history: pd.DataFrame, model):
        self._prepare_history(df_history)
        # week_start, week_end = self._get_week_window()
        week_start = pd.Timestamp("2026-02-07 07:00:00")
        week_end   = pd.Timestamp("2026-02-14 06:00:00")
        return self.recursive_forecast(df_history, model, week_start, week_end)
