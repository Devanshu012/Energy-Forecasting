import numpy as np
import pandas as pd

NATIONAL_HOLIDAYS = {
    (1, 26),   # 26 Jan
    (8, 15),   # 15 Aug
    (10, 2),   # 2 Oct
}


class WeeklyProfileRecursiveForecaster:
    """
    Builds inference features using:
    - Historical (weekday, hour) mean profiles
    - Time / cyclic / shift / off features
    - Lag + rolling HOURLY_KWH features

    Performs recursive multi-step forecasting.
    """

    def __init__(self, feature_columns: list):
        self.feature_columns = feature_columns
        self.df = None
        self.profile = None
        self.profile_indexed = None

    # ------------------------------------------------------------------
    # 1) Prepare history + profile
    # ------------------------------------------------------------------
    def _prepare_history(self, df_history: pd.DataFrame):
        df = df_history.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        self.df = df
        self.df["hour"] = self.df.index.hour
        self.df["weekday"] = self.df.index.weekday
        self.df["month"] = self.df.index.month

        # ✅ NEW: compute OFF max once
        off_values = self.df.loc[self.df["Off"] == 1, "HOURLY_KWH"]
        self.off_max_kwh = off_values.max() if len(off_values) > 0 else 0
    
        self.profile = self._build_profile()
        self.profile_indexed = self.profile.set_index(["weekday", "hour"])

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
        nextweek_thisday_06 = today_07 + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        return today_07, nextweek_thisday_06

    # ------------------------------------------------------------------
    # 3) Lag resolver
    # ------------------------------------------------------------------
    def _resolve_lag_value(self, history, target_col, base_time, lag_hours):
        required_time = base_time - pd.Timedelta(hours=lag_hours)

        if required_time in history.index:
            return history.loc[required_time, target_col]

        candidates = history[
            (history.index.weekday == required_time.weekday()) &
            (history.index.hour == required_time.hour)
        ]
        if not candidates.empty:
            return candidates[target_col].iloc[-1]

        candidates = history[history.index.hour == required_time.hour]
        if not candidates.empty:
            return candidates[target_col].iloc[-1]

        return history[target_col].iloc[-1]

    # ------------------------------------------------------------------
    # 4) Feature Builders
    # ------------------------------------------------------------------
    def _add_time_features(self, next_row, current_time):
        hour = current_time.hour
        weekday = current_time.weekday()
        month = current_time.month

        next_row["hour"] = hour
        next_row["weekday"] = weekday
        next_row["week_of_year"] = int(current_time.isocalendar().week)

        return next_row

    def _add_profile_features(self, next_row, weekday, hour):
        key = (weekday, hour)

        if key in self.profile_indexed.index:
            p = self.profile_indexed.loc[key]
        else:
            p = self.profile_indexed.mean()

        next_row["AVG_CURRENT"] = p["AVG_CURRENT_mean"]
        next_row["AVG_V_LN"] = p["AVG_V_LN_mean"]
        next_row["power_proxy"] = p["power_proxy_mean"]

        return next_row

    def _add_shift_off_features(self, next_row, current_time, off_schedule):

        hour = current_time.hour
        weekday = current_time.weekday()

        next_row["Shift_A"] = int(7 <= hour < 15)
        next_row["Shift_B"] = int(15 <= hour < 23)
        next_row["Shift_C"] = int(hour >= 23 or hour < 7)

        off_flag = False

        if weekday == 6 and next_row["Shift_B"].iloc[0] == 1:
            off_flag = True
        if weekday == 6 and hour >= 23:
            off_flag = True
        if weekday == 0 and hour < 7:
            off_flag = True

        current_date = current_time.date()

        if current_date in off_schedule:
            shifts_today = off_schedule[current_date]

            if any(next_row[s].iloc[0] == 1 for s in shifts_today):
                off_flag = True

            if "Shift_C" in shifts_today and hour >= 23:
                off_flag = True

        prev_date = current_date - pd.Timedelta(days=1)

        if prev_date in off_schedule and "Shift_C" in off_schedule[prev_date]:
            if hour < 7:
                off_flag = True

        next_row["Off"] = int(off_flag)

        return next_row

    def _add_lag_features(self, next_row, history, current_time, target_col):
        for lag in [1, 2, 24, 168]:
            lag_val = self._resolve_lag_value(history, target_col, current_time, lag)
            next_row[f"kwh_lag_{lag}"] = lag_val
        return next_row

    def _add_rolling_features(self, next_row, history, current_time, target_col):

        anchor_time = current_time - pd.Timedelta(hours=1)
        last_vals = history.loc[:anchor_time, target_col]

        if len(last_vals) < 168:
            raise ValueError("Not enough historical data for rolling features")

        l1 = last_vals.iloc[-1]

        next_row["kwh_roll_3h_mean"] = last_vals.iloc[-3:].mean()
        next_row["kwh_roll_24h_mean"] = last_vals.iloc[-24:].mean()
        next_row["kwh_roll_24h_std"] = last_vals.iloc[-24:].std()
        next_row["kwh_roll_24h_min"] = last_vals.iloc[-24:].min()
        next_row["kwh_roll_24h_max"] = last_vals.iloc[-24:].max()
        next_row["kwh_roll_168h_mean"] = last_vals.iloc[-168:].mean()
        next_row["kwh_roll_168h_std"] = last_vals.iloc[-168:].std()

        return next_row

    def _predict_and_update(self, next_row, history, model, target_col):
        # ✅ RULE-BASED OFF OVERRIDE
        if next_row["Off"].iloc[0] == 1:
            y_pred = self.off_max_kwh * 3.5
        else:
            X = next_row[self.feature_columns]
            y_pred = model.predict(X)[0]
        history.loc[next_row.index[0], target_col] = y_pred
        return y_pred, history


    def get_off_schedule(self, start_time, end_time):
        off_schedule = {}
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        # generate all dates between start and end
        date_range = pd.date_range(start_time.date(), end_time.date(), freq="D")
        for dt in date_range:
            date_only = dt.date()

            if (date_only.month, date_only.day) in NATIONAL_HOLIDAYS:
                off_schedule[date_only] = ["Shift_A", "Shift_B", "Shift_C"]
                
        return off_schedule


    # ------------------------------------------------------------------
    # 5) Recursive Forecast
    # ------------------------------------------------------------------
    def recursive_forecast(self, historical_df, model, start_time, end_time, off_schedule, off_timestamp, target_col="HOURLY_KWH"):

        if self.profile is None:
            self._prepare_history(historical_df)

        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        history = historical_df.sort_index().loc[:start_time - pd.Timedelta(hours=1)].copy()

        if len(history) < 168:
            raise ValueError("Need at least 168 hours of history before start_time")

        preds = []
        current_time = start_time

        manual_off_schedule  = {
            pd.to_datetime("2025-10-01").date(): ["Shift_B", "Shift_C"],
            pd.to_datetime("2026-02-17").date(): ["Shift_A", "Shift_B", "Shift_C"],
        }
        
        # Merge dynamic + manual schedules
        for dt, shifts in manual_off_schedule.items():
            if dt in off_schedule:
                # combine unique shifts
                off_schedule[dt] = list(set(off_schedule[dt] + shifts))
            else:
                off_schedule[dt] = shifts.copy()

        while current_time <= end_time:
            next_row = pd.DataFrame(index=[current_time])
            next_row = self._add_time_features(next_row, current_time)
            weekday = current_time.weekday()
            hour = current_time.hour
            next_row = self._add_profile_features(next_row, weekday, hour)
            next_row = self._add_shift_off_features(next_row, current_time, off_schedule)
            # ✅ NEW: Machine OFF timestamp override
            if off_timestamp and pd.Timestamp(current_time) in off_timestamp:
                next_row["Off"] = 1
            next_row["original_data"] = 0
            next_row = self._add_lag_features(next_row, history, current_time, target_col)
            next_row = self._add_rolling_features(next_row, history, current_time, target_col)
            y_pred, history = self._predict_and_update(next_row, history, model, target_col)
            off_value = next_row["Off"].iloc[0] if "Off" in next_row else 0
            preds.append((current_time, y_pred, off_value)) # Added Off column
            current_time += pd.Timedelta(hours=1)

        hourly_df = pd.DataFrame(preds, columns=["Time", "Predicted_KWH", "Off"]).set_index("Time")

        today_prediction = hourly_df.loc[start_time: start_time + pd.Timedelta(hours=23)]

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
    # 6) Public API
    # ------------------------------------------------------------------
    def build(self, df_history: pd.DataFrame, model, off_timestamp):
        self._prepare_history(df_history)
        week_start, week_end = self._get_week_window()
        off_schedule = self.get_off_schedule(week_start, week_end)
        return self.recursive_forecast(df_history, model, week_start, week_end, off_schedule, off_timestamp)
