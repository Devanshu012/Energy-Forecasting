import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class RowDataPreprocessingPipeline:
    """
    End-to-end data preparation + feature engineering pipeline
    Safe to store inside a .pkl / joblib bundle.
    """

    def __init__(self, gap_threshold=1.0):
        self.gap_threshold = gap_threshold

    def load_and_split_by_type(self, df: pd.DataFrame):
        df = df.copy()

        if 'Type' not in df.columns:
            raise KeyError("Column 'Type' not found in the DataFrame")

        df_YWNC2_CONE = df[df['Type'].isin(['YWNC-203', 'YWNC2 CONE'])].drop_duplicates()
        df_YWNC2_CUP  = df[df['Type'].isin(['YWNC-205', 'YWNC2 CUP'])].drop_duplicates()
        df_YWNC3_CONE = df[df['Type'].isin(['YWNC-303', 'YWNC3 CONE'])].drop_duplicates()
        df_YWNC3_CUP  = df[df['Type'].isin(['YWNC-305', 'YWNC3 CUP'])].drop_duplicates()

        return df_YWNC2_CONE, df_YWNC2_CUP, df_YWNC3_CONE, df_YWNC3_CUP

    def clean_time(self, df):
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
            mask = df["Time"].astype(str).str.contains(", 24:")
            df.loc[mask, "Time"] = df.loc[mask, "Time"].str.replace(", 24:", ", 00:", n=1)
            df["Time"] = pd.to_datetime(df["Time"], format="%d/%m/%Y, %H:%M:%S")
        else:
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
            
        return df

    def create_target_variable(self, df):
        df = df.copy()
        df = df.sort_index()
        df['KWH_diff'] = df["TOTAL_NET_KWH"].diff()
        df['KWH_diff'] = df['KWH_diff'].where(df['KWH_diff'] >= 0)

        if len(df) == 0:
            raise ValueError("No valid data remaining after removing negative KWH differences")

        return df

    def data_cleaning_before_resampling(self, df):
        # make the KWH_diff null if KWH value is more then 1 (cause it is either cause of time gap or it is anmolies)
        df.loc[df["KWH_diff"] > 0.4, "KWH_diff"] = np.nan
        
        # Create Weekday and Hour columns
        df["hour"] = df.index.hour
        df["weekday"] = df.index.weekday
        
        # Fill the nan in "KWH_diff" with the help of KNN Imputer
        features = ["AVG_CURRENT", "AVG_V_LN", "hour", "weekday", "KWH_diff"]

        # Step 1: Fit scaler ONLY on complete rows (no missing KWH_Diff)
        mask = df["KWH_diff"].notna()
        scaler = StandardScaler()
        scaler.fit(df.loc[mask, features])

        # Step 2: Scale all data
        scaled = scaler.transform(df[features])

        # Step 3: Impute
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        imputed_scaled = imputer.fit_transform(scaled)

        # Step 4: Inverse transform and update original dataframe
        imputed = scaler.inverse_transform(imputed_scaled)
        df.loc[df["KWH_diff"].isna(), "KWH_diff"] = imputed[df["KWH_diff"].isna(), -1]
        
        return df
        

    # def split_large_gaps(self, df):
    #     df = df.sort_index()
    #     rows_to_add = []

    #     gap_hours = df.index.to_series().diff().dt.total_seconds() / 3600

    #     for i in range(1, len(df)):
    #         gap = gap_hours.iloc[i]

    #         if gap > self.gap_threshold:
    #             prev_time = df.index[i - 1]
    #             curr_time = df.index[i]
    #             midpoint = prev_time + (curr_time - prev_time) / 2

    #             energy = df.iloc[i]['KWH_diff'] / 2
    #             df.at[curr_time, 'KWH_diff'] = energy

    #             new_row = df.loc[curr_time].copy()
    #             new_row.name = midpoint
    #             new_row['KWH_diff'] = energy

    #             for col in ['AVG_CURRENT', 'AVG_V_LL', 'AVG_V_LN', 'FREQUENCY']:
    #                 if col in new_row:
    #                     new_row[col] = np.nan

    #             rows_to_add.append(new_row)

    #     if not rows_to_add:
    #         return df, False

    #     df = pd.concat([df, pd.DataFrame(rows_to_add)]).sort_index()
    #     return df, True

    # def handle_large_gaps(self, df):
    #     while True:
    #         df, changed = self.split_large_gaps(df)
    #         if not changed:
    #             break
    #     return df

    def resample_to_hourly(self, df):
        hourly_df = df.resample('1h').agg({
            'KWH_diff': 'sum',
            'AVG_CURRENT': 'mean',
            'AVG_V_LN': 'mean'
        })

        hourly_df.rename(columns={'KWH_diff': 'HOURLY_KWH'}, inplace=True)

        # Optional: Track which hours have no original data
        hourly_df['original_data'] = hourly_df['AVG_CURRENT'].notna()

        return hourly_df
    
    def data_cleaning_after_resampling(self, hourly_df):
        hourly_df = hourly_df.sort_index()
        
        hourly_df.loc[hourly_df['AVG_CURRENT'].isna(), 'HOURLY_KWH'] = np.nan

        hourly_df["weekday"] = hourly_df.index.weekday
        hourly_df["hour"] = hourly_df.index.hour

        # What has this machine usually consumed on this weekday at this hour, based only on what I’ve seen so far?
        def fill_past_mean(s):
            result = s.expanding(min_periods=2).mean()
            return s.fillna(result)

        for col in ["HOURLY_KWH", "AVG_CURRENT", "AVG_V_LN"]:
            hourly_df[col] = (
                hourly_df
                .groupby(["weekday", "hour"])[col]
                .transform(fill_past_mean)
            )
        # Drop the temp columns
        hourly_df = hourly_df.drop(columns=["weekday", "hour"])

        return hourly_df


    def engineer_features(self, df):
        df = df.copy()

        df["power_proxy"] = df["AVG_CURRENT"] * df["AVG_V_LN"]

        df["hour"] = df.index.hour
        df["weekday"] = df.index.weekday
        df["month"] = df.index.month
        df["week_of_year"] = df.index.isocalendar().week.astype(int)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Shift: (A:- 7 to 15)(B:- 15 to 22)(C:- 22 to 7 next day)
        hours = df.index.hour
        # Initialize columns
        df["Shift_A"] = False
        df["Shift_B"] = False
        df["Shift_C"] = False
        # Assign shifts
        df.loc[(hours >= 7) & (hours < 15), "Shift_A"] = True
        df.loc[(hours >= 15) & (hours < 23), "Shift_B"] = True
        df.loc[(hours >= 23) | (hours < 7), "Shift_C"] = True
        
        # df["Off"] feature representing Off time
        # Default
        df["Off"] = False

        # Sunday rule: Shift B & C off on Sunday
        df["Off"] |= (df["Shift_B"] & (df["weekday"] == 6))
        df["Off"] |= (df["Shift_C"] & (hours >= 23) & (df["weekday"] == 6))
        df["Off"] |= (df["Shift_C"] & (hours < 7) & (df["weekday"] == 0))

        # Special off schedule
        off_schedule = {
            "2025-10-01": ["Shift_B", "Shift_C"],
            "2025-10-02": ["Shift_A", "Shift_B", "Shift_C"],
            "2025-08-15": ["Shift_A", "Shift_B", "Shift_C"],
            "2026-01-26": ["Shift_A", "Shift_B", "Shift_C"]
        }

        for date_str, shifts in off_schedule.items():
            date = pd.to_datetime(date_str).date()

            # Mask for the exact off day
            mask_today = df.index.date == date

            # Normal shifts (A & B) → whole day off
            normal_shifts = [s for s in shifts if s != "Shift_C"]
            if normal_shifts:
                df.loc[mask_today, "Off"] |= df.loc[mask_today, normal_shifts].any(axis=1)

            # Shift C → overnight logic
            if "Shift_C" in shifts:
                # Day itself: 22:00–23:59
                df.loc[mask_today, "Off"] |= (
                    df.loc[mask_today, "Shift_C"] & (df.loc[mask_today].index.hour >= 22)
                )

                # Next day: 00:00–06:59
                next_day = date + pd.Timedelta(days=1)
                mask_next = df.index.date == next_day

                df.loc[mask_next, "Off"] |= (
                    df.loc[mask_next, "Shift_C"] & (df.loc[mask_next].index.hour < 7)
                )
            
            # Final override: if machine is consuming power, it is NOT off, if it is consuming power it is Off
            off_power_threshold = df["HOURLY_KWH"].quantile(0.05)
            df.loc[df["HOURLY_KWH"] > off_power_threshold, "Off"] = False
            df.loc[df["HOURLY_KWH"] <= off_power_threshold, "Off"] = True

        # Convert boolean flags to integers (0/1)
        bool_cols = ["Shift_A", "Shift_B", "Shift_C", "Off"]
        df[bool_cols] = df[bool_cols].astype(int)

        # Lag values
        for lag in [1, 2, 24, 168]:
            df[f"kwh_lag_{lag}"] = df["HOURLY_KWH"].shift(lag)

        df["kwh_roll_3h_mean"] = df["kwh_lag_1"].rolling(3, min_periods=1).mean()
        df["kwh_roll_24h_mean"] = df["kwh_lag_1"].rolling(24, min_periods=12).mean()
        df["kwh_roll_24h_std"] = df["kwh_lag_1"].rolling(24, min_periods=12).std()
        df["kwh_roll_24h_min"] = df["kwh_lag_1"].rolling(24, min_periods=12).min()
        df["kwh_roll_24h_max"] = df["kwh_lag_1"].rolling(24, min_periods=12).max()

        df["kwh_roll_168h_mean"] = df["kwh_lag_1"].rolling(168, min_periods=84).mean()
        df["kwh_roll_168h_std"] = df["kwh_lag_1"].rolling(168, min_periods=84).std()

        df["kwh_ratio_to_24h_avg"] = df["kwh_lag_1"] / (df["kwh_roll_24h_mean"] + 1e-6)
        df["kwh_ratio_to_168h_avg"] = df["kwh_lag_1"] / (df["kwh_roll_168h_mean"] + 1e-6)

        return df

    def smooth_spikes(self, df, col='HOURLY_KWH', window=24, std_factor=5):
        df = df.copy()
        rolling_med = df[col].rolling(window, center=True).median()
        diff = (df[col] - rolling_med).abs()
        threshold = std_factor * df[col].std()
        spike_mask = diff > threshold
        df.loc[spike_mask, col] = rolling_med
        return df

    def trim_to_last_full_day(self, df):
        """
        Trim the DataFrame so it ends at the most recent 00:00 hour.
        If the last timestamp is between 00:00 and 23:00 (incomplete day),
        the DataFrame is cut back to the last 00:00 timestamp.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")

        df = df.sort_index()

        last_ts = df.index[-1]
        last_hour = last_ts.hour

        # If last time is not exactly at 00:00, trim back
        if last_hour != 0:
            # Find the last occurrence of 00:00
            midnight_mask = df.index.hour == 0
            if midnight_mask.any():
                last_midnight = df.index[midnight_mask][-1]
                df = df.loc[:last_midnight]
            else:
                print("⚠️ No 00:00 hour found in data. Data not trimmed.")

        return df
    
    def prepare(self, df: pd.DataFrame):
        df = df.copy()

        # Step 1: Split once
        df2, df2c, df3, df3c = self.load_and_split_by_type(df)

        # Map machine labels to DataFrames
        machine_map = {
            "YWNC2 CONE": df2,
            "YWNC2 CUP":  df2c,
            "YWNC3 CONE": df3,
            "YWNC3 CUP":  df3c,
        }

        processed_frames = []

        for machine_type, df_type in machine_map.items():
            # Work on a local copy so nothing gets overwritten
            temp = df_type.copy()

            temp = self.clean_time(temp).sort_values("Time").set_index("Time")
            temp = self.create_target_variable(temp)
            temp = self.data_cleaning_before_resampling(temp)
            temp = self.resample_to_hourly(temp)
            temp = self.data_cleaning_after_resampling(temp)
            temp = self.engineer_features(temp)
            temp = self.smooth_spikes(temp)

            # Assign machine type safely
            temp["Type"] = machine_type

            processed_frames.append(temp)

        final_df = pd.concat(processed_frames, axis=0)
        return final_df.dropna()

    
    def load_and_transform_data(self, file_path):
        df = pd.read_excel(file_path)
        final_df = self.prepare(df)
        
        return final_df
        
