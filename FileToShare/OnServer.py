import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import date
from FeatureBuilderForForcasting import WeeklyProfileRecursiveForecaster


BUNDLE_PATH = Path("forecasting_ready_bundle.pkl")
JSON_PATH = Path("1st_to_6th_feb.json")
CONCATINATED_DATA_PATH = Path("final_df.csv")

def load_bundle(bundle_path: Path):
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found at: {bundle_path}")
    return joblib.load(bundle_path)

def load_newer_concatenated_df(final_df: pd.DataFrame, csv_path: Path):
    """
    If a concatenated CSV exists and is newer than final_df, load and return it.
    Otherwise return the original final_df.
    """
    try:
        if csv_path.exists():
            concatinated_df = pd.read_csv(
                csv_path,
                parse_dates=["Time"],
                index_col="Time"
            ).sort_index()

            if concatinated_df.index[-1] > final_df.index[-1]:
                print("✔ Loaded newer concatenated dataframe.")
                return concatinated_df
            else:
                print("ℹ Concatenated dataframe has no newer data.")
        else:
            print("ℹ No concatenated DataFrame file found.")

    except Exception as e:
        print(f"⚠ Failed to load concatenated DataFrame: {e}")

    return final_df

def update_final_df_with_json(json_path, final_df: pd.DataFrame, bundle: dict):
    with open(json_path, "r") as f:
        json_data = json.load(f)

    new_df = pd.DataFrame(json_data)

    if "Time" not in new_df.columns:
        raise ValueError("Column 'Time' must exist in JSON data.")

    if not isinstance(final_df.index, pd.DatetimeIndex):
        raise ValueError("final_df index must be a DatetimeIndex.")

    preprocessing_pipeline = bundle["data_pipeline"]

    # Step 1: Split once
    df2, df2c, df3, df3c = preprocessing_pipeline.load_and_split_by_type(new_df)
    
    # Map machine labels to DataFrames
    machine_map = {
        "YWNC2 CONE": df2,
        "YWNC2 CUP":  df2c,
        "YWNC3 CONE": df3,
        "YWNC3 CUP":  df3c,
    }
    
    processed_frames = []
    for machine_type, df_type in machine_map.items():
        temp_new = df_type.copy()
        temp_old = final_df[final_df["Type"] == machine_type].sort_index()

        # Clean + index
        temp_new = preprocessing_pipeline.clean_time(temp_new).set_index("Time")

        if temp_old.empty:
            last_old_time = pd.Timestamp.min
        else:
            last_old_time = temp_old.index[-1]

        mask = temp_new.index > last_old_time
        if not mask.any():
            print(f"-> No new rows found in '{machine_type}' — skipping.")
            processed_frames.append(temp_old)
            continue

        temp_new = temp_new.loc[mask]

        # 🚨 Gap handling: enforce continuity
        if not temp_old.empty and not temp_new.empty:
            first_new_time = temp_new.index.min()
            gap = first_new_time - last_old_time

            if gap > pd.Timedelta(hours=1):
                bridge_time = last_old_time + pd.Timedelta(minutes=5)
                bridge_row = pd.DataFrame(
                    index=[bridge_time],
                    columns=temp_new.columns
                )
                temp_new = pd.concat([bridge_row, temp_new]).sort_index()
                temp_new = temp_new.bfill()
        
        # Process raw → hourly
        temp_new = preprocessing_pipeline.create_target_variable(temp_new)
        temp_new = preprocessing_pipeline.data_cleaning_before_resampling(temp_new)
        temp_new = preprocessing_pipeline.resample_to_hourly(temp_new)

        # Combine for lag/rolling context
        combined = pd.concat([temp_old, temp_new]).sort_index()
        
        # Feature engineering using history
        combined = preprocessing_pipeline.data_cleaning_after_resampling(combined)
        combined = preprocessing_pipeline.engineer_features(combined)

        # Keep only NEW engineered rows
        temp_new_eng = combined.loc[temp_new.index]
        temp_new_eng["Type"] = machine_type

        # Append safely
        temp_final = pd.concat([temp_old, temp_new_eng]).sort_index()
        temp_final = temp_final[~temp_final.index.duplicated(keep="first")]
        processed_frames.append(temp_final)

            
    final_df = pd.concat(processed_frames, axis=0)
        
    return final_df


def split_by_machine_type(final_df: pd.DataFrame):
    if "Type" not in final_df.columns:
        raise ValueError("Expected column 'Type' not found in dataset.")

    return {
        "YWNC2_CONE": final_df[final_df["Type"] == "YWNC2 CONE"].drop(columns="Type"),
        "YWNC2_CUP":  final_df[final_df["Type"] == "YWNC2 CUP"].drop(columns="Type"),
        "YWNC3_CONE": final_df[final_df["Type"] == "YWNC3 CONE"].drop(columns="Type"),
        "YWNC3_CUP":  final_df[final_df["Type"] == "YWNC3 CUP"].drop(columns="Type"),
    }


def run_recursive_forecasts(data_by_type: dict, models: dict, forecaster):
    """
    Runs WeeklyProfileRecursiveForecaster for each machine type.
    Returns one combined forecast DataFrame.
    """
    all_daily = []
    all_weekly = []

    for machine, df in data_by_type.items():
        print(f"🔮 Forecasting for {machine}...")

        model = models[machine]
        today_df, weekly_df = forecaster.build(df, model)  # <-- recursive forecast

        today_temp = today_df.copy()
        today_temp["machine_type"] = machine

        weekly_temp = weekly_df.copy()
        weekly_temp["machine_type"] = machine

        all_daily.append(today_temp)
        all_weekly.append(weekly_temp)

    final_daily = pd.concat(all_daily).sort_index()
    final_weekly = pd.concat(all_weekly).sort_index()
    return final_daily, final_weekly


def main():
    print("📦 Loading forecasting bundle...")
    bundle = load_bundle(BUNDLE_PATH)

    final_df = bundle["all_machine_type_data"]
    models = bundle["models"]
    forecaster = bundle["inputs_for_forcasting"]  # WeeklyProfileRecursiveForecaster
    # forecaster = WeeklyProfileRecursiveForecaster()

    # Look for concatinated data, If exist use this data as final_df
    final_df = load_newer_concatenated_df(final_df, CONCATINATED_DATA_PATH)

    # 🔥 JSON incremental update
    print("📥 Updating final_df with new JSON data...")
    final_df = update_final_df_with_json(JSON_PATH, final_df, bundle)
    final_df.drop_duplicates(inplace = True)
    final_df.to_csv("final_df.csv")
    
    print("🔀 Splitting data by machine type...")
    data_by_type = split_by_machine_type(final_df)

    
    print("🔮 Running recursive forecasts...")
    # today = date.today().strftime("%Y-%m-%d_%H-%M-%S")
    today = pd.Timestamp("2026-02-07 07:00:00").strftime("%Y-%m-%d_%H-%M-%S")
    daily_prediction, weekly_prediction = run_recursive_forecasts(data_by_type, models, forecaster)

    print("✅ Forecast DataFrame ready!")
    daily_prediction.to_csv(f"ForecastingData/{today}_daily_forecasting.csv")
    weekly_prediction.to_csv(f"ForecastingData/{today}_weekly_forecasting.csv")

if __name__ == "__main__":
    main()
