import pandas as pd
import joblib
from DataTransformationPipeline import RowDataPreprocessingPipeline
from ModelTrainer import ModelTrainerForEachMachineType
from FeatureBuilderForForcasting import WeeklyProfileRecursiveForecaster
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Data" / "Original_Data" / "EnergyParameterDataset.xlsx"
OUTPUT_PKL = "forecasting_ready_bundle.pkl"


def load_and_prepare_data(path: str):
    print("Loading and transforming raw data...")
    transformation = RowDataPreprocessingPipeline()
    final_df = transformation.load_and_transform_data(path)
    print(f"Data prepared. Shape: {final_df.shape}")
    return final_df


def split_by_machine_type(final_df: pd.DataFrame):
    print("Splitting data by machine type...")
    dfs = {
        "YWNC2_CONE": final_df[final_df["Type"] == "YWNC2 CONE"].drop(columns="Type"),
        "YWNC2_CUP":  final_df[final_df["Type"] == "YWNC2 CUP"].drop(columns="Type"),
        "YWNC3_CONE": final_df[final_df["Type"] == "YWNC3 CONE"].drop(columns="Type"),
        "YWNC3_CUP":  final_df[final_df["Type"] == "YWNC3 CUP"].drop(columns="Type"),
    }
    return dfs

def train_models(dfs: dict):
    print("Training models for each machine type...")
    trainer = ModelTrainerForEachMachineType(target_col="HOURLY_KWH")

    results = {}
    for name, df in dfs.items():
        results[name] = trainer.train_pipeline_for_dataset(df, name)

    models = {name: res["model"] for name, res in results.items()}
    return models


def build_bundle(final_df, models):
    print("Building bundle...")
    feature_cols = final_df.drop(columns=["HOURLY_KWH", "Type"]).columns.tolist()

    bundle = {
        "data_pipeline": RowDataPreprocessingPipeline(),
        "model_trainer": ModelTrainerForEachMachineType(),
        "inputs_for_forcasting": WeeklyProfileRecursiveForecaster(feature_columns=feature_cols),
        "all_machine_type_data": final_df,
        "models": models,
        "feature_columns": feature_cols,
        "target_column": "HOURLY_KWH",
        "machine_types": list(models.keys())
    }

    return bundle


def save_bundle(bundle, output_path):
    joblib.dump(bundle, output_path)
    print(f"💾 Bundle saved successfully as: {output_path}")


def main():
    final_df = load_and_prepare_data(DATA_PATH)
    dfs = split_by_machine_type(final_df)
    models = train_models(dfs)
    bundle = build_bundle(final_df, models)
    save_bundle(bundle, OUTPUT_PKL)


if __name__ == "__main__":
    main()
