# Energy Consumption Forecasting System

## Project Overview
This project focuses on forecasting electrical energy consumption using historical time-series data collected at an hourly resolution. The goal is to build a robust, production-ready forecasting pipeline that handles real-world challenges such as missing data, long gaps, outliers, and non-linear consumption patterns.

The system applies advanced feature engineering, time-series–aware validation, and machine learning models to accurately predict future energy usage.

---

## Objectives
- Forecast hourly energy consumption (kWh)
- Handle missing timestamps and long data gaps
- Reduce noise and impact of outliers
- Compare performance of statistical and ML-based models
- Build a reusable forecasting pipeline suitable for real-world energy systems

---

## Key Features & Techniques

### Time-Series Preprocessing
- Resampled raw data to hourly frequency
- Removed duplicate timestamps
- Identified and handled long gaps using:
  - Gap detection flags
  - `time_since_gap` feature
- Converted invalid zero readings to `NaN` where appropriate

### Feature Engineering
- Temporal features:
  - Hour, Day, Weekday
  - Sine–Cosine encoding for cyclic time patterns
- Rolling statistics:
  - Rolling mean (e.g., 7-hour window)
- Electrical parameters:
  - Average current
  - Line voltage (LN / LL)
  - Frequency
- Gap-aware features:
  - `long_gap_flag`
  - Time since last valid reading
- Outlier handling using capping (winsorization)

---

## Models Implemented
- Linear Regression (baseline)
- Random Forest Regressor
- Model validation using `TimeSeriesSplit` to prevent data leakage

---

## Evaluation Metrics
- R² Score
- Mean Squared Error (MSE)
- Comparison of training vs validation performance to detect overfitting

---

## Forecasting Strategy
- Generated next-day hourly forecasts (24 hours)
- Used latest known values for non-temporal features
- Maintained feature consistency between training and prediction pipelines

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn

---

## Use Cases
- Smart energy management systems
- Load forecasting for buildings or industries
- Demand planning and anomaly detection
- Foundation for real-time EMS (Energy Management System)

---

## Future Enhancements
- Add holiday & weather-based features
- Real-time inference with streaming data
- Deploy as a REST API or dashboard
