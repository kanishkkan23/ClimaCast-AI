# ============================================================
# WEATHER PREDICTION SYSTEM
# FINAL DATA PREPROCESSING
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv(
    "quality_checked_weather_dataset.csv"
)

print("Dataset Loaded Successfully!")

print("\nOriginal Dataset Shape:")
print(data.shape)


# ============================================================
# 2. TIMESTAMP → DATETIME
# ============================================================

data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    errors="coerce"
)

print("\nTimestamp converted to datetime successfully.")


# ============================================================
# 4. EXTRACT DATE AND TIME FEATURES
# ============================================================

data["year"] = data["timestamp"].dt.year

data["month"] = data["timestamp"].dt.month

data["day"] = data["timestamp"].dt.day

data["hour"] = data["timestamp"].dt.hour


print("\nTime-based features created:")
print("year, month, day, hour")


# ============================================================
# 5. CYCLICAL FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# HOUR
# ------------------------------------------------------------

data["hour_sin"] = np.sin(
    2 * np.pi * data["hour"] / 24
)

data["hour_cos"] = np.cos(
    2 * np.pi * data["hour"] / 24
)


# ------------------------------------------------------------
# DAY
# ------------------------------------------------------------

data["day_sin"] = np.sin(
    2 * np.pi * (data["day"] - 1) / 31
)

data["day_cos"] = np.cos(
    2 * np.pi * (data["day"] - 1) / 31
)


# ------------------------------------------------------------
# MONTH
# ------------------------------------------------------------

data["month_sin"] = np.sin(
    2 * np.pi * (data["month"] - 1) / 12
)

data["month_cos"] = np.cos(
    2 * np.pi * (data["month"] - 1) / 12
)


print("\nCyclical features created:")
print(
    "hour_sin, hour_cos, "
    "day_sin, day_cos, "
    "month_sin, month_cos"
)


# ============================================================
# 6. REMOVE UNNECESSARY TEMPORARY FEATURES
# ============================================================

# We keep:
# timestamp → useful for future LSTM/time-series forecasting
#
# We can remove:
# hour, day, month
# because their cyclical versions represent their periodicity.

data.drop(
    columns=[
        "hour",
        "day",
        "month"
    ],
    inplace=True
)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

numeric_columns = [
    "temperature",
    "humidity",
    "pressure",
    "rain_mm_1hr",
    "wind_speed",
    "year",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos"
]


for column in numeric_columns:

    if data[column].isnull().sum() > 0:

        data[column] = data[column].fillna(
            data[column].median()
        )


print("\nMissing numerical values handled.")


# ============================================================
# 8. CHECK TARGET VARIABLE
# ============================================================

print("\nWeather Condition Distribution:")
print(
    data["Weather Condition"].value_counts()
)


# ============================================================
# 9. LABEL ENCODING TARGET VARIABLE
# ============================================================

label_encoder = LabelEncoder()

data["Weather Condition"] = label_encoder.fit_transform(
    data["Weather Condition"]
)


print("\nLabel Encoding Completed.")

print("\nLabel Mapping:")

for number, label in enumerate(
    label_encoder.classes_
):
    print(
        label,
        "→",
        number
    )


"""# ============================================================
# 10. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

data[numeric_columns] = scaler.fit_transform(
    data[numeric_columns]
)


print("\nFeature Scaling Completed.")"""


# ============================================================
# 11. DISPLAY FINAL DATASET
# ============================================================

print("\n==============================================")
print("FINAL PREPROCESSED DATASET")
print("==============================================")

print(
    "\nDataset Shape:"
)

print(
    data.shape
)


print(
    "\nColumns:"
)

print(
    data.columns.tolist()
)


print(
    "\nFirst 5 rows:"
)

print(
    data.head()
)


# ============================================================
# 12. SAVE PREPROCESSED DATASET
# ============================================================

data.to_csv(
    "preprocessed_weather_data.csv",
    index=False
)


# ============================================================
# 13. SAVE LABEL ENCODER
# ============================================================

import joblib

joblib.dump(
    label_encoder,
    "weather_label_encoder.pkl"
)


"""# ============================================================
# 14. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "weather_scaler.pkl"
)
"""

print("\n==============================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("==============================================")

print(
    "\nDataset saved as:"
)

print(
    "preprocessed_weather_data.csv"
)

print(
    "\nLabel encoder saved as:"
)

print(
    "weather_label_encoder.pkl"
)

"""print(
    "\nScaler saved as:"
)

print(
    "weather_scaler.pkl"
)"""
