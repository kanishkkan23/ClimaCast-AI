import pandas as pd
import numpy as np


# ============================================================
# 1. READ BOTH DATASETS
# ============================================================

openmeteo = pd.read_csv(
    "hourly_weather_dataset_covai.csv"
)

noaa = pd.read_csv(
    "cleaned_stormy_weather_dataset.csv"
)


print("Open-Meteo Dataset:")
print(openmeteo.shape)

print("\nNOAA Stormy Dataset:")
print(noaa.shape)


# ============================================================
# 2. DISPLAY ORIGINAL COLUMNS
# ============================================================

print("\nOpen-Meteo Columns:")
print(openmeteo.columns.tolist())

print("\nNOAA Columns:")
print(noaa.columns.tolist())


# ============================================================
# 3. STANDARDIZE OPEN-METEO COLUMNS
# ============================================================

openmeteo = openmeteo.rename(
    columns={
        "precipitation_mm": "rain_mm_1hr"
    }
)


# ============================================================
# 4. KEEP ONLY REQUIRED COLUMNS
# ============================================================

openmeteo = openmeteo[
    [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "rain_mm_1hr",
        "wind_speed",
        "cloud_cover",
        "Weather Condition"
    ]
]


# ============================================================
# 5. STANDARDIZE NOAA COLUMNS
# ============================================================

noaa = noaa[
    [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "rain_mm_1hr",
        "wind_speed",
        "Weather Condition"
    ]
]


# ============================================================
# 6. CONVERT TIMESTAMP
# ============================================================

openmeteo["timestamp"] = pd.to_datetime(
    openmeteo["timestamp"],
    errors="coerce"
)

noaa["timestamp"] = pd.to_datetime(
    noaa["timestamp"],
    errors="coerce"
)


# ============================================================
# 7. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "temperature",
    "humidity",
    "pressure",
    "rain_mm_1hr",
    "wind_speed",
    "cloud_cover"
]

for column in numeric_columns:

    openmeteo[column] = pd.to_numeric(
        openmeteo[column],
        errors="coerce"
    )

    noaa[column] = pd.to_numeric(
        noaa[column],
        errors="coerce"
    )


# ============================================================
# 8. REMOVE INVALID PRESSURE VALUES
# ============================================================

openmeteo.loc[
    (openmeteo["pressure"] < 800) |
    (openmeteo["pressure"] > 1100),
    "pressure"
] = np.nan


noaa.loc[
    (noaa["pressure"] < 800) |
    (noaa["pressure"] > 1100),
    "pressure"
] = np.nan


# ============================================================
# 9. COMBINE BOTH DATASETS
# ============================================================

final_dataset = pd.concat(
    [
        openmeteo,
        noaa
    ],
    ignore_index=True
)


# ============================================================
# 10. REMOVE DUPLICATES
# ============================================================

final_dataset.drop_duplicates(
    subset=[
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "rain_mm_1hr",
        "wind_speed",
        "cloud_cover",
        "Weather Condition"
    ],
    keep="first",
    inplace=True
)


# ============================================================
# 11. SORT BY TIMESTAMP
# ============================================================

final_dataset.sort_values(
    by="timestamp",
    inplace=True
)


final_dataset.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 12. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(
    final_dataset.isnull().sum()
)


# ============================================================
# 13. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\n==========================================")
print("WEATHER CONDITION DISTRIBUTION")
print("==========================================")

print(
    final_dataset[
        "Weather Condition"
    ].value_counts()
)


# ============================================================
# 14. DISPLAY FINAL DATASET INFORMATION
# ============================================================

print("\n==========================================")
print("FINAL DATASET INFORMATION")
print("==========================================")

print(
    "Dataset Shape:",
    final_dataset.shape
)

print(
    "\nColumns:"
)

print(
    final_dataset.columns.tolist()
)


print(
    "\nFirst 10 rows:"
)

print(
    final_dataset.head(10)
)


# ============================================================
# 15. SAVE FINAL DATASET
# ============================================================

final_dataset.to_csv(
    "final_weather_dataset1.csv",
    index=False
)


print("\n==========================================")
print("DATASETS MERGED SUCCESSFULLY!")
print("==========================================")

print(
    "File: final_weather_dataset.csv"
)
