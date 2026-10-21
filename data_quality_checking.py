# ============================================================
# FINAL WEATHER DATASET - DATA QUALITY CHECK
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD FINAL DATASET
# ============================================================

data = pd.read_csv("final_weather_dataset.csv")

print("==============================================")
print("FINAL WEATHER DATASET QUALITY CHECK")
print("==============================================")


# ============================================================
# 2. DATASET SHAPE
# ============================================================

print("\n1. DATASET SHAPE")
print("----------------------------------------------")

print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])


# ============================================================
# 3. COLUMN NAMES
# ============================================================

print("\n2. COLUMN NAMES")
print("----------------------------------------------")

print(data.columns.tolist())


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "timestamp",
    "temperature",
    "humidity",
    "pressure",
    "rain_mm_1hr",
    "wind_speed",
    "Weather Condition"
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if len(missing_columns) == 0:
    print("All required columns are present.")
else:
    print("Missing columns:", missing_columns)


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n3. DATA TYPES")
print("----------------------------------------------")

print(data.dtypes)


# ============================================================
# 6. CONVERT TIMESTAMP
# ============================================================

data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    errors="coerce"
)

print("\nInvalid timestamps:",
      data["timestamp"].isna().sum())


# ============================================================
# 7. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "temperature",
    "humidity",
    "pressure",
    "rain_mm_1hr",
    "wind_speed"
]

for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\n4. MISSING VALUES")
print("----------------------------------------------")

missing_values = data.isnull().sum()
print("Before Removing the Missing Values: \n")
print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values are present.")

print("After Removing the Missing Values: \n")
# Fill numerical columns with mean
num_cols = data.select_dtypes(include=np.number).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
missing_values = data.isnull().sum()
print(missing_values)


# ============================================================
# 9. DUPLICATE ROWS
# ============================================================

print("\n5. DUPLICATE ROWS")
print("----------------------------------------------")

duplicate_rows = data.duplicated().sum()

print("Duplicate rows:", duplicate_rows)


# ============================================================
# 10. DUPLICATE TIMESTAMPS
# ============================================================

print("\n6. DUPLICATE TIMESTAMPS")
print("----------------------------------------------")

duplicate_timestamps = data.duplicated(
    subset=["timestamp"]
).sum()

print(
    "Duplicate timestamps:",
    duplicate_timestamps
)


# ============================================================
# 11. TEMPERATURE VALIDATION
# ============================================================

print("\n7. TEMPERATURE CHECK")
print("----------------------------------------------")

invalid_temperature = data[
    (data["temperature"] < -90) |
    (data["temperature"] > 60)
]

print(
    "Invalid temperature rows:",
    len(invalid_temperature)
)

if len(invalid_temperature) > 0:
    print(invalid_temperature[
        ["timestamp", "temperature"]
    ].head())


# ============================================================
# 12. HUMIDITY VALIDATION
# ============================================================

print("\n8. HUMIDITY CHECK")
print("----------------------------------------------")

invalid_humidity = data[
    (data["humidity"] < 0) |
    (data["humidity"] > 100)
]

print(
    "Invalid humidity rows:",
    len(invalid_humidity)
)

if len(invalid_humidity) > 0:
    print(invalid_humidity[
        ["timestamp", "humidity"]
    ].head())


# ============================================================
# 13. PRESSURE VALIDATION
# ============================================================

print("\n9. PRESSURE CHECK")
print("----------------------------------------------")

invalid_pressure = data[
    (data["pressure"] < 800) |
    (data["pressure"] > 1100)
]

print(
    "Invalid pressure rows:",
    len(invalid_pressure)
)

if len(invalid_pressure) > 0:
    print(invalid_pressure[
        ["timestamp", "pressure"]
    ].head())


# ============================================================
# 14. RAINFALL VALIDATION
# ============================================================

print("\n10. RAINFALL CHECK")
print("----------------------------------------------")

invalid_rain = data[
    data["rain_mm_1hr"] < 0
]

print(
    "Negative rainfall rows:",
    len(invalid_rain)
)

if len(invalid_rain) > 0:
    print(invalid_rain[
        ["timestamp", "rain_mm_1hr"]
    ].head())


# ============================================================
# 15. WIND SPEED VALIDATION
# ============================================================

print("\n11. WIND SPEED CHECK")
print("----------------------------------------------")

invalid_wind = data[
    data["wind_speed"] < 0
]

print(
    "Negative wind speed rows:",
    len(invalid_wind)
)

if len(invalid_wind) > 0:
    print(invalid_wind[
        ["timestamp", "wind_speed"]
    ].head())


# ============================================================
# 16. CHECK INFINITE VALUES
# ============================================================

print("\n12. INFINITE VALUES")
print("----------------------------------------------")

infinite_values = np.isinf(
    data[numeric_columns]
).sum()

print(infinite_values)

if infinite_values.sum() == 0:
    print("No infinite values found.")
else:
    print("Infinite values are present.")


# ============================================================
# 17. WEATHER CONDITION CHECK
# ============================================================

print("\n13. WEATHER CONDITION CHECK")
print("----------------------------------------------")

print(
    data["Weather Condition"].value_counts()
)


# ============================================================
# 18. CHECK FOUR CLASSES
# ============================================================

expected_classes = {
    "Sunny",
    "Cloudy",
    "Rainy",
    "Stormy"
}

actual_classes = set(
    data["Weather Condition"].dropna().unique()
)

print("\nExpected classes:")
print(expected_classes)

print("\nActual classes:")
print(actual_classes)


missing_classes = (
    expected_classes - actual_classes
)

if len(missing_classes) == 0:
    print("\nAll four weather classes are present.")
else:
    print(
        "\nMissing weather classes:",
        missing_classes
    )


# ============================================================
# 19. STATISTICAL SUMMARY
# ============================================================

print("\n14. STATISTICAL SUMMARY")
print("----------------------------------------------")

print(
    data[numeric_columns].describe()
)


# ============================================================
# 20. FINAL QUALITY SUMMARY
# ============================================================

print("\n==============================================")
print("FINAL DATA QUALITY SUMMARY")
print("==============================================")


checks = {
    "Required columns": len(missing_columns) == 0,
    "Missing values": data.isnull().sum().sum() == 0,
    "Duplicate rows": duplicate_rows == 0,
    "Duplicate timestamps": duplicate_timestamps != 0,
    "Temperature": len(invalid_temperature) == 0,
    "Humidity": len(invalid_humidity) == 0,
    "Pressure": len(invalid_pressure) == 0,
    "Rainfall": len(invalid_rain) == 0,
    "Wind speed": len(invalid_wind) == 0,
    "Infinite values": infinite_values.sum() == 0,
    "Four weather classes": len(missing_classes) == 0
}


for check, result in checks.items():

    if result:
        print("PASS :", check)
    else:
        print("FAIL :", check)


# ============================================================
# 21. OVERALL RESULT
# ============================================================

print("\n==============================================")

if all(checks.values()):

    print("FINAL RESULT: DATASET PASSED QUALITY CHECK")
    print("Dataset is ready for preprocessing and ML.")

else:

    print("FINAL RESULT: DATASET NEEDS CLEANING")
    print("Fix the failed checks before ML training.")

print("==============================================")


# ============================================================
# 22. SAVE QUALITY-CHECKED DATASET
# ============================================================

data.to_csv(
    "quality_checked_weather_dataset.csv",
    index=False
)

print(
    "\nSaved as: quality_checked_weather_dataset.csv"
)
