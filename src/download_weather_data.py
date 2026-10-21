import requests
import pandas as pd

# ============================================================
# 1. LOCATION DETAILS
# ============================================================

latitude = 11.0168
longitude = 76.9558

start_date = "2024-01-01"
end_date = "2025-12-31"

# ============================================================
# 2. OPEN-METEO HISTORICAL WEATHER API
# ============================================================

url = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={latitude}"
    f"&longitude={longitude}"
    f"&start_date={start_date}"
    f"&end_date={end_date}"
    "&hourly="
    "temperature_2m,"
    "relative_humidity_2m,"
    "pressure_msl,"
    "precipitation,"
    "wind_speed_10m,"
    "cloud_cover,"
    "weather_code"
    "&timezone=Asia/Kolkata"
    "&temperature_unit=celsius"
    "&wind_speed_unit=ms"
    "&precipitation_unit=mm"
)

# ============================================================
# 3. REQUEST DATA FROM OPEN-METEO
# ============================================================

response = requests.get(url)

if response.status_code != 200:
    print("Error while fetching data!")
    print(response.text)
    exit()

weather_data = response.json()

print("Weather data downloaded successfully!")

# ============================================================
# 4. CONVERT TO DATAFRAME
# ============================================================

df = pd.DataFrame(weather_data["hourly"])

# ============================================================
# 5. RENAME COLUMNS
# ============================================================

df.rename(columns={
    "time": "timestamp",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "pressure_msl": "pressure",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_speed",
    "cloud_cover": "cloud_cover",
    "weather_code": "weather_code"
}, inplace=True)

# ============================================================
# 6. CONVERT TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ============================================================
# 7. FUNCTION TO CONVERT WEATHER CODE
# ============================================================

def get_weather_condition(code):

    # Sunny / Clear
    if code in [0, 1]:
        return "Sunny"

    # Mainly clear, partly cloudy, overcast
    elif code in [2, 3]:
        return "Cloudy"

    # Fog
    elif code in [45, 48]:
        return "Cloudy"

    # Drizzle / Rain
    elif code in [51, 53, 55, 56, 57,
                  61, 63, 65, 66, 67,
                  80, 81, 82]:
        return "Rainy"

    # Snow
    elif code in [71, 73, 75, 77, 85, 86]:
        return "Rainy"

    # Thunderstorm
    elif code in [95, 96, 99]:
        return "Stormy"

    # Unknown condition
    else:
        return "Cloudy"


# ============================================================
# 8. CREATE TARGET VARIABLE
# ============================================================

df["Weather Condition"] = df["weather_code"].apply(
    get_weather_condition
)

# ============================================================
# 9. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# 10. REMOVE DUPLICATE TIMESTAMPS
# ============================================================

df.drop_duplicates(
    subset=["timestamp"],
    keep="first",
    inplace=True
)

# ============================================================
# 11. SORT BY TIMESTAMP
# ============================================================

#df.sort_values(
 #   by="timestamp",
  #  inplace=True
#)

#df.reset_index(drop=True, inplace=True)

# ============================================================
# 12. DISPLAY DATASET INFORMATION
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nWeather Condition Distribution:")
print(df["Weather Condition"].value_counts())

# ============================================================
# 13. SAVE DATASET
# ============================================================

df.to_csv(
    "hourly_weather_dataset_covai.csv",
    index=False
)

print("\nDataset saved successfully!")
print("File: hourly_weather_dataset.csv")
