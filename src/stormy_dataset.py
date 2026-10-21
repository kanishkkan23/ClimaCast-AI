import pandas as pd
import requests
from io import StringIO
import time

"""
# ============================================================
# 1. NOAA STATIONS
# ============================================================

# You can add more NOAA station IDs here.
# Format:
# "Station ID": "City"

stations = {
    "72295023174": "Miami",
    "72278013874": "Orlando",
    "72270023174": "Tampa",
    "72202012839": "Houston",
    "72219012960": "New Orleans"
}


# ============================================================
# 2. YEARS TO SEARCH
# ============================================================

years = range(2018, 2026)


# ============================================================
# 3. STORAGE
# ============================================================

stormy_data = []


# ============================================================
# 4. DOWNLOAD NOAA DATA
# ============================================================

for station_id, city in stations.items():

    print("\n======================================")
    print("Searching:", city)
    print("Station:", station_id)
    print("======================================")

    for year in years:

        print("Downloading:", year)

        url = (
            "https://www.ncei.noaa.gov/data/"
            "global-hourly/access/"
            f"{year}/{station_id}.csv"
        )

        try:

            response = requests.get(
                url,
                timeout=60
            )

            # ------------------------------------------------
            # CHECK DOWNLOAD
            # ------------------------------------------------

            if response.status_code != 200:

                print(
                    "Download failed:",
                    response.status_code
                )

                continue


            # ------------------------------------------------
            # READ CSV
            # ------------------------------------------------

            df = pd.read_csv(
                StringIO(response.text),
                low_memory=False
            )


            print(
                "Total observations:",
                len(df)
            )


            # ------------------------------------------------
            # CHECK WEATHER COLUMNS
            # ------------------------------------------------

            weather_columns = [
                col for col in df.columns
                if col.startswith("AW")
                or col.startswith("MW")
            ]

            if len(weather_columns) == 0:

                print(
                    "No present-weather columns found."
                )

                continue


            # ------------------------------------------------
            # FIND THUNDERSTORM OBSERVATIONS
            # ------------------------------------------------

            storm_mask = pd.Series(
                False,
                index=df.index
            )


            for column in weather_columns:

                values = (
                    df[column]
                    .fillna("")
                    .astype(str)
                )

                # NOAA present-weather fields
                # may contain encoded weather information.
                #
                # Search for thunderstorm codes:
                # 95, 96, 97, 98, 99

                storm_mask |= values.str.contains(
                    r"(^|,|\s)(95|96|97|98|99)(,|\s|$)",
                    regex=True,
                    na=False
                )


            stormy = df[
                storm_mask
            ].copy()


            print(
                "Stormy observations:",
                len(stormy)
            )


            # ------------------------------------------------
            # IF NO STORMY DATA
            # ------------------------------------------------

            if len(stormy) == 0:

                time.sleep(3)

                continue


            # ------------------------------------------------
            # ADD CITY
            # ------------------------------------------------

            stormy["city"] = city


            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            stormy_data.append(
                stormy
            )


            # ------------------------------------------------
            # WAIT
            # ------------------------------------------------

            time.sleep(3)


        except Exception as e:

            print(
                "Error:",
                e
            )

            time.sleep(5)


# ============================================================
# 5. CHECK RESULT
# ============================================================

if len(stormy_data) == 0:

    print("\n======================================")
    print("NO STORMY OBSERVATIONS FOUND")
    print("======================================")

    print(
        "Try adding more stations or years."
    )

    exit()


# ============================================================
# 6. COMBINE DATA
# ============================================================

df = pd.concat(
    stormy_data,
    ignore_index=True
)


# ============================================================
# 7. CREATE TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["DATE"],
    errors="coerce"
)


# ============================================================
# 8. EXTRACT TEMPERATURE
# ============================================================

def extract_value(value, scale=10):

    try:

        if pd.isna(value):
            return None

        value = str(value)

        number = value.split(",")[0]

        number = float(number)

        # NOAA temperature values are normally
        # reported in tenths of degrees Celsius.

        return number / scale

    except:

        return None


df["temperature"] = df["TMP"].apply(
    extract_value
)


# ============================================================
# 9. EXTRACT DEW POINT
# ============================================================

if "DEW" in df.columns:

    df["dew_point"] = df["DEW"].apply(
        extract_value
    )

else:

    df["dew_point"] = None


# ============================================================
# 10. CALCULATE RELATIVE HUMIDITY
# ============================================================

def calculate_humidity(temp, dew):

    if pd.isna(temp) or pd.isna(dew):

        return None

    try:

        import math

        rh = (
            100 *
            math.exp(
                (17.625 * dew) /
                (243.04 + dew)
            )
            /
            math.exp(
                (17.625 * temp) /
                (243.04 + temp)
            )
        )

        return min(
            max(rh, 0),
            100
        )

    except:

        return None


df["humidity"] = df.apply(
    lambda row:
    calculate_humidity(
        row["temperature"],
        row["dew_point"]
    ),
    axis=1
)


# ============================================================
# 11. EXTRACT PRESSURE
# ============================================================

if "SLP" in df.columns:

    df["pressure"] = df["SLP"].apply(
        extract_value
    )

else:

    df["pressure"] = None


# ============================================================
# 12. EXTRACT WIND SPEED
# ============================================================

def extract_wind_speed(value):

    try:

        value = str(value)

        speed = value.split(",")[3]

        speed = float(speed)

        # NOAA wind speed is in tenths of m/s

        return speed / 10

    except:

        return None


if "WND" in df.columns:

    df["wind_speed"] = df["WND"].apply(
        extract_wind_speed
    )

else:

    df["wind_speed"] = None


# ============================================================
# 13. EXTRACT WIND GUST
# ============================================================

if "OC1" in df.columns:

    df["wind_gust"] = df["OC1"].apply(
        extract_wind_speed
    )

else:

    df["wind_gust"] = None


# ============================================================
# 14. EXTRACT HOURLY PRECIPITATION
# ============================================================

def extract_precipitation(value):

    try:

        value = str(value)

        amount = value.split(",")[0]

        amount = float(amount)

        # NOAA precipitation is generally
        # recorded in hundredths of inches
        # in the relevant precipitation section.

        return amount * 0.254

    except:

        return None


if "AA1" in df.columns:

    df["rain_mm_1hr"] = df["AA1"].apply(
        extract_precipitation
    )

else:

    df["rain_mm_1hr"] = None


# ============================================================
# 15. CREATE TARGET VARIABLE
# ============================================================

df["Weather Condition"] = "Stormy"


# ============================================================
# 16. SELECT FINAL COLUMNS
# ============================================================

final_columns = [
    "timestamp",
    "temperature",
    "humidity",
    "pressure",
    "rain_mm_1hr",
    "wind_speed",
    "wind_gust",
    "city",
    "Weather Condition"
]


final_columns = [
    col
    for col in final_columns
    if col in df.columns
]


stormy_dataset = df[
    final_columns
].copy()


# ============================================================
# 17. REMOVE INVALID TIMESTAMPS
# ============================================================

stormy_dataset.dropna(
    subset=["timestamp"],
    inplace=True
)


# ============================================================
# 18. REMOVE DUPLICATES
# ============================================================

stormy_dataset.drop_duplicates(
    subset=[
        "city",
        "timestamp"
    ],
    keep="first",
    inplace=True
)


# ============================================================
# 19. SORT DATA
# ============================================================

stormy_dataset.sort_values(
    by=[
        "city",
        "timestamp"
    ],
    inplace=True
)


stormy_dataset.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 20. DISPLAY DATASET
# ============================================================

print("\n======================================")
print("STORMY DATASET CREATED")
print("======================================")

print(
    "\nDataset Shape:",
    stormy_dataset.shape
)

print(
    "\nStormy observations by city:"
)

print(
    stormy_dataset["city"].value_counts()
)

print(
    "\nFirst 10 observations:"
)

print(
    stormy_dataset.head(10)
)


# ============================================================
# 21. SAVE CSV
# ============================================================

stormy_dataset.to_csv(
    "stormy_weather_dataset1.csv",
    index=False
)


print("\n======================================")
print("CSV CREATED SUCCESSFULLY")
print("======================================")

print(
    "File: stormy_weather_dataset.csv"
)"""

df = pd.read_csv(
    StringIO(response.text),
    low_memory=False
)

print("\nAvailable cloud-related columns:")
print([
    col for col in df.columns
    if "CLOUD" in col.upper()
    or "GA" in col.upper()
    or "GF" in col.upper()
    or "GD" in col.upper()
])
