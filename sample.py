import pandas as pd

data = pd.read_csv(
    "hourly_weather_dataset_covai.csv"
)


print("\nWeather Condition Distribution:")
print(
    data["Weather Condition"].value_counts()
)


print("\nSunny vs Cloudy Statistics:")

sunny_cloudy = data[
    data["Weather Condition"].isin(
        ["Sunny", "Cloudy"]
    )
]


print(
    sunny_cloudy.groupby(
        "Weather Condition"
    )[
        [
            "temperature",
            "humidity",
            "pressure",
            "precipitation_mm",
            "wind_speed",
            "cloud_cover",
        ]
    ].mean()
)


print("\nSunny Cloud Cover Range:")

print(
    sunny_cloudy[
        sunny_cloudy["Weather Condition"] == "Sunny"
    ]["cloud_cover"].describe()
)


print("\nCloudy Cloud Cover Range:")

print(
    sunny_cloudy[
        sunny_cloudy["Weather Condition"] == "Cloudy"
    ]["cloud_cover"].describe()
)
