# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# -------------------------------
# 1. Load Trained Model
# -------------------------------
model = joblib.load("weather_model.pkl")
scaler = joblib.load("scaler.pkl") # Save your trained model before this

# -------------------------------
# 2. Label Mapping (IMPORTANT)
# -------------------------------
# Same mapping used during training
label_map = {
    0: "Cloudy",
    1: "Rainy",
    2: "Stormy",
    3: "Sunny"
}

# -------------------------------
# 3. Take User Input
# -------------------------------
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
pressure = float(input("Enter Pressure: "))
rain = float(input("Enter Rain (mm): "))
wind_speed = float(input("Enter Wind Speed: "))
year = int(input("Enter Year: "))
hour = int(input("Enter Hour (0-23): "))
day = int(input("Enter Day (1-31): "))
month = int(input("Enter Month (1-12): "))


# -------------------------------
# 4. Convert Time Features → Cyclic Encoding
# -------------------------------

# Hour
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# Day
day_sin = np.sin(2 * np.pi * day / 31)
day_cos = np.cos(2 * np.pi * day / 31)

# Month
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)


# -------------------------------
# 5. Create DataFrame (same format as training)
# -------------------------------
input_data = pd.DataFrame([[
    temperature, humidity, pressure, rain, wind_speed, year,
    hour_sin, hour_cos,
    day_sin, day_cos,
    month_sin, month_cos
]], columns=[
    'temperature', 'humidity', 'pressure', 'rain_mm_1min', 'wind_speed', 'year',
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'month_sin', 'month_cos'
])


# -------------------------------
# 6. Prediction
# -------------------------------
input_scaled = scaler.transform(input_data)

# Convert back to DataFrame (IMPORTANT FIX)
input_scaled = pd.DataFrame(
    input_scaled,
    columns=input_data.columns
)

prediction = model.predict(input_scaled)

# Convert numeric → label
predicted_weather = label_map[prediction[0]]

# -------------------------------
# 7. Output
# -------------------------------
print("\n🌤️ Predicted Weather Condition:", predicted_weather)

print("\nModel Feature Importances: ")
print(model.feature_importances_)

print("\nModel Classes: ")
print(model.classes_)

print("Model Probabilities: ")
print(model.predict_proba(input_data))
