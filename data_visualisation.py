import pandas as pd
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("weather_final_data.csv")
data1 = pd.read_csv("preprocessed_weather_data.csv")

#WEATHER CONDITION DISTRIBUTION CHART USE preprocessed_weather_data.csv

'''data1['Weather Condition'].value_counts().plot.pie(autopct='%1.1f%%')

plt.title("Weather Condition Distribution")
plt.ylabel("")
plt.show()'''

#FEATURE IMPORTANCE GRAPH USE preprocessed_weather_data.csv

'''model = joblib.load("weather_model.pkl")

features = data1.drop('Weather Condition', axis=1).columns
importance = model.feature_importances_

plt.figure()
plt.bar(features, importance)

plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Weather Prediction")

plt.show()'''

# HUMIDITY VS WEATHER CONDITION GRAPH use weather_final_data.csv

'''plt.figure()

# Plot for each weather condition
for condition in data['Weather Condition'].unique():
    subset = data[data['Weather Condition'] == condition]
    plt.scatter(subset['humidity'], [condition]*len(subset))

plt.xlabel("Humidity (%)")   # ✅ Original values
plt.ylabel("Weather Condition")
plt.title("Humidity vs Weather Condition (Original Values)")

plt.show()'''

# WIND_SPEED VS WEATHER CONDITION use weather_final_data.csv

'''plt.figure()
for condition in data['Weather Condition'].unique():
    subset = data[data['Weather Condition'] == condition]
    plt.scatter(subset['wind_speed'], [condition]*len(subset))

plt.xlabel("Wind Speed")
plt.ylabel("Weather Condition")
plt.title("Wind Speed vs Weather Condition")

plt.show()'''

# TEMPERATURE VS WEATHER CONDITION GRAPH use weather_final_data.csv

'''plt.figure()
for condition in data['Weather Condition'].unique():
    subset = data[data['Weather Condition'] == condition]
    plt.scatter(subset['temperature'], [condition]*len(subset))

plt.xlabel("Temperature")
plt.ylabel("Weather Condition")
plt.title("Temperature vs Weather Condition")

plt.show()'''

# RAINY VS WEATHER CONDITION use weather_final_data.csv

'''plt.figure()
for condition in data['Weather Condition'].unique():
    subset = data[data['Weather Condition'] == condition]
    plt.scatter(subset['rain_mm_1min'], [condition]*len(subset))

plt.xlabel("Rain (mm)")
plt.ylabel("Weather Condition")
plt.title("Rain vs Weather Condition")

plt.show()'''



'''
🔵 Blue line	1	Rainy
🟢 Green line	2	Stormy
🔴 Red line	3	Sunny
🟠 Orange line	0	Cloudy
'''
