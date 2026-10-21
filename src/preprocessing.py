# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("weather_final_data.csv")

print("Original Data:")
print(data.head())

print(data['Weather Condition'].value_counts())

# -------------------------------
# 2. Convert Timestamp to DateTime
# -------------------------------
data['timestamp'] = pd.to_datetime(data['timestamp'])


# -------------------------------
# 3. Remove Duplicates (IMPORTANT)
# Based on Timestamp (Best Approach)
# -------------------------------
print("\nDuplicate timestamps before:", data.duplicated(subset=['timestamp']).sum())

data.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)

print("Duplicate timestamps after:", data.duplicated(subset=['timestamp']).sum())


# -------------------------------
# 4. Handle Missing Values
# -------------------------------
print("\nMissing Values Before:")
print(data.isnull().sum())

# Fill numerical columns with mean
num_cols = data.select_dtypes(include=np.number).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

# Fill categorical columns with mode
cat_cols = data.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing Values After:")
print(data.isnull().sum())


#print("Original Data After Handling Missing Values: ")
#print(data.head())

# -------------------------------
# 5. Feature Engineering from Timestamp
# -------------------------------

# Extract time features
data['hour'] = data['timestamp'].dt.hour
data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month
data['year'] = data['timestamp'].dt.year

# Cyclical encoding
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)

data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# Drop original columns
data.drop(['timestamp', 'hour', 'day', 'month'], axis=1, inplace=True)


# -------------------------------
# 6. Encode Target Variable (Condition)
# -------------------------------
label_encoder = LabelEncoder()
data['Weather Condition'] = label_encoder.fit_transform(data['Weather Condition'])

print("\nEncoded Labels:")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


# -------------------------------
# 7. Feature Scaling
# -------------------------------
scaler = StandardScaler()

# Separate features and target
X = data.drop('Weather Condition', axis=1)
y = data['Weather Condition']

# Scale features
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Combine with target
final_data = pd.concat([X_scaled, y], axis=1)


# -------------------------------
# 8. Final Output
# -------------------------------
print("\nPreprocessed Data:")
print(final_data.head())

# Save cleaned dataset
final_data.to_csv("preprocessed_weather_data.csv", index=False)

print("\n✅ Data Preprocessing Completed Successfully!")
