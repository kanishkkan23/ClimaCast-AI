# Import libraries
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# 1. Load Preprocessed Data
# -------------------------------
data = pd.read_csv("preprocessed_weather_data.csv")

print("Dataset Loaded Successfully!")


# -------------------------------
# 2. Split Features & Target
# -------------------------------
X = data.drop('Weather Condition', axis=1)
y = data['Weather Condition']


# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# -------------------------------
# 4. Train Model (Random Forest)
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# -------------------------------
# 5. Model Testing
# -------------------------------
y_pred = model.predict(X_test)


# -------------------------------
# 6. Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "weather_model.pkl")
print(type(model))
