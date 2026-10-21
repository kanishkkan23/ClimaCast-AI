# ============================================================
# WEATHER PREDICTION SYSTEM
# MODEL TRAINING & TESTING MODULE
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================

data = pd.read_csv(
    "preprocessed_weather_data.csv"
)

print("Dataset Loaded Successfully!")

print("\nDataset Shape:")
print(data.shape)


# ============================================================
# 2. DISPLAY WEATHER CLASS DISTRIBUTION
# ============================================================

print("\nWeather Condition Distribution:")
print(
    data["Weather Condition"].value_counts()
)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(
    "Weather Condition",
    axis=1
)

y = data["Weather Condition"]


# ============================================================
# 4. REMOVE TIMESTAMP FROM RANDOM FOREST INPUT
# ============================================================

# Timestamp is kept in the dataset because it will be useful
# later for LSTM / time-series forecasting.
#
# But Random Forest should use the engineered numerical
# time features instead of the raw timestamp.

if "timestamp" in X.columns:

    X = X.drop(
        "timestamp",
        axis=1
    )


# ============================================================
# 5. SAVE FEATURE COLUMN ORDER
# ============================================================

feature_columns = X.columns.tolist()

print("\nFeatures Used by Random Forest:")
print(feature_columns)


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n==============================================")
print("TRAINING AND TESTING DATA")
print("==============================================")

print(
    "Training Data:",
    X_train.shape
)

print(
    "Testing Data :",
    X_test.shape
)


# ============================================================
# 7. CREATE RANDOM FOREST CLASSIFIER
# ============================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 8. MODEL TRAINING
# ============================================================

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed!")


# ============================================================
# 9. MODEL PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 10. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================================")
print("MODEL PERFORMANCE")
print("==============================================")

print(
    "\nAccuracy:",
    accuracy
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

print("\n==============================================")
print("FEATURE IMPORTANCE")
print("==============================================")


feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})


feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)


print(
    feature_importance
)


# ============================================================
# 14. MODEL CLASSES
# ============================================================

print("\n==============================================")
print("MODEL CLASSES")
print("==============================================")

print(
    model.classes_
)


# ============================================================
# 15. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "weather_model.pkl"
)

print(
    "\nTrained model saved as: weather_model.pkl"
)


# ============================================================
# 16. SAVE FEATURE COLUMN ORDER
# ============================================================

joblib.dump(
    feature_columns,
    "weather_feature_columns.pkl"
)

print(
    "Feature columns saved as: weather_feature_columns.pkl"
)


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("TRAINING & TESTING COMPLETED SUCCESSFULLY!")
print("==============================================")
