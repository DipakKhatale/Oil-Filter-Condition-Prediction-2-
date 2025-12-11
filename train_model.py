import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib


# ==========================================================
#  FIXED & IMPROVED SENSOR-DRIVEN DATA GENERATION
# ==========================================================
def generate_synthetic_oil_filter_data(n_rows=2000, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)

    vehicle_types = ["SUV", "Sedan", "Hatchback", "LCV", "Truck", "Bus"]
    road_types = ["city", "highway", "offroad", "mixed"]
    load_types = ["light", "medium", "heavy"]
    fuel_types = ["Petrol", "Diesel", "CNG"]
    driving_styles = ["calm", "normal", "aggressive"]

    condition_labels = ["Green", "Light-Green", "Yellow", "Orange", "Dark-Orange", "Red"]

    rows = []

    for _ in range(n_rows):

        #-------------- RANDOM FEATURES --------------
        vehicle_type = random.choice(vehicle_types)
        engine_capacity = random.randint(1000, 8000)
        current_date = datetime(2024, random.randint(1, 12), random.randint(1, 28))

        days_old = random.randint(0, 400)
        change_date = current_date - timedelta(days=days_old)
        km_after_change = int(days_old * random.uniform(10, 200))

        road = random.choice(road_types)
        load = random.choice(load_types)
        fuel = random.choice(fuel_types)
        driving = random.choice(driving_styles)

        # Temperature, pressure, coolant more extreme now
        avg_oil_temp = np.random.normal(100, 15) + (10 if load == "heavy" else 0)
        avg_oil_temp = max(60, min(avg_oil_temp, 160))

        coolant_temp = np.random.normal(95, 10)
        coolant_temp = max(70, min(coolant_temp, 140))

        oil_pressure = np.random.uniform(0.8, 7.5)
        oil_level_pct = random.randint(15, 100)

        oil_viscosity_index = random.uniform(30, 100)
        engine_rpm_avg = random.randint(1500, 4000)
        idling_percentage = random.randint(5, 40)
        ambient_temperature = random.uniform(15, 50)

        #-------------- BASE AGE-DRIVEN LABEL --------------
        if days_old <= 30:
            base_label = "Green"
        elif days_old <= 90:
            base_label = "Light-Green"
        elif days_old <= 180:
            base_label = "Yellow"
        elif days_old <= 270:
            base_label = "Orange"
        elif days_old <= 330:
            base_label = "Dark-Orange"
        else:
            base_label = "Red"

        #-------------- SENSOR STRESS ----------------
        stress = 0

        # Strong boosted signals
        if avg_oil_temp > 140: stress += 12
        elif avg_oil_temp > 135: stress += 8
        elif avg_oil_temp > 125: stress += 5
        elif avg_oil_temp > 115: stress += 3

        if coolant_temp > 130: stress += 10
        elif coolant_temp > 120: stress += 7
        elif coolant_temp > 110: stress += 5
        elif coolant_temp > 100: stress += 3

        if oil_pressure < 1.0 or oil_pressure > 7.0: stress += 10
        elif oil_pressure < 1.5 or oil_pressure > 6.5: stress += 6

        if oil_level_pct < 20: stress += 12
        elif oil_level_pct < 30: stress += 7
        elif oil_level_pct < 40: stress += 4

        if engine_rpm_avg > 3500: stress += 3
        if idling_percentage > 30: stress += 2
        if load == "heavy": stress += 3
        if driving == "aggressive": stress += 3

        #-------------- HARD OVERRIDES --------------
        if (
            avg_oil_temp > 140 or
            coolant_temp > 130 or
            oil_pressure < 1.0 or
            oil_pressure > 7.0 or
            oil_level_pct < 20
        ):
            final_label = "Red"
        else:
            idx = condition_labels.index(base_label)
            idx = min(idx + stress // 4, len(condition_labels) - 1)
            final_label = condition_labels[idx]

        rows.append([
            vehicle_type, engine_capacity, change_date.date(), current_date.date(),
            days_old, km_after_change, road, load, avg_oil_temp, oil_viscosity_index,
            engine_rpm_avg, idling_percentage, ambient_temperature, fuel, driving,
            oil_pressure, oil_level_pct, coolant_temp, final_label
        ])

    df = pd.DataFrame(rows, columns=[
        "vehicle_type", "engine_capacity_cc", "oil_filter_change_date", "current_date",
        "oil_filter_age_days", "km_after_change", "road_type", "load_type",
        "avg_oil_temperature", "oil_viscosity_index", "engine_rpm_avg",
        "idling_percentage", "ambient_temperature", "fuel_type", "driving_style",
        "oil_pressure", "oil_level_pct", "coolant_temp", "oil_filter_condition"
    ])

    return df


# ==========================================================
# TRAINING
# ==========================================================
if __name__ == "__main__":

    df = generate_synthetic_oil_filter_data(2000)
    df.to_csv("oil_filter_dataset.csv", index=False)

    df_model = df.drop(columns=["oil_filter_change_date", "current_date"])

    X = df_model.drop("oil_filter_condition", axis=1)
    y = df_model["oil_filter_condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1600, test_size=400, random_state=42
    )

    numeric_features = [
        "engine_capacity_cc", "oil_filter_age_days", "km_after_change",
        "avg_oil_temperature", "oil_viscosity_index", "engine_rpm_avg",
        "idling_percentage", "ambient_temperature", "oil_pressure",
        "oil_level_pct", "coolant_temp"
    ]

    categorical_features = [
        "vehicle_type", "road_type", "load_type", "fuel_type", "driving_style"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=28,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # --------------------------
    #  OUTPUT METRICS
    # --------------------------
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    plain_f1 = f1_score(y_test, y_pred, average="micro")
    print("\nPlain F1 Score:", plain_f1)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm, index=[f"Actual {c}" for c in sorted(y.unique())],
        columns=[f"Pred {c}" for c in sorted(y.unique())]
    )

    print("\nCONFUSION MATRIX TABLE:")
    print(cm_df)

    joblib.dump(pipeline, "oil_filter_model.pkl")
    print("\nModel saved: oil_filter_model.pkl")
