import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


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
        vehicle_type = random.choice(vehicle_types)
        # Truck/Bus/LCV tend to have bigger engines
        if vehicle_type in ["Truck", "Bus"]:
            engine_capacity = random.randint(3000, 8000)
        elif vehicle_type == "LCV":
            engine_capacity = random.randint(2000, 4000)
        else:
            engine_capacity = random.randint(1200, 2500)

        # Random current date in 2024
        current_date = datetime(2024, random.randint(1, 12), random.randint(1, 28))

        # Age of filter
        days_old = random.randint(0, 400)
        change_date = current_date - timedelta(days=days_old)

        # km proportional to age but with variation
        km_after_change = int(days_old * random.uniform(10, 200))

        road = random.choice(road_types)
        load = random.choice(load_types)
        fuel = random.choice(fuel_types)
        driving = random.choice(driving_styles)

        # Base oil temp
        base_temp = random.uniform(75, 110)
        # Highway or offroad often hotter
        if road in ["highway", "offroad"]:
            base_temp += random.uniform(0, 10)
        # Heavy load raises temp
        if load == "heavy":
            base_temp += random.uniform(5, 15)
        avg_oil_temp = min(max(base_temp, 70), 140)

        # Viscosity index (higher is better)
        oil_viscosity_index = random.uniform(30, 100)

        # Engine RPM
        if driving == "calm":
            engine_rpm_avg = random.randint(1500, 2500)
        elif driving == "normal":
            engine_rpm_avg = random.randint(1800, 3000)
        else:
            engine_rpm_avg = random.randint(2500, 4000)

        idling_percentage = random.randint(5, 40)
        ambient_temperature = random.uniform(15, 45)

        # NEW FEATURES
        # Oil pressure in bar (typical running range ~1–6 bar)
        oil_pressure = random.uniform(1.0, 6.5)
        # Low oil level is bad; in %
        oil_level_pct = random.randint(20, 100)
        # Coolant temp (°C)
        coolant_temp = random.uniform(75, 110)
        # If ambient is very hot or load heavy, bump coolant temp
        if ambient_temperature > 35 or load == "heavy":
            coolant_temp += random.uniform(0, 10)
        coolant_temp = min(coolant_temp, 120)

        # Base label from age
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

        # Stress score from other features
        stress = 0

        # Load & road
        if load == "heavy":
            stress += 1
        if road == "offroad":
            stress += 1
        if fuel == "Diesel":
            stress += 1
        if driving == "aggressive":
            stress += 1

        # Temps
        if avg_oil_temp > 110:
            stress += 1
        if coolant_temp > 100:
            stress += 1

        # Viscosity (too low is bad)
        if oil_viscosity_index < 40:
            stress += 1

        # RPM / idling
        if engine_rpm_avg > 3000:
            stress += 1
        if idling_percentage > 25:
            stress += 1

        # NEW: oil pressure extremes & low level
        if oil_pressure < 1.5 or oil_pressure > 5.5:
            stress += 1
        if oil_level_pct < 35:
            stress += 2  # low oil is quite harmful

        # Age-based index + stress-based escalation
        idx = condition_labels.index(base_label)
        # Roughly every 3 stress points escalate severity by 1 step
        idx = min(idx + stress // 3, len(condition_labels) - 1)
        final_label = condition_labels[idx]

        rows.append([
            vehicle_type,
            engine_capacity,
            change_date.date(),
            current_date.date(),
            days_old,
            km_after_change,
            road,
            load,
            avg_oil_temp,
            oil_viscosity_index,
            engine_rpm_avg,
            idling_percentage,
            ambient_temperature,
            fuel,
            driving,
            oil_pressure,
            oil_level_pct,
            coolant_temp,
            final_label
        ])

    df = pd.DataFrame(rows, columns=[
        "vehicle_type",
        "engine_capacity_cc",
        "oil_filter_change_date",
        "current_date",
        "oil_filter_age_days",
        "km_after_change",
        "road_type",
        "load_type",
        "avg_oil_temperature",
        "oil_viscosity_index",
        "engine_rpm_avg",
        "idling_percentage",
        "ambient_temperature",
        "fuel_type",
        "driving_style",
        "oil_pressure",
        "oil_level_pct",
        "coolant_temp",
        "oil_filter_condition"
    ])

    return df


if __name__ == "__main__":
    # 1. Generate dataset
    df = generate_synthetic_oil_filter_data(2000)
    df.to_csv("oil_filter_dataset_2000_extended.csv", index=False)
    print("Saved dataset: oil_filter_dataset_2000_extended.csv")
    
    # 2. Prepare features / target
    # Drop raw dates – we already use age in days
    df_model = df.drop(columns=["oil_filter_change_date", "current_date"])
    
    X = df_model.drop("oil_filter_condition", axis=1)
    y = df_model["oil_filter_condition"]
    
    # 3. Train-test split: 1600 train, 400 test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1600, test_size=400, shuffle=True, random_state=42
    )
    
    print("Train size:", X_train.shape[0])
    print("Test size :", X_test.shape[0])
    
    # 4. Preprocessing & model
    numeric_features = [
        "engine_capacity_cc",
        "oil_filter_age_days",
        "km_after_change",
        "avg_oil_temperature",
        "oil_viscosity_index",
        "engine_rpm_avg",
        "idling_percentage",
        "ambient_temperature",
        "oil_pressure",
        "oil_level_pct",
        "coolant_temp",
    ]
    
    categorical_features = [
        "vehicle_type",
        "road_type",
        "load_type",
        "fuel_type",
        "driving_style",
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    
    # 5. Train
    pipeline.fit(X_train, y_train)
    print("\nModel trained.")
    
    # 6. Evaluate
    y_pred = pipeline.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    
    # 7. Save model
    joblib.dump(pipeline, "oil_filter_model_extended.pkl")
    print("\nSaved model: oil_filter_model_extended.pkl")
