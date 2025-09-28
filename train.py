import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# Original 5 rows
original_data = pd.DataFrame({
    "engine": [1200, 1500, 1800, 2000, 1300],
    "weight": [1100, 1300, 1500, 1600, 1200],
    "horsepower": [85, 100, 120, 140, 90],
    "fuel": ["petrol", "diesel", "petrol", "diesel", "petrol"],
    "speed": [60, 80, 100, 120, 70],
    "mileage": [17.5, 20.0, 15.0, 13.0, 18.0]
})

# Generate 100 additional synthetic rows
n = 100

# Randomly choose fuel type
fuel_choices = ["petrol", "diesel"]
fuel = np.random.choice(fuel_choices, size=n)

# Generate engine size based on fuel type (diesel tends to be larger)
engine = np.where(
    fuel == "diesel",
    np.random.randint(1400, 2601, size=n),  # Diesel: 1400-2600 cc
    np.random.randint(800, 2001, size=n)    # Petrol: 800-2000 cc
)

# Generate weight correlated with engine size
weight = (800 + 0.4 * engine + np.random.normal(0, 100, size=n)).astype(int)
weight = np.clip(weight, 900, 2000)  # Reasonable bounds

# Generate horsepower correlated with engine and fuel
hp_base = engine * 0.07
hp_adjust = np.where(fuel == "diesel", -5, 0)
horsepower = (hp_base + hp_adjust + np.random.normal(0, 8, size=n)).astype(int)
horsepower = np.clip(horsepower, 60, 200)

# Generate speed correlated with horsepower
speed = (40 + 0.45 * horsepower + np.random.normal(0, 5, size=n)).astype(int)
speed = np.clip(speed, 50, 180)

# Generate mileage with realistic dependencies
mileage = np.where(
    fuel == "diesel",
    18.0 + (2000 - engine) * 0.002 + (1600 - weight) * 0.003 + np.random.normal(0, 1.0, size=n),
    15.0 + (1800 - engine) * 0.0025 + (1500 - weight) * 0.0025 + np.random.normal(0, 1.2, size=n)
)
mileage = np.clip(mileage, 8.0, 28.0)

# Create synthetic DataFrame
synthetic_data = pd.DataFrame({
    "engine": engine,
    "weight": weight,
    "horsepower": horsepower,
    "fuel": fuel,
    "speed": speed,
    "mileage": mileage
})

# Combine original and synthetic data
data = pd.concat([original_data, synthetic_data], ignore_index=True)

# Encode fuel type
le = LabelEncoder()
data["fuel"] = le.fit_transform(data["fuel"])  # petrol=1, diesel=0 for example

# Features & Target
X = data[["engine", "weight", "horsepower", "fuel", "speed"]]
y = data["mileage"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model as pickle
with open("car_mileage_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save label encoder too
with open("fuel_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
