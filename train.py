import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# -------------------------------
# 1️⃣ Load Dataset from CSV
# -------------------------------
data = pd.read_csv("Car_Mileage_Data.csv")  # 👈 change filename if needed

print("\n✅ Dataset loaded successfully!")
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
print("\nFirst 5 rows:\n", data.head())

# -------------------------------
# 2️⃣ Preprocessing
# -------------------------------

# Check for categorical columns (example: 'fuel')
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Encode categorical columns (if any)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target
target_column = "MPG"  # 👈 change target column name if different
X = data.drop(columns=[target_column])
y = data[target_column]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to DataFrame for saving
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df[target_column] = y.values

# Save preprocessed dataset
X_scaled_df.to_csv("Preprocessed_Car_Mileage_Data.csv", index=False)
print("\n💾 Preprocessed data saved as 'Preprocessed_Car_Mileage_Data.csv'")

# -------------------------------
# 3️⃣ Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df.drop(columns=[target_column]),
    X_scaled_df[target_column],
    test_size=0.2,
    random_state=42
)

# -------------------------------
# 4️⃣ Train Model
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 5️⃣ Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
accuracy_from_mae = (1 - (mae / np.mean(y_test))) * 100

print("\n--- 🔍 Model Evaluation ---")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Estimated Accuracy: {accuracy_from_mae:.2f}%")

print("\n--- 🔢 Sample Predictions ---")
for actual, pred in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual:.2f} | Predicted: {pred:.2f}")

# -------------------------------
# 6️⃣ Save Model & Preprocessors
# -------------------------------
with open("car_mileage_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

if label_encoders:
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

print("\n✅ Model, Scaler, and Encoders saved successfully!")
