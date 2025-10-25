from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from flask_cors import CORS

# -----------------------------
# 1️⃣ Initialize Flask App
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# 2️⃣ Load Model and Preprocessors
# -----------------------------
try:
    model = pickle.load(open("car_mileage_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    # Load label encoders safely
    if os.path.exists("label_encoders.pkl"):
        label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    else:
        label_encoders = {}
    print("✅ Model and preprocessors loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or preprocessors: {e}")
    raise

# -----------------------------
# 3️⃣ Prediction Endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])

        # -----------------------------
        # Apply label encoding if available
        # -----------------------------
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                # Map unseen categories to first known class
                input_df[col] = input_df[col].apply(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                input_df[col] = encoder.transform(input_df[col])

        # -----------------------------
        # Ensure column order matches training
        # -----------------------------
        expected_columns = model.feature_names_in_  # sklearn >=1.0
        input_df = input_df[expected_columns]

        # -----------------------------
        # Scale features
        # -----------------------------
        input_scaled = scaler.transform(input_df)

        # -----------------------------
        # Make prediction
        # -----------------------------
        prediction = model.predict(input_scaled)[0]

        return jsonify({"Predicted MPG": round(float(prediction), 2)})

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# -----------------------------
# 4️⃣ Run the Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
