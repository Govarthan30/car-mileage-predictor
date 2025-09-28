from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
 # Add this after creating your Flask app

app = Flask(__name__)
CORS(app) 

# Load model and encoder
model = pickle.load(open("car_mileage_model.pkl", "rb"))
fuel_encoder = pickle.load(open("fuel_encoder.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    engine = data["engine"]
    weight = data["weight"]
    horsepower = data["horsepower"]
    fuel = fuel_encoder.transform([data["fuel"]])[0]
    speed = data["speed"]

    features = np.array([[engine, weight, horsepower, fuel, speed]])
    prediction = model.predict(features)[0]

    return jsonify({"Predicted Mileage (km/l)": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)
