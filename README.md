# Car Fuel Efficiency Predictor

Predict the **mileage (km/l)** of a car based on engine specifications and driving parameters using a **Machine Learning model** with a Flask backend and HTML/JS frontend.

---

## 🚀 Features

* Predicts car mileage (km/l) using:

  * Engine Size (cc)
  * Weight (kg)
  * Horsepower
  * Fuel Type (Petrol/Diesel)
  * Speed (km/h)
* Machine Learning model trained with **Random Forest Regressor**
* Backend: **Flask (Python)**
* Frontend: **HTML + CSS + JavaScript**
* Model saved as `.pkl` (pickle) for reuse

---

## 🧠 Tech Stack

* **Python 3.13+**
* **scikit-learn** (ML model)
* **Flask** (API backend)
* **HTML / CSS / JS** (frontend UI)
* **Pickle** (for saving model)

---

## 📂 Project Structure

```
Car-Fuel-Efficiency-Predictor/
│── model_training.py       # Train model & save pickle
│── car_mileage_model.pkl   # Trained model
│── fuel_encoder.pkl        # Saved label encoder
│── app.py                  # Flask backend
│── static/                 # CSS, JS files
│── templates/              # HTML frontend
│── README.md               # Project documentation
│── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/Car-Fuel-Efficiency-Predictor.git
   cd Car-Fuel-Efficiency-Predictor
   ```

2. **Create virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training the Model

Run the training script to generate the pickle files:

```bash
python model_training.py
```

This will create:

* `car_mileage_model.pkl` (trained model)
* `fuel_encoder.pkl` (label encoder for fuel type)

---

## ▶️ Running the App

Start the Flask server:

```bash
python app.py
```

By default, the API runs at:

```
http://127.0.0.1:5000
```

---

## 📡 API Usage

**Endpoint:** `/predict`
**Method:** `POST`
**Content-Type:** `application/json`

### Example Request (PowerShell)

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -Body (@{engine=1200; weight=1100; horsepower=85; fuel="petrol"; speed=60} | ConvertTo-Json) -ContentType "application/json"
```

### Example Response

```json
{
  "Predicted Mileage (km/l)": 17.5
}
```

---

## 🌐 Frontend

Open the HTML file in `templates/` folder in your browser.
Fill in the car details → click Predict → shows mileage result.

---

## 📜 Requirements

Add this to `requirements.txt`:

```
Flask
scikit-learn
pandas
numpy
```

---

## 🙌 Target Audience

* **Car Buyers** → choose cost-efficient cars
* **Car Owners** → estimate fuel needs
* **Automobile Engineers** → optimize designs

---

## 🔮 Future Improvements

* Add more parameters (gear type, tire size, aerodynamics)
* Deploy on cloud (Heroku / Render / AWS)
* Build interactive dashboard with charts
