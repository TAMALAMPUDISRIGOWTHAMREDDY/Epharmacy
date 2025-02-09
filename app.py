


from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load trained model and label encoder
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load dataset to get symptom list
df = pd.read_csv("model/dataset.csv")
symptoms = list(df.columns[:-1])  # Exclude medicine column

def encode_symptoms(symptom_list):
    encoded_symptoms = np.zeros(len(symptoms))
    for symptom in symptom_list:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            encoded_symptoms[index] = 1
    return encoded_symptoms.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        gender = request.form.get("Gender")
        age = request.form.get("age")
        symptoms_input = request.form.get("symptoms")
        symptoms_list = [sym.strip() for sym in symptoms_input.split(",") if sym.strip()]
        
        if len(symptoms_list) == 0:
            return render_template("index.html", error="Please enter at least one symptom.")
        elif len(symptoms_list) > 3:
            return render_template("index.html", error="Please enter only up to 3 symptoms.")
        
        # Encode symptoms
        encoded_input = encode_symptoms(symptoms_list)
        
        # Predict medicine
        prediction = model.predict(encoded_input)
        recommended_medicine = encoder.inverse_transform(prediction)[0]
        
        return render_template("results.html", gender=gender, age=age, symptoms=", ".join(symptoms_list), medicine=recommended_medicine)
    
    return render_template("index.html")

# Emergency First Aid Pages (Ensure correct paths)
@app.route("/cpr")
def cpr():
    return render_template("first_aid/cpr.html")

@app.route("/electric_shock")
def electric_shock():
    return render_template("first_aid/electric_shocks.html")

@app.route("/fire_accidents")
def fire_accidents():
    return render_template("first_aid/fire_accidents.html")

@app.route("/snake_bites")
def snake_bites():
    return render_template("first_aid/snake_bites.html")

if __name__ == "__main__":
    app.run(debug=True)

