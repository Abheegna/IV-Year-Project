from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import mysql.connector
import traceback
import warnings
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = "STRESS_DETECTION"

# ================= DATABASE =================
mydb = None
mycursor = None

try:
    mydb = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "stress_detection"),
        port=3306
    )
    mycursor = mydb.cursor()
    print("✅ Database connected")
except Exception as e:
    print("⚠️ Database not connected:", e)

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ================= REGISTER =================
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            if not mycursor:
                return render_template("register.html", message="⚠️ DB not connected")

            name = request.form['name']
            email = request.form['email'].strip().lower()
            phone = request.form['phone']
            password = request.form['password']
            confirm_password = request.form['confirm_password']

            if password != confirm_password:
                return render_template("register.html", message="❌ Password mismatch")

            # Check existing user
            mycursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            if mycursor.fetchone():
                return render_template("register.html", message="⚠️ Email already exists")

            # Insert user
            mycursor.execute(
                "INSERT INTO users (name, email, phone, password) VALUES (%s, %s, %s, %s)",
                (name, email, phone, password)
            )
            mydb.commit()

            return render_template("login.html", message="✅ Registered successfully")

        except Exception as e:
            traceback.print_exc()
            return render_template("register.html", message=str(e))

    return render_template("register.html")

# ================= LOGIN =================
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            if not mycursor:
                return render_template("login.html", message="⚠️ DB not connected")

            email = request.form['email'].strip().lower()
            password = request.form['password']

            mycursor.execute(
                "SELECT id, name FROM users WHERE email=%s AND password=%s",
                (email, password)
            )
            user = mycursor.fetchone()

            print("LOGIN DEBUG:", user)

            if user:
                session['user_id'] = user[0]
                session['username'] = user[1]
                return redirect(url_for('home'))
            else:
                return render_template("login.html", message="❌ Invalid email or password")

        except Exception as e:
            traceback.print_exc()
            return render_template("login.html", message=str(e))

    return render_template("login.html")

# ================= HOME =================
@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html', username=session.get('username'))
    return redirect(url_for('login'))

# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ================= MODEL =================
def train_model():
    df = pd.read_csv('SaYoPillow.csv')

    X = df.drop("sl", axis=1)
    y = df["sl"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier()
    model.fit(X_train_bal, y_train_bal)

    return model, scaler

rf_model, scaler = train_model()

# ================= PREDICTION =================
@app.route('/tools/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [
            data.get("snoring_rate"),
            data.get("respiration_rate"),
            data.get("body_temperature"),
            data.get("limb_movement"),
            data.get("blood_oxygen"),
            data.get("eye_movement"),
            data.get("sleeping_hours"),
            data.get("heart_rate")
        ]

        if None in features:
            return jsonify({"error": "All fields required"}), 400

        features = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features)
        pred = rf_model.predict(scaled)[0]

        result = "High Stress" if pred == 1 else "Low Stress"

        return jsonify({"stress_level": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
