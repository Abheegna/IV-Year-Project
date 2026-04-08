from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import mysql.connector
import traceback
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = "STRESS_DETECTION"

# MySQL DB Connection
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database="stress_detection"
)
mycursor = mydb.cursor()

# Helper Functions
def execute_query(query, values):
    mycursor.execute(query, values)
    mydb.commit()

def retrieve_query(query, values=None):
    if values:
        mycursor.execute(query, values)
    else:
        mycursor.execute(query)
    return mycursor.fetchall()

# Routes
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            name = request.form['name']
            email = request.form['email'].strip().lower()
            phone = request.form['phone']
            password = request.form['password']
            confirm_password = request.form['confirm_password']

            if password != confirm_password:
                return render_template("register.html", message="❌ Passwords do not match")

            mycursor.execute("SELECT email FROM users WHERE email = %s", (email,))
            if mycursor.fetchone():
                return render_template("register.html", message="⚠️ Email already registered")

            query = "INSERT INTO users (name, email, phone, password) VALUES (%s, %s, %s, %s)"
            execute_query(query, (name, email, phone, password))

            return render_template("login.html", message="✅ Registration successful! Please login.")

        except Exception as e:
            traceback.print_exc()
            return render_template("register.html", message=f"Server Error: {str(e)}")

    return render_template("register.html")

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            email = request.form['email'].strip().lower()
            password = request.form['password']

            mycursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
            user = mycursor.fetchone()

            if user:
                session['user_id'] = user[0]
                session['username'] = user[1]
                return redirect(url_for('home'))
            else:
                return render_template("login.html", message="❌ Invalid email or password")

        except Exception as e:
            traceback.print_exc()
            return render_template("login.html", message=f"Error: {str(e)}")

    return render_template("login.html")

@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html', username=session.get('username'))
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Model Accuracy Page
model_accuracies = {
    "MLPClassifier": "100%",
    "DecisionTreeClassifier": "99%",
    "LogisticRegression": "100%",
    "RandomForestClassifier": "100%"
}

@app.route("/tools/model-training", methods=["GET", "POST"])
def model_training():
    selected_model = None
    accuracy = None

    if request.method == "POST":
        selected_model = request.form.get("model")
        accuracy = model_accuracies.get(selected_model)

    return render_template("model.html", accuracy=accuracy, selected_model=selected_model)

# ======================================
# Model Training on Startup
def train_model():
    df = pd.read_csv('SaYoPillow.csv')


    # # Mapping of short column names to full names
    # column_mapping = {
    #     'sr': 'snoring_rate',
    #     'rr': 'respiration_rate',
    #     't': 'body_temperature',
    #     'lm': 'limb_movement',
    #     'bo': 'blood_oxygen',
    #     'rem': 'eye_movement',
    #     'sr.1': 'sleeping_hours',
    #     'hr': 'heart_rate',
    #     'sl': 'stress_level'
    # }

    print("Columns in CSV:", df.columns.tolist())


    X = df.drop("sl", axis=1)
    y = df["sl"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train_bal, y_train_bal)

    return model, scaler

rf_model, scaler = train_model()

# Prediction Page
@app.route('/tools/prediction')
def prediction_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('prediction.html', username=session.get('username'))

# Prediction API
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
            return jsonify({"error": "All input fields are required"}), 400

        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = rf_model.predict(scaled_features)[0]
        label = "High Stress, Consult a doctor!" if prediction == 1 else "Low Stress, You are fine!"

        return jsonify({"stress_level": label})

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ======================================

if __name__ == "__main__":
    app.run(debug=True)
