import firebase_admin
from firebase_admin import credentials, firestore
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Initialize Firebase Admin
cred = credentials.Certificate("startraceFirebaseJSONAuth.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Fetch data from Firestore
scout_data_ref = db.collection('starscoutData_sim')
scout_data_docs = scout_data_ref.stream()

# Preprocess data
data = []
for data_doc in scout_data_docs:
    doc_data = data_doc.to_dict()
    geopoint = doc_data.get('geolocation', None)
    latitude, longitude = (geopoint.latitude, geopoint.longitude) if geopoint else (None, None)
    data.append({
        'ScoutID': doc_data.get('ScoutID'),
        'City': doc_data.get('City', ''),
        'Country': doc_data.get('Country', ''),
        'Region': doc_data.get('Region', ''),
        'Latitude': latitude,
        'Longitude': longitude,
        'DownloadSpeed': doc_data.get('DownloadSpeed', 0),
        'UploadSpeed': doc_data.get('UploadSpeed', 0),
        # You will need a mechanism to label anomalies in your data. This is a placeholder.
        'AnomalyFlag': doc_data.get('AnomalyFlag', False),
    })

df = pd.DataFrame(data)
if df.empty:
    raise ValueError("No data fetched from Firestore.")

# Convert 'City', 'Country', 'Region' to one-hot encoding, include 'Latitude', 'Longitude'
df = pd.get_dummies(df, columns=['City', 'Country', 'Region'])

# Assuming you're predicting whether a record is anomalous (anomaly detection)
# and predicting a QoS score based on features (QoS prediction)

# Split the DataFrame into features and labels for both tasks
features = df.drop(['ScoutID', 'AnomalyFlag'], axis=1)
anomaly_labels = df['AnomalyFlag']  # For anomaly detection task
# Placeholder QoS scores - you'll replace this with your actual logic or data
qos_labels = np.random.randint(1, 101, size=len(df))  # For QoS prediction task

# Split the data
X_train, X_test, y_anomaly_train, y_anomaly_test, y_qos_train, y_qos_test = train_test_split(
    features, anomaly_labels, qos_labels, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define TensorFlow models for both anomaly detection and QoS scoring
# Anomaly Detection Model
model_anomaly = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_anomaly.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# QoS Prediction Model
model_qos = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Linear activation for regression
])

model_qos.compile(optimizer='adam', loss='mean_squared_error')

# Train the models
model_anomaly.fit(X_train_scaled, y_anomaly_train, epochs=10, batch_size=10, validation_split=0.1)
model_qos.fit(X_train_scaled, y_qos_train, epochs=10, batch_size=10, validation_split=0.1)

# Predict with the models
anomaly_predictions = model_anomaly.predict(X_test_scaled).flatten()
qos_predictions = model_qos.predict(X_test_scaled).flatten()

# Update Firestore with predictions (simplified for illustration)
scout_ids_test = df.iloc[X_test.index]['ScoutID']  # Get ScoutIDs for the test set
for scout_id, anomaly_pred, qos_pred in zip(scout_ids_test, anomaly_predictions, qos_predictions):
    # Simplified logic for determining if a record is predicted as an anomaly
    is_anomalous = anomaly_pred > 0.5
    # Store results in Firestore
    db.collection('starsage_predictions').document(scout_id).set({
        'IsAnomalous': bool(is_anomalous),
        'QoSScore': float(qos_pred)
    })

print("Model training and Firestore update with predictions complete.")
