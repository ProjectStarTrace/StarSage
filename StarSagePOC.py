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
        'AnomalyFlag': doc_data.get('AnomalyFlag', False),
    })

df = pd.DataFrame(data)
if df.empty:
    raise ValueError("No data fetched from Firestore.")

# Calculate a placeholder QoSScore for the demonstration; replace with actual logic
df['QoSScore'] = (df['DownloadSpeed'] + df['UploadSpeed']) / 2

# Copy original values for City, Country, and Region before any transformations
df['OriginalCity'] = df['City']
df['OriginalCountry'] = df['Country']
df['OriginalRegion'] = df['Region']

# Apply one-hot encoding
df = pd.get_dummies(df, columns=['City', 'Country', 'Region'])

# Prepare features and labels for modeling
features = df.drop(['ScoutID', 'AnomalyFlag', 'QoSScore', 'OriginalCity', 'OriginalCountry', 'OriginalRegion'], axis=1)
anomaly_labels = df['AnomalyFlag']
qos_labels = df['QoSScore']

# Split the data
X_train, X_test, y_anomaly_train, y_anomaly_test, y_qos_train, y_qos_test = train_test_split(features, anomaly_labels, qos_labels, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and compile TensorFlow models for anomaly detection and QoS scoring
model_anomaly = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(1, activation='sigmoid')])
model_anomaly.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_qos = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(1)])
model_qos.compile(optimizer='adam', loss='mean_squared_error')

# Train the models
model_anomaly.fit(X_train_scaled, y_anomaly_train, epochs=10, batch_size=10, validation_split=0.1)
model_qos.fit(X_train_scaled, y_qos_train, epochs=10, batch_size=10, validation_split=0.1)

# Predict with the models
anomaly_predictions = model_anomaly.predict(X_test_scaled).flatten()
qos_predictions = model_qos.predict(X_test_scaled).flatten()

# Example Firestore update logic with type conversion
ANOMALY_PREDICTION_THRESHOLD = 0.5  # Define the threshold for considering something an anomaly
QOS_THRESHOLD = 25  # Define a threshold for considering QoS too low

# Assuming the predictions and the corresponding rows in X_test are in the same order
# Iterate using enumerate to get the correct index in the prediction arrays
for idx, (index, row) in enumerate(X_test.iterrows()):
    scout_id = df.at[index, 'ScoutID']
    # Access predictions using idx to avoid IndexError
    is_anomalous = bool(anomaly_predictions[idx] > ANOMALY_PREDICTION_THRESHOLD or qos_predictions[idx] < QOS_THRESHOLD)
    doc_ref = db.collection('starsage_predictions').document(scout_id)
    doc_ref.set({
        'City': df.at[index, 'OriginalCity'],
        'Country': df.at[index, 'OriginalCountry'],
        'Region': df.at[index, 'OriginalRegion'],
        'Latitude': float(df.at[index, 'Latitude']),
        'Longitude': float(df.at[index, 'Longitude']),
        'DownloadSpeed': float(df.at[index, 'DownloadSpeed']),
        'UploadSpeed': float(df.at[index, 'UploadSpeed']),
        'IsAnomalous': is_anomalous,
        'QoSScore': float(qos_predictions[idx]),
    })

print("Firestore update with predictions and additional info complete.")
