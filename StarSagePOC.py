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
    data.append({
        'ScoutID': doc_data.get('ScoutID'),
        'DownloadSpeed': doc_data.get('DownloadSpeed', 0),
        'UploadSpeed': doc_data.get('UploadSpeed', 0),
        # Include additional features as necessary
    })

# Convert to DataFrame
df = pd.DataFrame(data)
if df.empty:
    raise ValueError("No data fetched from Firestore.")

# Separate ScoutID and features for modeling
scout_ids = df['ScoutID']
features = df[['DownloadSpeed', 'UploadSpeed']]
target = np.random.randint(1, 101, size=len(df))  # Placeholder target, replace with actual data or model

# Split the data
X_train, X_test, y_train, y_test, scout_ids_train, scout_ids_test = train_test_split(features, target, scout_ids, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid to bound output between 0 and 1
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=10, validation_split=0.1)

# Predict QoS scores
predictions = model.predict(X_test_scaled).flatten()
qos_scores = (predictions * 99) + 1  # Scale to 1-100

# Store predictions in Firestore
for scout_id, qos_score in zip(scout_ids_test, qos_scores):
    db.collection('starsage_predictions').document(scout_id).set({
        'QoSScore': float(qos_score)
    })

print("Model training and Firestore update complete.")
