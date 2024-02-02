import pandas as pd

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate('//content/startrace-81336-firebase-adminsdk-hiz9b-a034d691c7.json')  # Update the path accordingly

# Initialize the app with a None check to prevent reinitialization errors
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Initialize Firestore instance
db = firestore.client()


# Fetching data from Firestore and storing it in a list
data = []
users_ref = db.collection('users')
users_docs = users_ref.stream()

for user_doc in users_docs:
    starcout_data_ref = users_ref.document(user_doc.id).collection('starscoutData')
    starcout_data_docs = starcout_data_ref.stream()

    for data_doc in starcout_data_docs:
        doc_data = data_doc.to_dict()
        if 'geolocation' in doc_data and isinstance(doc_data['geolocation'], dict):  # Ensuring 'geolocation' is a dictionary
            latitude = doc_data['geolocation'].get('latitude')
            longitude = doc_data['geolocation'].get('longitude')
        else:
            latitude, longitude = None, None  # Default values if 'geolocation' is missing or not a dict

        data.append({
            'RSI': doc_data.get('RSI'),
            'deviceID': doc_data.get('deviceID'),  # Assuming inclusion for completeness; might not be used in modeling
            'latitude': latitude,
            'longitude': longitude,
            'percentageUptime': doc_data.get('percentageUptime')
            # 'signal_strength': doc_data.get('signal_strength')  # Assuming you have this or a similar field for labels
        })

# Creating a DataFrame from the list
df = pd.DataFrame(data)

print(df.columns)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Assuming 'RSI' is the label for demonstration
features = df[['latitude', 'longitude', 'percentageUptime']]  # Adjust features as needed
labels = df['RSI']  # Update based on your actual label

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


import tensorflow as tf

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
model.evaluate(X_test_scaled, y_test)

# Predict and analyze
predictions = model.predict(X_test_scaled)
# Further analysis can be done on `predictions` to find areas with strongest and weakest signals
