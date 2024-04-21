import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Assuming the JSON credentials file is at this location
cred = credentials.Certificate("startraceFirebaseJSONAuth.json")

# Initialize the app with a None check to prevent reinitialization errors
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Initialize Firestore instance
db = firestore.client()

# Fetching data from Firestore and storing it in a list
data = []
# Assuming 'starscoutData_simulated' is the correct collection name
scout_data_ref = db.collection('starscoutData_simulated')
scout_data_docs = scout_data_ref.stream()

for data_doc in scout_data_docs:
    doc_data = data_doc.to_dict()
    if 'geolocation' in doc_data and isinstance(doc_data['geolocation'], firestore.GeoPoint):
        latitude = doc_data['geolocation'].latitude
        longitude = doc_data['geolocation'].longitude
    else:
        latitude, longitude = None, None  # Default values if 'geolocation' is missing or not a GeoPoint

    # Update this part to reflect the actual structure of your Firestore documents
    data.append({
        'DeviceID': doc_data.get('DeviceID', None),  # Update based on your Firestore document structure
        'latitude': latitude,
        'longitude': longitude,
        'DownloadSpeed': doc_data.get('DownloadSpeed', None),
        'UploadSpeed': doc_data.get('UploadSpeed', None)
        # Add more fields as needed based on your Firestore document structure
    })

# Creating a DataFrame from the list
df = pd.DataFrame(data)

print("Below are the columns pulled from the Firestore:")
print(df.columns)
print("\n")

# Proceed with your data preprocessing, training, and evaluation as you've outlined.
# The code from here on assumes you have the required fields in your Firestore documents
# and that you adjust the machine learning part as per your specific use case and data.
