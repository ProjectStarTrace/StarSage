import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame
features = df[['RSI', 'speed', 'uptime', 'latitude', 'longitude']]
labels = df['signal_strength']  # This column needs to exist in your dataset

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
# Analyze `predictions` to find areas with strongest and weakest signals
