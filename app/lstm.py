import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# ----------------------------
# 1. LIRE LES DONNÉES PARQUET
# ----------------------------

spark = SparkSession.builder.appName("TrainLSTM").getOrCreate()

train_df = spark.read.parquet("hdfs://namenode:9000/data/processed/train").toPandas()
val_df = spark.read.parquet("hdfs://namenode:9000/data/processed/validation").toPandas()
test_df = spark.read.parquet("hdfs://namenode:9000/data/processed/test").toPandas()

# Extraire les valeurs normalisées (déjà entre 0 et 1)
train_scaled = train_df["scaled"].apply(lambda x: x[0]).values
val_scaled = val_df["scaled"].apply(lambda x: x[0]).values
test_scaled = test_df["scaled"].apply(lambda x: x[0]).values

# ------------------------------------
# 2. FONCTION POUR CRÉER LES SÉQUENCES
# ------------------------------------

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled)
X_val, y_val = create_sequences(val_scaled)
X_test, y_test = create_sequences(test_scaled)

# Reshape pour LSTM : (samples, time steps, features)
X_train = X_train.reshape(-1, 60, 1)
X_val = X_val.reshape(-1, 60, 1)
X_test = X_test.reshape(-1, 60, 1)

# ---------------------------
# 3. CONSTRUIRE LE MODÈLE LSTM
# ---------------------------

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# --------------------------
# 4. ENTRAÎNEMENT DU MODÈLE
# --------------------------

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# -------------------------------
# 5. ÉVALUATION SUR VALIDATION
# -------------------------------

y_pred_val = model.predict(X_val)

rmse = math.sqrt(mean_squared_error(y_val, y_pred_val))
mae = mean_absolute_error(y_val, y_pred_val)

print(f"📈 RMSE validation : {rmse:.4f}")
print(f"📉 MAE validation : {mae:.4f}")

# -------------------------------
# 6. VISUALISATION DES PRÉDICTIONS
# -------------------------------

plt.figure(figsize=(15, 6))
plt.plot(y_val, label='Valeurs Réelles')
plt.plot(y_pred_val, label='Prédictions')
plt.title("Prédictions du LSTM sur l'ensemble de validation")
plt.xlabel("Temps")
plt.ylabel("Valeur Normalisée")
plt.legend()
plt.grid(True)
plt.savefig("/app/prediction_plot.png")
