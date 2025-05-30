import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

from pyspark.sql import SparkSession
 
spark = SparkSession.builder.appName("LireTestParquet").getOrCreate()
 
# Lire le fichier Parquet dans HDFS

df = spark.read.parquet("hdfs://namenode:9000/data/processed/train")

# Convertir le DataFrame Spark en DataFrame Pandas
df = df.toPandas()

# Vérifie les colonnes disponibles
print(df.columns)

# On suppose que la colonne pour entraîner le modèle s'appelle 'Close'
# Sinon, adapte ici avec la colonne correcte
data = df[['Close']].values

# Modèle LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Entraînement
from sklearn.model_selection import train_test_split

# Diviser les données (par exemple, 80% entraînement, 20% validation)
# Pour les séries temporelles, la division doit être chronologique !
# Ne pas utiliser train_test_split directement si l'ordre est important.
# On prend les 80% premières données pour l'entraînement et les 20% dernières pour la validation.

train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Faire des prédictions sur l'ensemble de validation
y_pred_val_scaled = model.predict(X_val)

# Inverser la normalisation pour obtenir les valeurs réelles prédites
# N'oubliez pas que scaler a été entraîné sur l'ensemble complet (scaled_data)
# Si scaler est entraîné sur une seule colonne 'Close', il peut être unidimensionnel.
# Assurez-vous que y_pred_val_scaled a la bonne forme (n_samples, 1) pour l'inversion.
y_pred_val = scaler.inverse_transform(y_pred_val_scaled)
y_val_actual = scaler.inverse_transform(y_val)


# Calculer les métriques
rmse = math.sqrt(mean_squared_error(y_val_actual, y_pred_val))
mae = mean_absolute_error(y_val_actual, y_pred_val)

print(f"RMSE sur l'ensemble de validation : {rmse:.2f}")
print(f"MAE sur l'ensemble de validation : {mae:.2f}")

# Pour MAPE, faites attention à la division par zéro si y_val_actual peut contenir 0
# mape = np.mean(np.abs((y_val_actual - y_pred_val) / y_val_actual)) * 100
import matplotlib.pyplot as plt

# Tracer les valeurs réelles et les prédictions
plt.figure(figsize=(15, 6))
plt.plot(y_val_actual, label='Valeurs Réelles')
plt.plot(y_pred_val, label='Prédictions du Modèle')
plt.title('Prédictions du Modèle LSTM vs Valeurs Réelles')
plt.xlabel('Temps')
plt.ylabel('Valeur')
plt.legend()
plt.show()