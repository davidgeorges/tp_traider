import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
 
# --- 1. LIRE LES DONNÉES PARQUET ---
 
spark = SparkSession.builder.appName("TrainLSTM").getOrCreate()
 
# Initialiser et ajuster le MinMaxScaler sur la colonne 'Close' du DataFrame d'entraînement original.
# Ce scaler sera utilisé pour inverser la normalisation des prédictions.
original_close_scaler = None
 
try:
    # Charger les DataFrames bruts depuis HDFS. Nous avons besoin de la colonne 'Close' originale
    # pour ajuster le scaler qui servira à l'inverse-transformation des prédictions.
    train_df_raw = spark.read.parquet("hdfs://namenode:9000/data/processed/train").toPandas()
    val_df_raw = spark.read.parquet("hdfs://namenode:9000/data/processed/validation").toPandas()
    test_df_raw = spark.read.parquet("hdfs://namenode:9000/data/processed/test").toPandas()
    print("DataFrames bruts chargés avec succès depuis HDFS.")
 
    if 'Close' in train_df_raw.columns:
        original_close_scaler = MinMaxScaler(feature_range=(0, 1))
        original_close_scaler.fit(train_df_raw[['Close']].values)
        print("MinMaxScaler initialisé et ajusté sur la colonne 'Close' du DataFrame d'entraînement original.")
    else:
        print("Avertissement: La colonne 'Close' n'a pas été trouvée dans train_df_raw. L'inverse_transformation ne sera pas possible sans un scaler ajusté sur les données originales.")
 
    # Extraire les valeurs normalisées (déjà entre 0 et 1) de la colonne 'scaled'
    # Nous supposons que la colonne 'scaled' contient des tableaux à un élément (e.g., [0.5])
    # et nous la remodelons en (n_samples, 1) pour être compatible avec le scaler.
    train_scaled = train_df_raw["scaled"].apply(lambda x: x[0]).values.reshape(-1, 1)
    val_scaled = val_df_raw["scaled"].apply(lambda x: x[0]).values.reshape(-1, 1)
    test_scaled = test_df_raw["scaled"].apply(lambda x: x[0]).values.reshape(-1, 1)
 
except Exception as e:
    print(f"Erreur lors du chargement des DataFrames Spark : {e}")
    print("Veuillez vous assurer que HDFS est accessible et que les fichiers contiennent la colonne 'scaled'.")
    print("Création de DataFrames Pandas d'exemple pour la démonstration.")
   
    # --- Génération de données d'exemple si le chargement HDFS échoue ---
    np.random.seed(42)
    num_samples_train = 200
    num_samples_val = 50
    num_samples_test = 50
 
    def create_dummy_df_with_scaled(num_samples):
        # Générer des valeurs 'Close' brutes pour l'exemple
        raw_close_values = np.random.rand(num_samples) * 100 + 50
        raw_close_values = pd.Series(raw_close_values).rolling(window=5).mean().fillna(method='bfill').values
       
        # Normaliser ces valeurs pour créer la colonne 'scaled'
        temp_scaler = MinMaxScaler()
        scaled_values = temp_scaler.fit_transform(raw_close_values.reshape(-1, 1))
       
        df_dummy = pd.DataFrame({
            'Close': raw_close_values, # Garder la colonne 'Close' originale pour le scaler
            'scaled': list(scaled_values) # Stocker les valeurs normalisées
        })
        return df_dummy
 
    train_df_raw = create_dummy_df_with_scaled(num_samples_train)
    val_df_raw = create_dummy_df_with_scaled(num_samples_val)
    test_df_raw = create_dummy_df_with_scaled(num_samples_test)
    print("DataFrames d'exemple créés.")
 
    # Ajuster le scaler original_close_scaler avec les valeurs 'Close' originales des données d'entraînement d'exemple
    original_close_scaler = MinMaxScaler(feature_range=(0, 1))
    original_close_scaler.fit(train_df_raw[['Close']].values)
    print("MinMaxScaler initialisé et ajusté sur les données 'Close' originales de l'exemple.")
 
    train_scaled = train_df_raw["scaled"].apply(lambda x: x[0]).values.reshape(-1, 1)
    val_scaled = val_df_raw["scaled"].apply(lambda x: x[0]).values.reshape(-1, 1)
    test_scaled = test_df_raw["scaled"].apply(lambda x: x[0]).values.reshape(-1, 1)
 
 
# --- 2. FONCTION POUR CRÉER LES SÉQUENCES ---
sequence_length = 60 # Définir la longueur de la séquence
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, 0]) # Prendre la seule feature (index 0)
        y.append(data[i + seq_length, 0])  # Prendre la seule feature (index 0)
    return np.array(X), np.array(y)
 
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_val, y_val = create_sequences(val_scaled, sequence_length)
X_test, y_test = create_sequences(test_scaled, sequence_length)
 
# Reshape pour LSTM : (samples, time steps, features)
X_train = X_train.reshape(-1, sequence_length, 1)
X_val = X_val.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)
 
# S'assurer que les tableaux y sont en 1D pour une utilisation directe dans model.fit
y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()
 
# --- 3. CONSTRUIRE LE MODÈLE LSTM ---
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
 
model.compile(optimizer='adam', loss='mse')
 
# --- 4. ENTRAÎNEMENT DU MODÈLE ---
print("\nDébut de l'entraînement du modèle LSTM...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)
print("Entraînement terminé.")
 
# --- 5. ÉVALUATION SUR VALIDATION ---
print("\nRéalisation des prédictions sur l'ensemble de validation...")
y_pred_val_scaled = model.predict(X_val)
 
# Inverser la normalisation pour obtenir les prix réels (prédictions de prix futurs)
if original_close_scaler:
    y_pred_val_unscaled = original_close_scaler.inverse_transform(y_pred_val_scaled)
    y_val_unscaled = original_close_scaler.inverse_transform(y_val.reshape(-1, 1))
    print("Prédictions et valeurs réelles inversées à l'échelle des prix originaux.")
else:
    # Si le scaler n'est pas disponible, nous travaillons avec les valeurs normalisées
    print("Impossible d'inverser la normalisation. Les métriques et graphiques seront basés sur les valeurs normalisées (0-1).")
    y_pred_val_unscaled = y_pred_val_scaled # Garder les valeurs scalées
    y_val_unscaled = y_val.reshape(-1, 1) # Garder les valeurs scalées
 
rmse = math.sqrt(mean_squared_error(y_val_unscaled, y_pred_val_unscaled))
mae = mean_absolute_error(y_val_unscaled, y_pred_val_unscaled)
 
print(f"📈 RMSE validation : {rmse:.4f}")
print(f"📉 MAE validation : {mae:.4f}")
 
# --- 6. Calcul d'Indicateurs de Tendance ---
# Les sorties du LSTM incluent des prédictions de prix futurs (y_pred_val_unscaled)
# et des indicateurs de tendance.
 
# Pour l'indicateur de tendance, nous comparons la prédiction avec la dernière valeur connue de la séquence d'entrée.
# X_val[:, -1, 0] donne la dernière valeur normalisée de chaque séquence d'entrée.
last_known_close_val_scaled = X_val[:, -1, 0].reshape(-1, 1)
 
# Inverser la normalisation de la dernière valeur connue si le scaler est disponible
if original_close_scaler:
    last_known_close_val_actual = original_close_scaler.inverse_transform(last_known_close_val_scaled)
else:
    last_known_close_val_actual = last_known_close_val_scaled # Utiliser la valeur normalisée si pas de scaler
 
# Calcul de l'indicateur de tendance (Hausse/Baisse)
tendency_indicator_pred = np.where(y_pred_val_unscaled > last_known_close_val_actual, "Hausse", "Baisse")
tendency_indicator_actual = np.where(y_val_unscaled > last_known_close_val_actual, "Hausse", "Baisse")
 
print("\nExemples d'indicateurs de tendance (Prédit vs Réel) :")
for i in range(min(10, len(tendency_indicator_pred))): # Afficher les 10 premières prédictions de tendance
    print(f"Jour {i+1}: Prédit: {tendency_indicator_pred[i][0]}, Réel: {tendency_indicator_actual[i][0]}")
 
# Test de la capacité à prédire correctement les tendances (Précision de l'indicateur de tendance)
correct_tendency_predictions = np.sum(tendency_indicator_pred == tendency_indicator_actual)
accuracy_tendency = correct_tendency_predictions / len(tendency_indicator_pred)
print(f"\nPrécision de l'indicateur de tendance : {accuracy_tendency:.2%}")
 
 
# --- 7. VISUALISATION DES PRÉDICTIONS ---
plt.figure(figsize=(15, 6))
plt.plot(y_val_unscaled, label='Valeurs Réelles', color='blue', alpha=0.7)
plt.plot(y_pred_val_unscaled, label='Prédictions', color='orange', alpha=0.7)
plt.title("Prédictions du LSTM sur l'ensemble de validation")
plt.xlabel("Temps")
plt.ylabel("Valeur") # Le label de l'axe Y est 'Valeur' car les données sont (si possible) inversées
plt.legend()
plt.grid(True)
plt.savefig("/app/prediction_plot.png")
plt.show()
 
# --- 8. Sauvegarde du Modèle ---
model.save('lstm_model.h5')
print("\nModèle LSTM sauvegardé sous 'lstm_model.h5'")
