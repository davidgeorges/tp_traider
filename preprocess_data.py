import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Charger les données
df = pd.read_csv('data/2008_Globla_Markets_Data.csv')

# Convertir la colonne Date en datetime
df['Date'] = pd.to_datetime(df['Date'])

# Créer des séquences pour le LSTM (utiliser les 60 derniers jours pour prédire le jour suivant)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Sélectionner les colonnes pertinentes pour l'analyse
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Normaliser les données
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Créer des séquences de 60 jours
seq_length = 60
X, y = create_sequences(df_normalized.values, seq_length)
print(X[0:60])
print("mi")
print(y)

# Diviser les données en ensembles d'entraînement (70%), validation (15%) et test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, shuffle=False)

# Sauvegarder les ensembles de données prétraités
np.save('data/X_train.npy', X_train)
np.save('data/X_val.npy', X_val)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_val.npy', y_val)
np.save('data/y_test.npy', y_test)

# Sauvegarder le scaler pour une utilisation ultérieure
import joblib
joblib.dump(scaler, 'data/scaler.joblib')

print(f"Forme des données d'entraînement : {X_train.shape}")
print(f"Forme des données de validation : {X_val.shape}")
print(f"Forme des données de test : {X_test.shape}")