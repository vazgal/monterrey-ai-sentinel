import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# --- CONFIGURACIÓN ---
CSV_FILE = "historial_clima.csv"
MODEL_FILE = "modelo_lstm_multivariado.h5" # Nombre nuevo para no borrar el viejo aún
SCALER_FILE = "scaler_multivariado.pkl"
LOOK_BACK = 24 # Miramos 24 horas hacia atrás
FEATURES = ['temperatura', 'pm2_5', 'co', 'no2', 'o3'] # Las 5 variables que usaremos

# 1. CARGAR DATOS
print("📂 Cargando historial climático completo...")
if not os.path.exists(CSV_FILE):
    print("❌ Error: No existe historial_clima.csv")
    exit()

df = pd.read_csv(CSV_FILE)

# Verificar que existan todas las columnas
missing_cols = [col for col in FEATURES if col not in df.columns]
if missing_cols:
    print(f"❌ Faltan columnas en tu CSV para el entrenamiento avanzado: {missing_cols}")
    print("Deja correr el 'recolector.py' nuevo por más tiempo.")
    exit()

# Limpieza básica
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.dropna() # Eliminar filas vacías

# Necesitamos suficientes datos
if len(df) < (LOOK_BACK + 20):
    print(f"⚠️ Datos insuficientes. Tienes {len(df)} filas, necesitas al menos {LOOK_BACK + 20}.")
    print("Sugerencia: Modifica 'recolector.py' para guardar cada 5 min (300s) para pruebas rápidas.")
    exit()

# 2. PREPROCESAMIENTO (ESCALADO)
print("⚖️ Normalizando datos (0 a 1)...")
dataset = df[FEATURES].values # Tomamos solo las columnas numéricas
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# 3. CREAR VENTANAS DE TIEMPO (X e y)
# X = Lo que pasó en las últimas 24 horas (Temp, CO, NO2...)
# y = La Temperatura de la siguiente hora (Nuestro objetivo a predecir)
def create_dataset(dataset, look_back=24):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        # Input: Ventana de 24 horas con TODAS las variables
        a = dataset[i:(i + look_back), :] 
        X.append(a)
        # Output: Solo predecimos Temperatura (columna 0)
        # Si quisieras predecir PM2.5, cambiarías el 0 por el 1
        y.append(dataset[i + look_back, 0]) 
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, LOOK_BACK)

# Formato para LSTM: [samples, time steps, features]
# features ahora será 5 (antes era 1)
print(f"📊 Dimensiones de entrenamiento: {X.shape}")

# 4. CONSTRUIR MODELO AVANZADO
print("🧠 Construyendo Red Neuronal Multivariable...")
model = Sequential()

# Capa 1: LSTM
# input_shape=(24 pasos de tiempo, 5 variables)
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2)) # Evita sobreajuste

# Capa 2: LSTM
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))

# Capa Salida
model.add(Dense(1)) # 1 neurona porque predecimos 1 valor (Temperatura)

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. ENTRENAR
print("🚀 Iniciando entrenamiento...")
model.fit(X, y, epochs=20, batch_size=16, verbose=1)

# 6. GUARDAR
model.save(MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

print("------------------------------------------------")
print(f"✅ ¡MODELO MULTIVARIABLE CREADO EXITOSAMENTE!")
print(f"   Cerebro guardado en: {MODEL_FILE}")
print(f"   Traductor guardado en: {SCALER_FILE}")
print("------------------------------------------------")
print("💡 Siguiente paso: Modificar 'app_visual.py' para usar este nuevo cerebro.")