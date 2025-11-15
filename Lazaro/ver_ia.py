import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os

# --- CONFIGURACIÓN ---
CSV_FILE = "historial_clima.csv"
MODEL_FILE = "modelo_lstm_multivariado.h5"
SCALER_FILE = "scaler_multivariado.pkl"
LOOK_BACK = 24 # El mismo valor que usamos al entrenar

# 1. CARGAR RECURSOS
print("📂 Cargando Cerebro y Datos...")
if not os.path.exists(MODEL_FILE) or not os.path.exists(CSV_FILE):
    print("❌ Faltan archivos. Asegúrate de haber entrenado primero.")
    exit()

model = load_model(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
df = pd.read_csv(CSV_FILE)

# Limpieza rápida
df['timestamp'] = pd.to_datetime(df['timestamp'])
features = ['temperatura', 'pm2_5', 'co', 'no2', 'o3']

# Validar que existan las columnas (si no, rellenar con 0 para no romper la demo)
for col in features:
    if col not in df.columns: df[col] = 0

# 2. PREPARAR LOS DATOS (Igual que en el entrenamiento)
data_raw = df[features].values
data_scaled = scaler.transform(data_raw)

X, y_real = [], []
for i in range(len(data_scaled) - LOOK_BACK - 1):
    X.append(data_scaled[i:(i + LOOK_BACK), :])
    y_real.append(data_raw[i + LOOK_BACK, 0]) # Guardamos el valor REAL de temperatura (sin escalar)

X = np.array(X)
y_real = np.array(y_real)

if len(X) == 0:
    print("⚠️ No hay suficientes datos para graficar. Necesitas más de 25 horas registradas.")
    exit()

# 3. ¡ACCIÓN! LA IA PREDICE TODO EL HISTORIAL
print(f"🧠 La IA está analizando {len(X)} puntos de tiempo...")
pred_scaled = model.predict(X)

# 4. DES-ESCALAR PREDICCIONES
# (Truco para invertir el scaler multivariable)
preds_final = []
for p in pred_scaled:
    dummy = np.zeros((1, 5)) # Fila vacía de 5 huecos
    dummy[0, 0] = p          # Ponemos la predicción en el hueco de Temperatura
    res = scaler.inverse_transform(dummy)[0][0]
    preds_final.append(res)

# 5. GRAFICAR EL PENSAMIENTO
print("📊 Generando Radiografía...")

plt.figure(figsize=(14, 7))
plt.style.use('dark_background') # Estilo Hacker

# Línea Real
plt.plot(y_real, color='#00E5FF', label='Realidad (API)', linewidth=2, alpha=0.8)

# Línea IA
plt.plot(preds_final, color='#FF3D00', label='Lo que "piensa" la IA', linewidth=2, linestyle='--')

plt.title('RADIOGRAFÍA DE LA RED NEURONAL: PREDICCIÓN VS REALIDAD', fontsize=16, color='white')
plt.xlabel('Horas Transcurridas', fontsize=12)
plt.ylabel('Temperatura (°C)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(color='gray', linestyle=':', alpha=0.3)

# Guardar y Mostrar
plt.savefig("radiografia_ia.png")
print("✅ Imagen guardada como 'radiografia_ia.png'")
plt.show()