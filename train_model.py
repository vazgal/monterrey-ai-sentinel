import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os # Importar el módulo del sistema operativo

# Imprimir el directorio de trabajo actual
print(f"--- ATENCIÓN ---")
print(f"El directorio de trabajo actual es: {os.getcwd()}")
print(f"Los archivos se guardarán AQUÍ.")
print(f"----------------")

# ... (resto de tu código: import numpy, import tensorflow, etc.)
# ----------------------------------------------------------------
# FASE 1: SIMULACIÓN DE DATOS
# ----------------------------------------------------------------
# (En un proyecto real, aquí cargaríamos datos desde la API/BBDD)
print("Generando datos históricos simulados...")

def generar_datos_climaticos(n_puntos):
    # Crea una señal base (senoidal, como las estaciones)
    tiempo = np.arange(n_puntos)
    amplitud = 10
    temperatura = amplitud * np.sin(tiempo * 0.02) + 15 # Base de 15°C
    # Añadir "ruido" (variabilidad diaria)
    ruido = np.random.normal(0, 1.5, n_puntos)
    temperatura = temperatura + ruido
    return temperatura.reshape(-1, 1) # Devolver como columna

# Generamos aprox. 1 año de datos por hora (8760 horas)
datos_simulados = generar_datos_climaticos(8760)

# ----------------------------------------------------------------
# FASE 2: PREPROCESAMIENTO (Ventanas de Series Temporales)
# ----------------------------------------------------------------
print("Preprocesando datos (creando ventanas)...")

# Normalizamos los datos (crucial para redes neuronales)
scaler = MinMaxScaler(feature_range=(0, 1))
datos_escalados = scaler.fit_transform(datos_simulados)

def crear_ventanas(data, look_back=24):
    """
    Crea ventanas de datos.
    look_back: cuántas horas pasadas usamos para predecir.
    Ej: look_back=24. Usamos [hora 1...hora 24] para predecir [hora 25]
    """
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        ventana = data[i:(i + look_back), 0]
        X.append(ventana)
        valor_siguiente = data[i + look_back, 0]
        Y.append(valor_siguiente)
    return np.array(X), np.array(Y)

LOOK_BACK = 24 # Usaremos 24 horas pasadas
X, y = crear_ventanas(datos_escalados, LOOK_BACK)

# Las LSTMs esperan datos 3D: [muestras, timesteps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Datos de entrenamiento: {X_train.shape[0]} secuencias")
print(f"Datos de prueba: {X_test.shape[0]} secuencias")

# ----------------------------------------------------------------
# FASE 3: DEFINICIÓN DEL MODELO (LSTM)
# ----------------------------------------------------------------
print("Construyendo el modelo LSTM...")

model = Sequential()
model.add(LSTM(50, input_shape=(LOOK_BACK, 1))) # 50 neuronas, input_shape=(timesteps, features)
model.add(Dense(1)) # Capa de salida: 1 neurona (la predicción de temperatura)
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# ----------------------------------------------------------------
# FASE 4: ENTRENAMIENTO
# ----------------------------------------------------------------
print("Entrenando el modelo (esto puede tardar)...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=10,          # 10 pasadas sobre los datos
    batch_size=64,      # Procesar en lotes de 64
    validation_data=(X_test, y_test),
    verbose=1
)

# ----------------------------------------------------------------
# FASE 5: EVALUACIÓN Y GUARDADO
# ----------------------------------------------------------------
print("Entrenamiento completo. Evaluando...")

# Hacer una predicción de prueba
prediccion_escalada = model.predict(X_test)
# Revertir la normalización para entender los resultados
prediccion = scaler.inverse_transform(prediccion_escalada)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# (Opcional) Visualizar los resultados
# plt.figure(figsize=(15,6))
# plt.plot(y_test_real[0:200], label="Valor Real")
# plt.plot(prediccion[0:200], label="Predicción IA", linestyle='--')
# plt.legend()
# plt.title("Predicción de IA vs. Valor Real (Datos de Prueba)")
# plt.show()

# Guardar el modelo entrenado y el scaler
model.save("modelo_lstm_clima.h5")
print("Modelo guardado como 'modelo_lstm_clima.h5'")

# También necesitamos guardar el 'scaler' para usarlo en producción
import joblib
joblib.dump(scaler, 'scaler_clima.pkl')
print("Scaler guardado como 'scaler_clima.pkl'")