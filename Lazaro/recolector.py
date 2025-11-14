# --- recolector.py ---
# Versión 3: Laboratorio Ambiental Completo (Temp, PM2.5, CO, NO2, O3)

import requests
import pandas as pd
from datetime import datetime
import os
import time
import numpy as np # Necesario para manejar columnas nuevas

# --- Configuración ---
API_KEY = "352a68073cee67874db9a5892b8d1d8a" 
MONTERREY_LAT = 25.6866
MONTERREY_LON = -100.3161
CSV_FILE = "historial_clima.csv"
INTERVALO_SEGUNDOS = 300 # 1 hora
# ---------------------

def obtener_datos_completo():
    """
    Obtiene Temperatura y desglose completo de contaminantes.
    """
    datos = {}
    
    # 1. Obtener Temperatura
    try:
        url_w = f"https://api.openweathermap.org/data/2.5/weather?lat={MONTERREY_LAT}&lon={MONTERREY_LON}&appid={API_KEY}&units=metric"
        res_w = requests.get(url_w)
        res_w.raise_for_status()
        w_data = res_w.json()
        datos['temperatura'] = w_data['main']['temp']
    except Exception as e:
        print(f"[{datetime.now()}] Error clima: {e}")
        return None

    # 2. Obtener Contaminantes (La misma API nos da todo esto)
    try:
        url_a = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={MONTERREY_LAT}&lon={MONTERREY_LON}&appid={API_KEY}"
        res_a = requests.get(url_a)
        res_a.raise_for_status()
        a_data = res_a.json()
        
        componentes = a_data['list'][0]['components']
        
        # Extraemos los gases clave
        datos['pm2_5'] = componentes.get('pm2_5', 0)
        datos['co'] = componentes.get('co', 0)   # Monóxido de Carbono (Autos)
        datos['no2'] = componentes.get('no2', 0) # Dióxido de Nitrógeno (Industria)
        datos['o3'] = componentes.get('o3', 0)   # Ozono (Calor)
        
    except Exception as e:
        print(f"[{datetime.now()}] Error AQI: {e}")
        return None
        
    return datos

def actualizar_historial_csv(datos):
    try:
        # Agregamos timestamp al diccionario de datos
        datos['timestamp'] = datetime.now()
        
        # Creamos el DataFrame de una sola fila
        df_nuevo = pd.DataFrame([datos])

        if os.path.exists(CSV_FILE):
            df_viejo = pd.read_csv(CSV_FILE)
            
            # Fusión Inteligente:
            # Si el CSV viejo no tiene las columnas nuevas (co, no2...), pandas las creará
            # y pondrá "NaN" (vacío) en los registros antiguos.
            df_actualizado = pd.concat([df_viejo, df_nuevo], ignore_index=True)
        else:
            df_actualizado = df_nuevo

        # Guardar
        df_actualizado.to_csv(CSV_FILE, index=False)
        
        print(f"[{datetime.now()}] Registro guardado:")
        print(f"   🌡️ Temp: {datos['temperatura']}°C")
        print(f"   🚗 CO: {datos['co']} | 🏭 NO2: {datos['no2']} | ☀️ O3: {datos['o3']}")
        print(f"   Total registros: {len(df_actualizado)}")
    
    except Exception as e:
        print(f"[{datetime.now()}] Error al guardar CSV: {e}")

# --- Bucle Principal ---
if __name__ == "__main__":
    print(f"Iniciando Laboratorio Ambiental Automático...")
    print(f"Monitoreando: Temp, PM2.5, CO, NO2, O3")
    
    while True:
        datos = obtener_datos_completo()
        
        if datos is not None:
            actualizar_historial_csv(datos)
        
        print(f"Durmiendo {INTERVALO_SEGUNDOS} segundos...")
        time.sleep(INTERVALO_SEGUNDOS)