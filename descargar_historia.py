import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# --- CONFIGURACIÓN ---
# Coordenadas Monterrey
LAT = 25.6866
LON = -100.3161
START_DATE = "2023-01-01" # Desde hace más de un año
END_DATE = datetime.now().strftime("%Y-%m-%d") # Hasta hoy
FILENAME = "historial_clima.csv"

def descargar_datos():
    print(f"📡 Conectando con Open-Meteo Archives...")
    print(f"📅 Descargando datos desde {START_DATE} hasta {END_DATE}...")

    # 1. URL PARA CLIMA (Temperatura)
    url_clima = "https://archive-api.open-meteo.com/v1/archive"
    params_clima = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m",
        "timezone": "America/Monterrey"
    }

    # 2. URL PARA CALIDAD DEL AIRE (Gases)
    # Nota: Usamos el endpoint de calidad del aire que tiene historial
    url_aqi = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params_aqi = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "pm2_5,carbon_monoxide,nitrogen_dioxide,ozone",
        "timezone": "America/Monterrey"
    }

    try:
        # --- DESCARGAR CLIMA ---
        print("   ⬇️ Descargando Temperatura...")
        r_clima = requests.get(url_clima, params=params_clima)
        r_clima.raise_for_status()
        data_clima = r_clima.json()
        
        df_clima = pd.DataFrame({
            'timestamp': data_clima['hourly']['time'],
            'temperatura': data_clima['hourly']['temperature_2m']
        })

        # --- DESCARGAR AQI ---
        print("   ⬇️ Descargando Gases (PM2.5, CO, NO2, O3)...")
        r_aqi = requests.get(url_aqi, params=params_aqi)
        r_aqi.raise_for_status()
        data_aqi = r_aqi.json()
        
        df_aqi = pd.DataFrame({
            'timestamp': data_aqi['hourly']['time'],
            'pm2_5': data_aqi['hourly']['pm2_5'],
            'co': data_aqi['hourly']['carbon_monoxide'],
            'no2': data_aqi['hourly']['nitrogen_dioxide'],
            'o3': data_aqi['hourly']['ozone']
        })

        # --- FUSIONAR Y LIMPIAR ---
        print("   🔄 Fusionando bases de datos...")
        
        # Convertir timestamp a formato fecha real para asegurar la fusión correcta
        df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
        df_aqi['timestamp'] = pd.to_datetime(df_aqi['timestamp'])
        
        # Unir (Merge) por la columna tiempo
        df_final = pd.merge(df_clima, df_aqi, on='timestamp', how='inner')
        
        # Eliminar filas con datos vacíos (NaN) que a veces vienen de la API
        df_final = df_final.dropna()

        # --- GUARDAR ---
        # Si ya existe un archivo, preguntamos si queremos sobrescribir
        if os.path.exists(FILENAME):
            print(f"⚠️ ¡CUIDADO! El archivo {FILENAME} ya existe.")
            user_input = input("¿Quieres SOBRESCRIBIRLO con los datos históricos masivos? (s/n): ")
            if user_input.lower() != 's':
                print("Cancelado. No se guardó nada.")
                return

        df_final.to_csv(FILENAME, index=False)
        
        print("-" * 40)
        print(f"✅ ¡ÉXITO! Base de datos generada.")
        print(f"📂 Archivo: {FILENAME}")
        print(f"📊 Total de registros: {len(df_final)}")
        print(f"📅 Periodo: {df_final['timestamp'].min()} a {df_final['timestamp'].max()}")
        print("-" * 40)
        print("👉 AHORA: Ejecuta 'entrenar_avanzado.py' para crear la IA Maestra.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    descargar_datos()