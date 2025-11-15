import pandas as pd
from datetime import datetime
import requests
import numpy as np
import joblib
import os
import folium  # Importar para el mapa
import webbrowser # Importar para abrir el mapa
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------
# CLASE 1: EL DETECTOR (VERSIÓN CON RECOLECCIÓN DE DATOS)
# ----------------------------------------------------------------

class DetectorIA_ML:
    """
    Esta IA ahora también guarda la temperatura real en un CSV
    para construir un dataset histórico.
    """
    
    def __init__(self, api_key, lat, lon, model_path, scaler_path, csv_path):
        # Configuración de API
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        if self.api_key == "352a68073cee67874db9a5892b8d1d8a" or not self.api_key:
            raise ValueError("API Key no configurada en DetectorIA_ML.")
        
        # Ruta del archivo CSV para el historial
        self.historial_csv_path = csv_path
        
        # Cargar el modelo de IA y el scaler
        try:
            print(f"Cargando modelo de IA desde: {model_path}")
            self.model = load_model(model_path)
            print(f"Cargando scaler desde: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            print("Modelo y scaler cargados exitosamente.")
        except Exception as e:
            raise IOError(f"Error al cargar el modelo o scaler: {e}")
            
        self.LOOK_BACK = 24 # Debe ser el mismo valor usado en el entrenamiento
        self.ERROR_THRESHOLD = 2.5 # Umbral de anomalía (ej. 2.5°C de error)

    def _actualizar_historial_real(self, temp_real):
        """
        NUEVA FUNCIÓN:
        Guarda la temperatura real actual en nuestro archivo CSV.
        """
        try:
            # 1. Definir el nuevo registro
            data_nueva = {"timestamp": [datetime.now()], "temperatura": [temp_real]}
            df_nuevo = pd.DataFrame(data_nueva)

            # 2. Comprobar si el archivo ya existe
            if os.path.exists(self.historial_csv_path):
                # 3a. Si existe, cargar los datos antiguos
                df_viejo = pd.read_csv(self.historial_csv_path)
                # 3b. Añadir la nueva fila
                df_actualizado = pd.concat([df_viejo, df_nuevo], ignore_index=True)
            else:
                # 3c. Si no existe, este es el primer registro
                df_actualizado = df_nuevo

            # 4. Guardar el archivo CSV actualizado
            df_actualizado.to_csv(self.historial_csv_path, index=False)
            print(f"Historial real guardado. Total de registros: {len(df_actualizado)}")
        
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo guardar el historial CSV. Error: {e}")

    def _get_real_and_simulated_data(self):
        """
        FUNCIÓN MODIFICADA:
        Ahora también llama a _actualizar_historial_real
        """
        
        # 1. OBTENER VALOR REAL (Llamada real a la API)
        print("Obteniendo temperatura real actual desde la API...")
        current_temp_real = None
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={self.lat}&lon={self.lon}&appid={self.api_key}&units=metric"
            response = requests.get(url)
            response.raise_for_status()
            weather_data = response.json()
            current_temp_real = weather_data['main']['temp']
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener clima real: {e}")
            return None, None
        except KeyError:
            print("Error: Respuesta de API inesperada.")
            return None, None

        # 2. (¡NUEVO!) GUARDAR EL DATO REAL
        if current_temp_real is not None:
            self._actualizar_historial_real(current_temp_real)

        # 3. SIMULAR HISTORIAL (Input para la IA)
        # (Aún usamos esto para la predicción, hasta que el CSV tenga datos)
        print("Generando historial de 24h simulado...")
        tiempo = np.arange(self.LOOK_BACK)
        temp_base = 10 * np.sin(tiempo * 0.02) + 15
        ruido = np.random.normal(0, 1.5, self.LOOK_BACK)
        historical_temps = (temp_base + ruido)
        
        return historical_temps, current_temp_real

    def check_anomaly(self):
        """
        (Esta función no cambia, pero la incluyo para que la clase esté completa)
        Ejecuta la predicción y la compara con la realidad.
        """
        
        # 1. Obtener datos
        last_24h_temps, current_temp_real = self._get_real_and_simulated_data()
        
        # 2. Chequeo de error de API
        if current_temp_real is None or last_24h_temps is None:
            return {"error": "No se pudieron obtener datos para la IA."}

        # 3. Preprocesar los datos
        input_data = np.array(last_24h_temps).reshape(-1, 1)
        input_data_scaled = self.scaler.transform(input_data)
        
        # Formato 3D para la LSTM
        input_data_lstm = np.reshape(input_data_scaled, (1, self.LOOK_BACK, 1))
        
        # 4. Hacer la predicción (IA)
        predicted_temp_scaled = self.model.predict(input_data_lstm)
        
        # 5. "Des-escalar" la predicción
        predicted_temp = self.scaler.inverse_transform(predicted_temp_scaled)[0][0]
        
        # 6. Calcular el error (Anomalía)
        error = abs(current_temp_real - predicted_temp)
        
        print(f"Predicción de IA (Próx. hora): {predicted_temp:.2f}°C")
        print(f"Valor Real (API): {current_temp_real:.2f}°C")
        print(f"Error de predicción: {error:.2f}°C")

        report = {
            "is_anomaly": error > self.ERROR_THRESHOLD,
            "error": error,
            "threshold": self.ERROR_THRESHOLD,
            "current_temp": current_temp_real,
            "predicted_temp": predicted_temp,
            "message": ""
        }
        
        if report["is_anomaly"]:
            report["message"] = f"¡Anomalía detectada! El clima se desvía en {error:.2f}°C de la predicción."
        else:
            report["message"] = "Comportamiento climático normal (dentro de las predicciones)."
            
        return report

# ----------------------------------------------------------------
# CLASE 2: LA APLICACIÓN DE SIMULACIÓN (CON LÓGICA DE ESCALA)
# ----------------------------------------------------------------
class SimuladorApp:
    def __init__(self, detector: DetectorIA_ML, center_coords):
        self.detector = detector
        self.center_coords = center_coords
        print("SimuladorApp (Versión GIS v2 - Lógica de Escala) inicializado.")
        
        # --- BASE DE DATOS GIS (Simulada) ---
        # Definimos nuestras zonas de riesgo por adelantado
        self.ZONA_RIESGO_A = {
          "type": "Feature",
          "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-100.3161, 25.6866], [-100.3261, 25.6866],
                [-100.3261, 25.6766], [-100.3161, 25.6766],
                [-100.3161, 25.6866]
            ]]
          },
          "properties": { "nombre": "Zona A (Centro)", "riesgo": "Moderado" }
        }
        
        self.ZONA_RIESGO_B = {
          "type": "Feature",
          "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-100.30, 25.70], [-100.31, 25.70],
                [-100.31, 25.69], [-100.30, 25.69],
                [-100.30, 25.70]
            ]]
          },
          "properties": { "nombre": "Zona B (Norte)", "riesgo": "Alto" }
        }
        # -------------------------------------

    def _get_risk_zones(self, reporte):
        """
        (SIMULACIÓN GIS - INTELIGENTE)
        Esta es la nueva lógica.
        Decide QUÉ zonas activar basado en la magnitud del error de la IA.
        """
        error_magnitud = reporte['error']
        zonas_activadas = []
        
        print(f"Lógica de simulación: Magnitud de error es {error_magnitud:.2f}°C")
        
        # Lógica de escala
        if 2.5 <= error_magnitud < 4.0:
            print("Activando Nivel 1: Zona de Riesgo A")
            zonas_activadas.append(self.ZONA_RIESGO_A)
            
        elif error_magnitud >= 4.0:
            print("Activando Nivel 2: Zonas de Riesgo A y B")
            zonas_activadas.append(self.ZONA_RIESGO_A)
            zonas_activadas.append(self.ZONA_RIESGO_B)
            
        # Creamos un FeatureCollection de GeoJSON con las zonas activadas
        return {
          "type": "FeatureCollection",
          "features": zonas_activadas
        }

    def _generar_mapa_riesgo(self, reporte, zonas_de_riesgo):
        """
        (SIMULACIÓN VISUAL)
        Genera el mapa de riesgo con las zonas que la lógica decidió.
        """
        print(f"Generando mapa con {len(zonas_de_riesgo['features'])} zona(s) de riesgo...")
        
        # 1. Crear el mapa centrado en Monterrey
        m = folium.Map(location=self.center_coords, zoom_start=12) # Zoom más lejano
        
        # 2. Función de estilo dinámico
        # El mapa coloreará el polígono basado en su propiedad 'riesgo'
        def estilo_gis(feature):
            riesgo = feature['properties'].get('riesgo', 'Moderado')
            color = 'orange' # Color por defecto
            if riesgo == 'Moderado':
                color = 'orange'
            elif riesgo == 'Alto':
                color = 'crimson'
            
            return {
                'fillColor': color,
                'color': color,
                'weight': 2,
                'fillOpacity': 0.6,
            }

        # 3. (Simulación de Impacto)
        # Añadimos los polígonos GeoJSON al mapa
        folium.GeoJson(
            zonas_de_riesgo,
            style_function=estilo_gis,
            tooltip=folium.GeoJsonTooltip(
                fields=['nombre', 'riesgo'],
                aliases=['Zona:', 'Nivel:'],
                sticky=True
            )
        ).add_to(m)

        # 4. Guardar el mapa en un archivo HTML
        map_filename = "simulacion_riesgo_escalado.html" # Nuevo nombre
        m.save(map_filename)
        
        print(f"Simulación completada. Mapa de riesgo guardado en: {map_filename}")
        
        # 5. Abrir el mapa en el navegador
        try:
            webbrowser.open(f"file://{os.path.realpath(map_filename)}")
        except Exception as e:
            print(f"No se pudo abrir el navegador automáticamente. Abre el archivo manualmente. {e}")

    def run_check(self):
        """
        Método principal de la aplicación.
        (Actualizado para pasar el reporte a la lógica GIS)
        """
        print("\n--- Nuevo Chequeo de Simulación (IA-LSTM) ---")
        print("Contactando al DetectorIA_ML...")
        reporte = self.detector.check_anomaly()
        
        if "is_anomaly" not in reporte:
            print(f"Error del detector: {reporte.get('error', 'Error desconocido')}")
            return

        print(reporte["message"])
        
        if reporte["is_anomaly"]:
            print("¡Anomalía detectada! Iniciando protocolo de simulación GIS...")
            
            # 1. Decide qué zonas activar
            zonas_a_simular = self._get_risk_zones(reporte)
            
            # 2. Genera el mapa si hay zonas activas
            if zonas_a_simular["features"]:
                self._generar_mapa_riesgo(reporte, zonas_a_simular)
            else:
                print("La anomalía era muy pequeña, no se activaron zonas de riesgo.")
                
        else:
            print("Condiciones normales. No se requiere simulación.")
# ----------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ----------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Definir la ruta al script y a los modelos
    #    (¡ACTUALIZADO! Ahora busca los modelos en la misma carpeta que el script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_FILE = os.path.join(script_dir, "modelo_lstm_clima.h5")
    SCALER_FILE = os.path.join(script_dir, "scaler_clima.pkl")
    
    # 2. Configuración (¡TU API KEY YA ESTÁ PUESTA!)
    API_KEY = "352a68073cee67874db9a5892b8d1d8a" # ¡TU API KEY YA ESTÁ AQUÍ!
    MONTERREY_COORDS = [25.6866, -100.3161]
    
    # --- LÍNEA DE DIAGNÓSTICO ---
    print(f"DEBUG: Iniciando script. La API Key configurada es: '{API_KEY}'")
    # -----------------------------
    
    try:
        # 3. Crear la instancia del Detector ML
        mi_detector_ia_ml = DetectorIA_ML(
            API_KEY, 
            MONTERREY_COORDS[0], 
            MONTERREY_COORDS[1],
            MODEL_FILE,
            SCALER_FILE
        )
        
        # 4. Crear la instancia del Simulador
        mi_simulador = SimuladorApp(detector=mi_detector_ia_ml, center_coords=MONTERREY_COORDS)
        
        # 5. Ejecutar el chequeo
        mi_simulador.run_check()
        
    except ValueError as e:
        print(f"Error de configuración: {e}")
    except IOError as e:
        print(f"Error de archivos: {e}")
        print(f"Asegúrate de tener los archivos .h5 y .pkl en: {script_dir}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")