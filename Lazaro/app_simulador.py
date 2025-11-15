import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
from tensorflow.keras.models import load_model
import joblib
import os

# --- 1. CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="Metropolitan AI Sentinel", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .css-card {
        background-color: rgba(38, 39, 48, 0.8);
        border: 1px solid rgba(250, 250, 250, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .recommendation-box {
        background-color: rgba(100, 100, 100, 0.1);
        border-left: 5px solid #888;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    h1, h2, h3, p { color: #ffffff !important; }
    .big-metric { font-size: 28px; font-weight: bold; color: #FFF; }
    .metric-label { font-size: 11px; text-transform: uppercase; color: #aaa; }
    #MainMenu {visibility: hidden;} header {visibility: hidden;}
    
    /* Ajuste para mapa más grande */
    iframe { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DICCIONARIO DE COORDENADAS (ÁREA METROPOLITANA) ---
LOCATIONS = {
    "Monterrey (Centro)": [25.6866, -100.3161],
    "San Pedro": [25.6576, -100.4029],
    "Apodaca": [25.7818, -100.1906],
    "Guadalupe": [25.6774, -100.2597],
    "San Nicolás": [25.7413, -100.2953],
    "Santa Catarina": [25.6768, -100.4627],
    "Escobedo": [25.7969, -100.3260]
}

# --- 3. FUNCIONES DE NEGOCIO ---

def get_health_recommendations(temp, aqi):
    recommendations = []
    status_color = "green" # Default
    
    # Temp
    if temp >= 38:
        recommendations.append("🔥 PELIGRO: Calor Extremo.")
        status_color = "red"
    elif temp < 12:
        recommendations.append("❄️ ALERTA: Bajas Temperaturas.")
        status_color = "blue"
        
    # AQI
    if aqi >= 4:
        recommendations.append("🌫️ AIRE TÓXICO: Use cubrebocas.")
        status_color = "purple"
    elif aqi == 3:
        recommendations.append("😷 Aire Moderado: Precaución grupos sensibles.")
        if status_color not in ["red", "purple"]: status_color = "yellow"
        
    return recommendations, status_color

@st.cache_resource
def load_ai_resources():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "modelo_lstm_clima.h5")
    scaler_path = os.path.join(script_dir, "scaler_clima.pkl")
    if not os.path.exists(model_path): return None, None
    return load_model(model_path), joblib.load(scaler_path)

def get_live_data(lat, lon, api_key):
    try:
        w = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric").json()
        a = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}").json()
        return w['main']['temp'], w['main']['humidity'], a['list'][0]['main']['aqi'], a['list'][0]['components']['pm2_5']
    except:
        return None, None, None, None

def predict_temp(model, scaler, current_temp):
    # Nota: Usamos el mismo modelo para toda el área (simplificación válida para distancias cortas)
    tiempo = np.arange(24)
    input_fake = (10 * np.sin(tiempo * 0.02) + 15) + np.random.normal(0, 1.5, 24)
    input_fake[-1] = current_temp 
    input_scaled = scaler.transform(input_fake.reshape(-1, 1))
    return scaler.inverse_transform(model.predict(input_scaled.reshape(1, 24, 1)))[0][0]

# --- 4. INTERFAZ DE USUARIO ---

with st.sidebar:
    st.title("🛡️ AI Lazaro")
    st.caption("Red de Monitoreo Metropolitano")
    
    # SELECCION DE CIUDAD PARA DETALLE
    selected_city = st.selectbox("📍 Ver Detalles de:", list(LOCATIONS.keys()))
    
    api_key = st.text_input("API Key", value="352a68073cee67874db9a5892b8d1d8a", type="password")
    st.divider()
    st.info("El mapa muestra el estado simultáneo de todos los municipios.")

# --- LÓGICA PRINCIPAL ---
model, scaler = load_ai_resources()

if not model:
    st.error("⚠️ Modelos no encontrados.")
else:
    # Título
    st.markdown(f"## 📡 Monitoreo en Tiempo Real: {selected_city}")
    
    # 1. Obtener datos de la ciudad seleccionada para las tarjetas
    sel_lat, sel_lon = LOCATIONS[selected_city]
    temp, hum, aqi, pm25 = get_live_data(sel_lat, sel_lon, api_key)
    
    if temp is not None:
        pred = predict_temp(model, scaler, temp)
        error = abs(temp - pred)
        
        # Tarjetas de Métricas (Detalle de ciudad seleccionada)
        c1, c2, c3, c4 = st.columns(4)
        
        aqi_labels = ["Bueno", "Aceptable", "Moderado", "Malo", "Peligroso"]
        aqi_text = aqi_labels[aqi-1] if 1 <= aqi <= 5 else "N/A"
        aqi_color = "#4CAF50" if aqi <= 2 else ("#FFC107" if aqi == 3 else "#FF5252")
        
        with c1: st.markdown(f"""<div class="css-card"><div class="metric-label">🌡️ Temp Real</div><div class="big-metric">{temp:.1f}°C</div></div>""", unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class="css-card"><div class="metric-label">🤖 Predicción IA</div><div class="big-metric" style="color:#4FC3F7">{pred:.1f}°C</div></div>""", unsafe_allow_html=True)
        with c3: st.markdown(f"""<div class="css-card"><div class="metric-label">📉 Desviación</div><div class="big-metric">{error:.1f}°C</div></div>""", unsafe_allow_html=True)
        with c4: st.markdown(f"""<div class="css-card"><div class="metric-label">🌫️ Calidad Aire</div><div class="big-metric" style="color:{aqi_color}">{aqi_text}</div></div>""", unsafe_allow_html=True)

        # Recomendaciones rápidas
        recos, status_color = get_health_recommendations(temp, aqi)
        if recos:
            st.markdown(f"""
            <div class="recommendation-box" style="border-left-color: {status_color};">
                <b style="color:{status_color}">⚠️ ALERTA ACTIVA EN {selected_city.upper()}:</b> { " | ".join(recos) }
            </div>
            """, unsafe_allow_html=True)

        # --- 2. GENERACIÓN DEL MAPA METROPOLITANO ---
        st.markdown("### 🗺️ Vista Satelital Metropolitana")
        
        # Inicializar mapa centrado en el promedio de coordenadas
        m = folium.Map(location=[25.72, -100.30], zoom_start=11, tiles="CartoDB dark_matter")
        
        # BUCLE PARA PINTAR TODOS LOS MUNICIPIOS
        for city_name, coords in LOCATIONS.items():
            # Consultar API para cada punto (Rápido)
            c_temp, _, c_aqi, _ = get_live_data(coords[0], coords[1], api_key)
            
            if c_temp is not None:
                # Determinar color del marcador según riesgo
                color = "green" # Todo bien
                radius = 1500
                fill_opacity = 0.4
                
                # Lógica de colores
                if c_temp >= 38: color = "red" # Calor
                elif c_temp < 12: color = "#29B6F6" # Frío
                elif c_aqi >= 4: color = "purple" # Contaminación Severa
                elif c_aqi == 3: color = "orange" # Contaminación Moderada
                
                # Crear Popup con info detallada al dar clic
                popup_html = f"""
                <b>{city_name}</b><br>
                Temp: {c_temp}°C<br>
                AQI: {c_aqi}<br>
                """
                
                # Dibujar círculo
                folium.Circle(
                    location=coords,
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                    popup=folium.Popup(popup_html, max_width=150),
                    tooltip=f"{city_name}: {c_temp}°C | AQI {c_aqi}"
                ).add_to(m)
                
                # Añadir etiqueta de texto permanente
                folium.Marker(
                    location=coords,
                    icon=folium.DivIcon(html=f'<div style="color: white; font-size: 10pt; text-shadow: 2px 2px 4px #000;">{city_name}</div>')
                ).add_to(m)

        # Mostrar mapa con altura aumentada (height=600)
        st_folium(m, width="100%", height=600)