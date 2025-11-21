import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd
import math
import fisica_ambiental # Tu módulo de física

# --- CONFIGURACIÓN PARA SERVIR ARCHIVOS ESTÁTICOS ---
# Streamlit sirve automáticamente archivos desde una carpeta 'static' 
# si se configura, pero un truco más fácil es usar la ruta relativa directa
# soportada por versiones recientes o usar una URL http local.

st.set_page_config(page_title="Digital Twin: Ternium", page_icon="🏭", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #ddd; }
    h1, h2, h3 { color: #FF4B4B; font-family: 'Segoe UI', sans-serif; }
    .metric-container {
        background: rgba(20, 20, 30, 0.8);
        border: 1px solid #333;
        border-left: 5px solid #FF4B4B;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIMULACIÓN DE HUMO ---
def generate_smoke_particles(chimney_locs, wind_s, wind_d, emission_rate):
    particles = []
    rad = math.radians((wind_d + 180) % 360)
    count = int(emission_rate / 8)
    
    for clon, clat in chimney_locs:
        for i in range(count):
            rise = 20 + (np.log(i+1) * 8) 
            z = np.random.normal(rise, 5)
            dist = i * wind_s * 0.00002
            spread = i * 0.000005 * (30/max(wind_s, 1)) 
            dx = dist * math.sin(rad) + np.random.normal(0, spread)
            dy = dist * math.cos(rad) + np.random.normal(0, spread)
            gray = min(200, 50 + (i*2))
            alpha = max(0, 200 - (i*3))
            color = [gray, gray, gray, alpha]
            particles.append([clon + dx, clat + dy, z, color])
    return particles

class FactorySim:
    def __init__(self):
        self.lat = 25.7480; self.lon = -100.2930 
    def calculate_metrics(self, production, wind_s):
        temp = 1100 * (production / 80)
        emission = (production * 12) + (200 / max(wind_s, 0.5))
        risk = "Normal"
        if temp > 1500: risk = "CRÍTICO (FUSIÓN)"
        elif temp > 1350: risk = "ALERTA TÉRMICA"
        return temp, emission, risk

# --- INTERFAZ ---
st.title("🏭 GEMELO DIGITAL FOTOREALISTA")
col_ctrl, col_vis = st.columns([1, 4])
factory = FactorySim()

with col_ctrl:
    st.markdown("### 🎛️ CONTROLES")
    prod = st.slider("Producción (%)", 0, 150, 90)
    wind_s = st.slider("Viento (m/s)", 0, 25, 4)
    wind_d = st.slider("Dirección Viento (°)", 0, 360, 200)
    temp, emission, risk = factory.calculate_metrics(prod, wind_s)
    
    st.markdown("---")
    st.markdown("### 📐 Calibración")
    model_scale = st.slider("Escala Modelo", 1, 1000, 100) 
    model_rot = st.slider("Rotación", 0, 360, 0)
    
    st.markdown("---")
    r_col = "red" if "CRÍTICO" in risk else ("orange" if "ALERTA" in risk else "#00E676")
    st.markdown(f"ESTATUS: <b style='color:{r_col}'>{risk}</b>", unsafe_allow_html=True)

with col_vis:
    chimney_locs = [[factory.lon, factory.lat]] 
    smoke_data = generate_smoke_particles(chimney_locs, wind_s, wind_d, emission)
    df_smoke = pd.DataFrame(smoke_data, columns=["lon", "lat", "z", "color"])
    
    layers = []
    
    # --- CAMBIO IMPORTANTE: URL LOCAL ---
    # Para que esto funcione, el archivo debe estar en una carpeta 'static'
    # y accedemos a él como si fuera una web:
    # Nota: En Streamlit local, a veces es necesario servir la carpeta.
    # Si esto falla, usaremos la URL pública de GitHub como respaldo.
    
    # OPCIÓN A: URL Pública (La más segura para evitar errores locales)
    # Si ya subiste el archivo a GitHub, usa el enlace "Raw" aquí.
    # Si no, usa esta URL de prueba de un edificio genérico para verificar que funcione:
    MODEL_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/CesiumMilktruck/glTF-Binary/CesiumMilktruck.glb"
    
    # OPCIÓN B: Archivo Local (Requiere configuración de servidor estático de Streamlit)
    # Por defecto Streamlit bloquea archivos locales. 
    # La mejor forma local sin configurar servidor es usar la URL pública de GitHub raw.
    
    layers.append(pdk.Layer(
        "ScenegraphLayer",
        data=[{"position": [factory.lon, factory.lat, 0], "size": model_scale, "angle": model_rot}],
        scenegraph=MODEL_URL, # <--- Carga desde URL
        get_position="position",
        get_orientation=[0, "angle", 90],
        get_scale=[1, 1, 1],
        size_scale="size",
        _lighting="pbr",
        pickable=True,
    ))
    
    layers.append(pdk.Layer(
        "PointCloudLayer",
        data=df_smoke,
        get_position=["lon", "lat", "z"],
        get_color="color",
        point_size=6,
        opacity=0.8
    ))

    view_state = pdk.ViewState(latitude=factory.lat, longitude=factory.lon, zoom=18, pitch=60, bearing=-45)
    
    r = pdk.Deck(layers=layers, initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10", tooltip={"text": "Planta Industrial"})
    st.pydeck_chart(r)