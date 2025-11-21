import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from tensorflow.keras.models import load_model
import joblib
import os
import altair as alt
import warnings
from fpdf import FPDF

# --- IMPORTAR NUESTRO MOTOR CIENTÍFICO ---
import fisica_ambiental # <--- ¡ASEGÚRATE DE HABER CREADO EL ARCHIVO ANTES!

# --- 0. CONFIGURACIÓN ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

st.set_page_config(page_title="GlobalAir", page_icon="🌍", layout="wide")

# --- 1. ESTILOS (LIQUID GLASS) ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(-45deg, #000000, #1a1a1a, #0d0d0d, #2b2b2b); background-size: 400% 400%; animation: gradient 15s ease infinite; }
    @keyframes gradient { 0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;} }
    .css-card { background: rgba(255, 255, 255, 0.05); border-radius: 20px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); padding: 25px; }
    h1, h2, h3 { color: #ffffff !important; font-family: 'Helvetica Neue', sans-serif; }
    .big-metric { font-size: 38px; font-weight: 800; color: #FFF; text-shadow: 0 0 10px rgba(255,255,255,0.3); }
    .metric-label { font-size: 12px; text-transform: uppercase; color: rgba(255,255,255,0.7); }
    iframe { border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. UBICACIONES ---
LOCATIONS = {
    "Monterrey (Centro)": [25.6866, -100.3161], "San Pedro": [25.6576, -100.4029],
    "Apodaca (Aeropuerto)": [25.7785, -100.1864], "Guadalupe": [25.6774, -100.2597],
    "San Nicolás": [25.7413, -100.2953], "Santa Catarina": [25.6768, -100.4627],
    "Escobedo": [25.7969, -100.3260], "García": [25.8119, -100.5928],
    "Juárez": [25.6493, -100.0951], "Santiago": [25.4317, -100.1533],
    "Cadereyta": [25.5879, -99.9976], "Cemex (Planta Mty)": [25.7080, -100.2960],
    "Ternium (Guerrero)": [25.7480, -100.2930], "Kia (Pesquería)": [25.7735, -99.9565]
}

# --- 3. FUNCIONES DE NEGOCIO ---
def deg_to_cardinal(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((deg + 11.25)/22.5)
    return dirs[ix % 16]

def get_status_color(temp, aqi):
    if aqi >= 4: return "#D500F9"  
    if aqi == 3: return "#FF1744"
    if aqi == 2: return "#FF9100"
    if temp >= 38: return "#FF3D00"
    if temp < 12:  return "#2979FF"
    return "#00E676"

def get_protocols(temp, aqi):
    protocols = []
    status = "normal"
    if temp >= 38: protocols.append("🔥 GOLPE DE CALOR: Hidratación obligatoria."); status = "danger"
    elif temp < 12: protocols.append("❄️ BAJAS TEMPERATURAS: Ropa térmica."); status = "info"
    if aqi >= 3: protocols.append("☣️ AIRE TÓXICO/MALO: Cubrebocas N95."); status = "danger"
    elif aqi == 2: protocols.append("😷 AIRE REGULAR: Precaución."); 
    if status != "danger": status = "warning"
    if not protocols: protocols.append("✅ OPERACIÓN NORMAL")
    return protocols, status

# --- 4. CARGA DE RECURSOS ---
@st.cache_resource
def load_ai_resources():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "modelo_lstm_multivariado.h5")
    scaler_path = os.path.join(script_dir, "scaler_multivariado.pkl")
    if not os.path.exists(model_path): return None, None
    return load_model(model_path), joblib.load(scaler_path)

def get_live_data(lat, lon, api_key):
    try:
        w = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric").json()
        a = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}").json()
        wind_s = w['wind']['speed']; wind_d = w['wind']['deg']
        return w['main']['temp'], w['main']['humidity'], a['list'][0]['main']['aqi'], a['list'][0]['components'], wind_s, wind_d
    except: return None, None, None, None, None, None

# --- 5. INTERFAZ ---
try:
    if "OPENWEATHER_KEY" in st.secrets: api_key = st.secrets["OPENWEATHER_KEY"]
    else: api_key = "7bb94235f544dd5e37b0262258a9cdbc"
except: api_key = "7bb94235f544dd5e37b0262258a9cdbc"

with st.sidebar:
    st.markdown("## 🌍 GlobalAir")
    st.markdown("### `v11.0 // PHYSICS`") 
    selected_city = st.selectbox("📍 UBICACIÓN", list(LOCATIONS.keys()), key="city_final")
    st.divider()
    map_layer = st.radio("Vista Mapa:", ["Táctico", "Satélite"])
    st.divider()
    refresh_rate = st.slider("Refresh (s):", 60, 300, 60)
    live_mode = st.toggle("🔴 VIGILANCIA", value=False)
    if live_mode:
        ph = st.empty()
        for i in range(refresh_rate, 0, -1): ph.caption(f"Scan: {i}s"); time.sleep(1)
        st.rerun()

model, scaler = load_ai_resources()
if selected_city in LOCATIONS: slat, slon = LOCATIONS[selected_city]
else: slat, slon = LOCATIONS["Monterrey (Centro)"]
temp, hum, aqi, comps, wind_s, wind_d = get_live_data(slat, slon, api_key)

# Tabs
tab1, tab2, tab4 = st.tabs(["📡 MONITOREO", "📈 ANALÍTICA", "🧪 SIMULADOR DE FUGAS"])

# TAB 1: MONITOR (Resumido para brevedad, usa tu lógica anterior)
with tab1:
    if temp:
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f"""<div class="css-card"><div class="big-metric">{temp}°C</div><div class="metric-label">TEMP</div></div>""", unsafe_allow_html=True)
        with k2: st.markdown(f"""<div class="css-card"><div class="big-metric">{wind_s} m/s</div><div class="metric-label">VIENTO {deg_to_cardinal(wind_d)}</div></div>""", unsafe_allow_html=True)
        with k3: st.markdown(f"""<div class="css-card"><div class="big-metric">Nivel {aqi}</div><div class="metric-label">CALIDAD AIRE</div></div>""", unsafe_allow_html=True)
        
        st.markdown("### 🗺️ RADAR")
        tiles = "CartoDB dark_matter" if map_layer == "Táctico" else "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attr = "Esri" if "ArcGIS" in tiles else "OpenStreetMap"
        m = folium.Map(location=[slat, slon], zoom_start=13, tiles=tiles, attr=attr)
        
        # Marcador Central
        folium.Marker([slat, slon], popup=selected_city, icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        st_folium(m, width="100%", height=400)

# TAB 2: ANALÍTICA (Tu código existente iría aquí)
with tab2:
    st.info("Funcionalidad de análisis histórico activa en background.")

# --- TAB 4: SIMULADOR DE FUGAS (FÍSICA AVANZADA) ---
with tab4:
    st.markdown("### 🧪 Modelo de Dispersión Gaussiana (Pluma Tóxica)")
    st.caption(f"Simulación de fuga química hipotética en: **{selected_city}** basada en condiciones de viento actuales.")
    
    col_sim1, col_sim2 = st.columns([1, 3])
    
    with col_sim1:
        st.markdown("#### Parámetros")
        # Datos reales del viento (automáticos)
        st.metric("Viento Real", f"{wind_s} m/s", deg_to_cardinal(wind_d))
        
        st.markdown("---")
        emission_rate = st.slider("Tasa Emisión (g/s):", 100, 5000, 1000)
        
        if st.button("⚠️ SIMULAR FUGA"):
            run_sim = True
        else:
            run_sim = False
            
    with col_sim2:
        if run_sim:
            with st.spinner("Calculando dinámica de fluidos atmosféricos..."):
                # 1. LLAMAR AL MOTOR CIENTÍFICO
                heatmap_data = fisica_ambiental.generar_pluma_toxica(
                    lat_origen=slat,
                    lon_origen=slon,
                    viento_vel=wind_s,
                    viento_dir_grados=wind_d,
                    q_emision=emission_rate
                )
                
                # 2. VISUALIZAR
                if not heatmap_data:
                    st.warning("El viento está en calma (0 m/s), la pluma no se dispersa horizontalmente.")
                else:
                    m_sim = folium.Map(location=[slat, slon], zoom_start=14, tiles="CartoDB dark_matter")
                    
                    # Capa de Calor (La Pluma Tóxica)
                    HeatMap(heatmap_data, radius=20, blur=15, gradient={0.2:'blue', 0.5:'lime', 0.8:'yellow', 1.0:'red'}).add_to(m_sim)
                    
                    # Marcador de la Fuente
                    folium.Marker(
                        [slat, slon], 
                        popup="FUENTE DE EMISIÓN", 
                        icon=folium.Icon(color="red", icon="fire", prefix='fa')
                    ).add_to(m_sim)
                    
                    # Flecha de Viento (Marcador simple indicando dirección)
                    folium.Marker(
                        [slat, slon],
                        icon=folium.DivIcon(html=f'<div style="font-size:24px; transform: rotate({wind_d-180}deg);">⬇️</div>')
                    ).add_to(m_sim)

                    st_folium(m_sim, width="100%", height=500)
                    
                    st.success(f"Simulación completada. Pluma proyectada a 5km en dirección {deg_to_cardinal(wind_d + 180)}.")
        else:
            st.info("Presiona 'SIMULAR FUGA' para calcular la trayectoria de contaminantes.")

# --- 0. CONFIGURACIÓN ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

st.set_page_config(page_title="GlobalAir", page_icon="🌍", layout="wide")

# --- 1. ESTILOS (LIQUID GLASS + PULSO) ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #000000, #1a1a1a, #0d0d0d, #2b2b2b);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient { 0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;} }
    
    .css-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
    }
    
    /* ANIMACIÓN DE PULSO PARA ALERTAS */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 82, 82, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); }
    }
    
    h1, h2, h3 { color: #ffffff !important; font-family: 'Helvetica Neue', sans-serif; }
    .big-metric { font-size: 38px; font-weight: 800; color: #FFF; }
    .metric-label { font-size: 12px; text-transform: uppercase; color: rgba(255,255,255,0.7); }
    iframe { border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. UBICACIONES ---
LOCATIONS = {
    "Monterrey (Centro)": [25.6866, -100.3161],
    "San Pedro": [25.6576, -100.4029],
    "Apodaca (Aeropuerto)": [25.7785, -100.1864],
    "Guadalupe": [25.6774, -100.2597],
    "San Nicolás": [25.7413, -100.2953],
    "Santa Catarina": [25.6768, -100.4627],
    "Escobedo": [25.7969, -100.3260],
    "García": [25.8119, -100.5928],
    "Juárez": [25.6493, -100.0951],
    "Santiago": [25.4317, -100.1533],
    "Cadereyta": [25.5879, -99.9976],
    "Cemex (Planta Mty)": [25.7080, -100.2960],
    "Ternium (Guerrero)": [25.7480, -100.2930],
    "Kia (Pesquería)":    [25.7735, -99.9565]
}

# --- 3. LÓGICA DE NEGOCIO ---
def deg_to_cardinal(deg):
    """Convierte grados de viento a dirección (N, NE, E...)"""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((deg + 11.25)/22.5)
    return dirs[ix % 16]

def calculate_heat_index(temp, humidity):
    hi = 0.5 * (temp + 61.0 + ((temp-68.0)*1.2) + (humidity*0.094))
    if hi > 80:
        hi = -42.379 + 2.04901523*temp + 10.14333127*humidity - .22475541*temp*humidity - .00683783*temp*temp - .05481717*humidity*humidity + .00122874*temp*temp*humidity + .00085282*temp*humidity*humidity - .00000199*temp*temp*humidity*humidity
    return temp + (0.33 * humidity / 10) if temp > 26 else temp

def get_nom015_status(heat_index):
    if heat_index < 28: return "RIESGO BAJO", "100% Trabajo / 0% Descanso", "Hidratacion normal."
    elif 28 <= heat_index < 30: return "PRECAUCION", "100% Trabajo (Vigilancia)", "Agua fresca disponible."
    elif 30 <= heat_index < 32: return "RIESGO MODERADO", "75% Trabajo / 25% Descanso", "Descanso 15 min/hora."
    elif 32 <= heat_index < 54: return "RIESGO ALTO", "50% Trabajo / 50% Descanso", "Descanso 30 min/hora."
    else: return "PELIGRO EXTREMO", "0% Trabajo / 100% Descanso", "SUSPENSION DE ACTIVIDADES."

def get_status_color(temp, aqi):
    if aqi >= 4: return "#D500F9", 4000  
    if aqi == 3: return "#FF1744", 3500  
    if aqi == 2: return "#FF9100", 2500  
    if temp >= 38: return "#FF3D00", 3000 
    if temp < 12:  return "#2979FF", 2500 
    return "#00E676", 1500 

def get_protocols(temp, aqi):
    protocols = []
    status = "normal"
    if temp >= 38: protocols.append("🔥 GOLPE DE CALOR: Hidratación obligatoria."); status = "danger"
    elif temp < 12: protocols.append("❄️ BAJAS TEMPERATURAS: Ropa térmica."); status = "info"
    if aqi >= 3: protocols.append("☣️ AIRE TÓXICO/MALO: Cubrebocas N95."); protocols.append("⚠️ ALERTA AMBIENTAL ACTIVA"); status = "danger"
    elif aqi == 2: protocols.append("😷 AIRE REGULAR: Precaución grupos sensibles."); 
    if status != "danger": status = "warning"
    if not protocols: protocols.append("✅ OPERACIÓN NORMAL")
    return protocols, status

# --- 4. MOTOR DE PDF ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'GlobalAir - REPORTE DE CUMPLIMIENTO AMBIENTAL', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()} - Generado por GlobalAir System v1.0', 0, 0, 'C')

def create_pdf_download(city, temp, hum, aqi, pred, comps, wind_s, wind_d):
    pdf = PDFReport(); pdf.add_page(); pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Ubicacion: {city}", ln=True); pdf.ln(10)
    
    pdf.set_fill_color(200, 220, 255); pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, "1. DATOS METEOROLOGICOS Y DISPERSION", 1, 1, 'L', 1)
    pdf.set_font("Arial", '', 10)
    pdf.cell(50, 8, f"Temp: {temp} C", 1); pdf.cell(50, 8, f"Humedad: {hum}%", 1); pdf.ln()
    pdf.cell(50, 8, f"Viento: {wind_s} m/s", 1); pdf.cell(50, 8, f"Direccion: {deg_to_cardinal(wind_d)}", 1); pdf.ln(5)

    heat_index = calculate_heat_index(temp, hum)
    riesgo_nom, regimen, accion = get_nom015_status(heat_index)
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(255, 200, 200) if heat_index > 30 else pdf.set_fill_color(220, 255, 220)
    pdf.cell(0, 8, f"2. ANALISIS NOM-015-STPS (CALOR)", 1, 1, 'L', 1)
    pdf.set_font("Arial", '', 10)
    pdf.cell(50, 8, f"Indice Termico: {heat_index:.1f} C", 1); pdf.cell(0, 8, f"Riesgo: {riesgo_nom}", 1, 1)
    pdf.multi_cell(0, 6, f"Regimen: {regimen}\nAccion: {accion}", 1); pdf.ln(5)

    pdf.set_font("Arial", 'B', 10); pdf.set_fill_color(230, 230, 250)
    pdf.cell(0, 8, "3. CALIDAD DEL AIRE", 1, 1, 'L', 1)
    pdf.set_font("Arial", '', 10)
    pdf.cell(50, 8, f"AQI: Nivel {aqi}", 1); pdf.cell(0, 8, f"PM2.5: {comps.get('pm2_5')} ug/m3", 1, 1)
    pdf.cell(0, 8, f"Prediccion IA (1h): {pred:.1f} C", 1, 1); pdf.ln(15)
    
    pdf.cell(90, 10, "_"*30, 0, 0, 'C'); pdf.cell(0, 10, "_"*30, 0, 1, 'C')
    pdf.cell(90, 5, "Firma Supervisor EHS", 0, 0, 'C'); pdf.cell(0, 5, "Firma Gerencia", 0, 1, 'C')
    return pdf.output(dest='S').encode('latin-1')

# --- 5. FUNCIONES BACKEND ---
@st.cache_resource
def load_ai_resources():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "modelo_lstm_multivariado.h5")
    scaler_path = os.path.join(script_dir, "scaler_multivariado.pkl")
    if not os.path.exists(model_path): return None, None
    return load_model(model_path), joblib.load(scaler_path)

def get_live_data(lat, lon, api_key):
    try:
        w = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric").json()
        a = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}").json()
        # EXTRAEMOS VIENTO TAMBIÉN
        wind_s = w['wind']['speed']
        wind_d = w['wind']['deg']
        return w['main']['temp'], w['main']['humidity'], a['list'][0]['main']['aqi'], a['list'][0]['components'], wind_s, wind_d
    except: return None, None, None, None, None, None

def predict_temp_advanced(model, scaler, current_data_dict):
    temp = current_data_dict['temp']; pm25 = current_data_dict['comps'].get('pm2_5', 0); co = current_data_dict['comps'].get('co', 0); no2 = current_data_dict['comps'].get('no2', 0); o3 = current_data_dict['comps'].get('o3', 0)
    current_vector = np.array([[temp, pm25, co, no2, o3]])
    input_sequence = np.repeat(current_vector, 24, axis=0) 
    input_sequence = input_sequence + np.random.normal(0, 0.1, input_sequence.shape)
    input_scaled = scaler.transform(input_sequence)
    input_reshaped = input_scaled.reshape(1, 24, 5)
    pred_scaled = model.predict(input_reshaped)
    dummy_row = np.zeros((1, 5)); dummy_row[0, 0] = pred_scaled
    pred_final = scaler.inverse_transform(dummy_row)[0][0]
    return pred_final

def load_historical_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "historial_clima.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path); df['timestamp'] = pd.to_datetime(df['timestamp'])
            for c in ['temperatura', 'pm2_5', 'co', 'no2', 'o3']:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            return df
        except: pass
    return pd.DataFrame()

# --- FUNCIONES PRONÓSTICO CACHÉ ---
def _get_initial_sequence_uncached(scaler):
    LOOK_BACK = 24; FEATURES = ['temperatura', 'pm2_5', 'co', 'no2', 'o3']
    df = load_historical_csv()
    if not df.empty and len(df) >= LOOK_BACK:
        last_data = df.dropna(subset=FEATURES).tail(LOOK_BACK)
        if len(last_data) == LOOK_BACK:
            scaled_data = scaler.transform(last_data[FEATURES])
            return scaled_data.reshape(1, LOOK_BACK, len(FEATURES))
    temp, pm25, co, no2, o3 = 20, 10, 100, 10, 20 
    current_vector = np.array([[temp, pm25, co, no2, o3]])
    input_sequence = np.repeat(current_vector, 24, axis=0)
    return scaler.transform(input_sequence).reshape(1, LOOK_BACK, len(FEATURES))

def _predict_future_sequence_uncached(model, scaler, initial_sequence, n_steps=48):
    future_predictions_scaled = []
    current_batch = initial_sequence.copy()
    for i in range(n_steps):
        next_pred_scaled = model.predict(current_batch)[0] 
        future_predictions_scaled.append(next_pred_scaled[0])
        last_contaminants = current_batch[0, -1, 1:].copy()
        last_contaminants[1] = last_contaminants[1] * (1 + np.sin(i * 0.5) * 0.1) 
        new_step_features = np.insert(last_contaminants, 0, next_pred_scaled[0])
        new_step_reshaped = new_step_features.reshape(1, 1, 5)
        current_batch = np.append(current_batch[:, 1:, :], new_step_reshaped, axis=1)
    dummy_array = np.zeros((n_steps, 5))
    dummy_array[:, 0] = future_predictions_scaled
    return scaler.inverse_transform(dummy_array)[:, 0] 

@st.cache_data(ttl=3600)
def generate_cached_forecast(n_steps=48):
    model, scaler = load_ai_resources()
    if model is None or scaler is None: return None
    initial_seq = _get_initial_sequence_uncached(scaler)
    future_temps = _predict_future_sequence_uncached(model, scaler, initial_seq, n_steps)
    now = datetime.now()
    future_timestamps = [now + timedelta(hours=i) for i in range(1, n_steps + 1)]
    return pd.DataFrame({'timestamp': future_timestamps, 'temperatura': future_temps})

# --- 7. INTERFAZ PRINCIPAL ---
try:
    if "OPENWEATHER_KEY" in st.secrets: api_key = st.secrets["OPENWEATHER_KEY"]
    else: api_key = "7bb94235f544dd5e37b0262258a9cdbc"
except: api_key = "7bb94235f544dd5e37b0262258a9cdbc"

with st.sidebar:
    st.markdown("## 🌍 GlobalAir")
    selected_city = st.selectbox("📍 UBICACIÓN OBJETIVO", list(LOCATIONS.keys()), key="city_sel_final")
    st.divider()
    # NUEVO CONTROL DE CAPAS CON SATÉLITE
    map_layer = st.radio("Modo de Visualización:", ["Táctico (Oscuro)", "Satélite (Real)", "Híbrido"])
    st.divider()
    protocol_container = st.empty()
    st.divider()
    refresh_rate = st.slider("Refresh (s):", 60, 300, 60)
    live_mode = st.toggle("🔴 MODO VIGILANCIA", value=False)
    if live_mode:
        ph = st.empty()
        for i in range(refresh_rate, 0, -1):
            ph.caption(f"Scan: {i}s"); time.sleep(1)
        st.rerun()

model, scaler = load_ai_resources()
if selected_city in LOCATIONS: slat, slon = LOCATIONS[selected_city]
else: slat, slon = LOCATIONS["Monterrey (Centro)"]
# AHORA OBTENEMOS VIENTO TAMBIÉN
temp, hum, aqi, comps, wind_s, wind_d = get_live_data(slat, slon, api_key)

if temp:
    recos, status = get_protocols(temp, aqi)
    color_map = {"danger": "#FF1744", "warning": "#FF9100", "info": "#2979FF", "normal": "#00E676"}
    with protocol_container.container():
        st.markdown(f"""<div style="background:rgba(255,255,255,0.05); border-radius:10px; padding:10px; border-left:4px solid {color_map[status]};"><small style="color:{color_map[status]}">ESTATUS:</small><br>{'<br>'.join(recos)}</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📡 MONITOREO TÁCTICO", "📈 ANALÍTICA", "🔮 PRONÓSTICO 48H"])

with tab1:
    if model and temp:
        data_packet = {'temp': temp, 'comps': comps}
        pred = predict_temp_advanced(model, scaler, data_packet)
        error = abs(temp - pred)
        k1, k2, k3, k4 = st.columns(4)
        temp_col = "#FF3D00" if temp >= 35 else "#FFF"
        aqi_col = "#D500F9" if aqi >= 4 else ("#FF1744" if aqi == 3 else ("#FF9100" if aqi == 2 else "#00E676"))
        aqi_lbl = ["BUENO", "REGULAR", "MALO (ALERTA)", "MUY MALO", "PELIGROSO"][aqi-1] if 1<=aqi<=5 else "N/A"

        with k1: st.markdown(f"""<div class="css-card"><div class="metric-label">TEMP. ACTUAL</div><div class="big-metric" style="color:{temp_col}">{temp:.1f}°C</div></div>""", unsafe_allow_html=True)
        with k2: st.markdown(f"""<div class="css-card"><div class="metric-label">IA PREDICTIVA</div><div class="big-metric" style="color:#00E5FF">{pred:.1f}°C</div></div>""", unsafe_allow_html=True)
        with k3: st.markdown(f"""<div class="css-card"><div class="metric-label">VIENTO</div><div class="big-metric">{wind_s} m/s</div><div style="color:#888; font-size:12px">Dir: {deg_to_cardinal(wind_d)}</div></div>""", unsafe_allow_html=True)
        with k4: st.markdown(f"""<div class="css-card"><div class="metric-label">CALIDAD AIRE</div><div class="big-metric" style="color:{aqi_col}">{aqi_lbl}</div></div>""", unsafe_allow_html=True)

        # Matrix View
        with st.expander("📋 Ver Resumen de Todas las Plantas (Matrix View)", expanded=False):
            matrix_data = []
            for c_name, c_coords in LOCATIONS.items():
                if c_name == selected_city:
                    icon = "🔴" if status=="danger" else ("🟡" if status=="warning" else "🟢")
                    matrix_data.append({"Ubicación": c_name, "Temp": f"{temp}°C", "AQI": f"Nivel {aqi}", "Viento": f"{wind_s} m/s {deg_to_cardinal(wind_d)}", "Estado": icon})
                else:
                    matrix_data.append({"Ubicación": c_name, "Temp": "---", "AQI": "---", "Viento": "---", "Estado": "⚪"})
            st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)

        st.markdown("---")
        c_pdf1, c_pdf2 = st.columns([3, 1])
        with c_pdf1: st.markdown("#### 📄 Reporte de Cumplimiento"); st.caption(f"Evidencia forense para: **{selected_city}**")
        with c_pdf2:
            # PDF CON VIENTO
            pdf_data = create_pdf_download(selected_city, temp, hum, aqi, pred, comps, wind_s, wind_d)
            st.download_button(label="📥 DESCARGAR PDF", data=pdf_data, file_name=f"Reporte_GlobalAir_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", type="primary")
        st.markdown("---")

        st.markdown("### 🗺️ RADAR METROPOLITANO")
        
        # SELECCIÓN DE CAPAS DE MAPA
        if map_layer == "Táctico (Oscuro)":
            tiles = "CartoDB dark_matter"
        elif map_layer == "Satélite (Real)":
            tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        else: # Híbrido
            tiles = "OpenStreetMap" # O alguna otra capa clara
            
        attr = "Esri" if "ArcGIS" in tiles else "OpenStreetMap"
        m = folium.Map(location=[25.69, -100.32], zoom_start=11, tiles=tiles, attr=attr)
        
        # Botón de Pantalla Completa
        Fullscreen().add_to(m)

        for city, coords in LOCATIONS.items():
            if city == selected_city: ct, caqi = temp, aqi
            else: ct, caqi = None, None 
            
            if ct:
                # LÓGICA DE POLÍGONOS + TOOLTIP CON VIENTO
                hex_color, _ = get_status_color(ct, caqi)
                offset = 0.008
                poly = [[coords[0]-offset, coords[1]-offset], [coords[0]+offset, coords[1]-offset], [coords[0]+offset, coords[1]+offset], [coords[0]-offset, coords[1]+offset]]
                
                popup_content = f"""
                <div style="font-family:sans-serif">
                    <b>{city}</b><hr>
                    🌡️ Temp: {ct}°C<br>
                    🌫️ AQI: {caqi}<br>
                    💨 Viento: {wind_s} m/s ({deg_to_cardinal(wind_d)})
                </div>
                """
                
                folium.Polygon(locations=poly, color=hex_color, fill=True, fill_color=hex_color, fill_opacity=0.3, popup=folium.Popup(popup_content, max_width=200)).add_to(m)
                
                # Icono simple si es satélite para que resalte
                icon_color = "white" if "ArcGIS" in tiles else "black"
                folium.Marker(location=coords, icon=folium.DivIcon(html=f'<div style="color:{icon_color};font-size:9pt;text-shadow:0 0 4px #000;font-weight:bold">{city}</div>')).add_to(m)
            else:
                 folium.Marker(location=coords, icon=folium.DivIcon(html=f'<div style="color:#777;font-size:9pt;">{city}</div>')).add_to(m)

        st_folium(m, width="100%", height=550)
    
    else:
        st.error("⚠️ Error de Conexión con Sensores")
        st.info("El sistema no pudo recuperar datos en tiempo real.")

# ... (PESTAÑAS 2 y 3 se mantienen igual que antes) ...
# PESTAÑA 2: ANALÍTICA
with tab2:
    st.markdown("### 📊 HISTÓRICO MULTIVARIABLE")
    df = load_historical_csv()
    if not df.empty:
        c1, c2 = st.columns([1,3])
        with c1:
            days = st.slider("Días:", 1, 30, 3)
            cols = [c for c in ['temperatura', 'pm2_5', 'co', 'no2', 'o3'] if c in df.columns]
            param = st.selectbox("Parámetro:", cols)
        cutoff = datetime.now() - pd.Timedelta(days=days); data = df[df['timestamp'] > cutoff].copy()
        if not data.empty:
            p_colors = {'temperatura': '#FF3D00', 'pm2_5': '#D500F9', 'co': '#B0BEC5', 'no2': '#795548', 'o3': '#00E5FF'}
            curr_color = p_colors.get(param, '#FFF')
            chart = alt.Chart(data).mark_area(line={'color': curr_color}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color=curr_color, offset=1), alt.GradientStop(color='rgba(0,0,0,0)', offset=0)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('timestamp:T', title=None), y=alt.Y(f'{param}:Q', scale=alt.Scale(zero=False), title=param.upper()), tooltip=['timestamp', param]).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
    else: st.info("Esperando datos del recolector...")

# PESTAÑA 3: PRONÓSTICO
with tab3:
    st.markdown("### 🔮 Pronóstico Extendido (48 Horas)")
    st.caption(f"Simulación avanzada para: **{selected_city}**")
    forecast_df = generate_cached_forecast(n_steps=48)
    if forecast_df is None: 
        st.error("Error: No hay modelo o datos suficientes para pronosticar.")
    else:
        base = alt.Chart(forecast_df).encode(x=alt.X('timestamp:T', title="Hora"), tooltip=['timestamp', 'temperatura'])
        linea = base.mark_line(color="#00E5FF", point=True).encode(y=alt.Y('temperatura:Q', title="Temp (°C)", scale=alt.Scale(zero=False)))
        umbral = alt.Chart(pd.DataFrame({'u': [35.0]})).mark_rule(color="#FF3D00", strokeDash=[5,5]).encode(y='u:Q')
        st.altair_chart((linea + umbral).properties(height=400).interactive(), use_container_width=True)
        st.info("Nota: La simulación asume ciclos de tráfico (CO) sinusoidales para mayor realismo.")


