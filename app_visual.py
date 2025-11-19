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

# --- 0. CONFIGURACIÓN ANTI-ERRORES ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Monterrey AI Sentinel", page_icon="🛡️", layout="wide")

# --- 1. ESTÉTICA "LIQUID GLASS" ---
st.markdown("""
    <style>
    /* Animación de Fondo */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #141e30);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* Tarjetas de Vidrio */
    .css-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        transition: transform 0.2s;
    }
    .css-card:hover { transform: translateY(-3px); }
    
    h1, h2, h3 { color: #ffffff !important; font-family: 'Helvetica Neue', sans-serif; }
    .big-metric { font-size: 38px; font-weight: 800; color: #FFF; text-shadow: 0 0 10px rgba(255,255,255,0.3); }
    .metric-label { font-size: 12px; text-transform: uppercase; color: rgba(255,255,255,0.7); letter-spacing: 1px; }
    
    /* Ajustes de Mapa y Tablas */
    iframe { border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); }
    [data-testid="stExpander"] { background-color: rgba(0,0,0,0.2); border-radius: 10px; }
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
    "Cadereyta": [25.5879, -99.9976]
}

# --- 3. LÓGICA DE NEGOCIO (UMBRALES AJUSTADOS) ---
def get_status_color(temp, aqi):
    # Umbrales más estrictos para AQI
    if aqi >= 4: return "#D500F9", 4000  # Púrpura (Malo/Peligroso)
    if aqi == 3: return "#FF1744", 3500  # Rojo (Moderado alto)
    if aqi == 2: return "#FF9100", 2500  # Naranja (Aceptable/Precaución)
    
    # Temperatura
    if temp >= 38: return "#FF3D00", 3000 # Rojo Fuego
    if temp < 12:  return "#2979FF", 2500 # Azul Frío
    
    return "#00E676", 1500 # Verde (Solo AQI 1 es verde)

def get_protocols(temp, aqi):
    protocols = []
    status = "normal"
    
    # Temperatura
    if temp >= 38:
        protocols.append("🔥 GOLPE DE CALOR: Hidratación obligatoria.")
        status = "danger"
    elif temp < 12:
        protocols.append("❄️ BAJAS TEMPERATURAS: Ropa térmica.")
        status = "info"
        
    # AQI (Ajustado: Nivel 3 ya es peligro)
    if aqi >= 3:
        protocols.append("☣️ AIRE TÓXICO/MALO: Cubrebocas N95.")
        protocols.append("⚠️ ALERTA AMBIENTAL ACTIVA")
        status = "danger"
    elif aqi == 2:
        protocols.append("😷 AIRE REGULAR: Precaución grupos sensibles.")
        if status != "danger": status = "warning"
        
    if not protocols: protocols.append("✅ OPERACIÓN NORMAL")
    return protocols, status

# --- 4. MOTOR DE PDF ---
class PDFReport(FPDF):
    def header(self): self.set_font('Arial', 'B', 15); self.cell(0, 10, 'AI SENTINEL - REPORTE DE SITUACION', 0, 1, 'C'); self.ln(10)
    def footer(self): self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def create_pdf_download(city, temp, aqi, pred, error, recos, comps):
    pdf = PDFReport(); pdf.add_page(); pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Ubicacion: {city}", ln=True); pdf.ln(10)
    pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "METRICAS CLAVE", ln=True); pdf.set_font("Arial", size=12)
    pdf.cell(90, 10, f"Temperatura: {temp} C", border=1); pdf.cell(90, 10, f"IA Prediccion: {pred:.2f} C", border=1); pdf.ln()
    pdf.cell(90, 10, f"AQI: Nivel {aqi}", border=1); pdf.cell(90, 10, f"PM2.5: {comps.get('pm2_5')} ug/m3", border=1); pdf.ln(15)
    pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "PROTOCOLOS ACTIVOS", ln=True); pdf.set_font("Arial", size=12)
    for rec in recos:
        clean = rec.replace("🔥", "[CALOR]").replace("❄️", "[FRIO]").replace("☣️", "[PELIGRO]").replace("😷", "[AIRE]").replace("✅", "[OK]").replace("⚠️", "[ALERTA]")
        pdf.cell(0, 10, f"- {clean}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 5. FUNCIONES BACKEND E IA ---
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
        return w['main']['temp'], w['main']['humidity'], a['list'][0]['main']['aqi'], a['list'][0]['components']
    except: return None, None, None, None

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

# --- 6. FUNCIONES DE PRONÓSTICO OPTIMIZADAS (CACHÉ) ---
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
    st.markdown("## 💧 AI SENTINEL")
    st.markdown("### `v8.0 // FINAL`") 
    selected_city = st.selectbox("📍 UBICACIÓN OBJETIVO", list(LOCATIONS.keys()), key="city_final_v8")
    st.divider()
    layer_type = st.radio("Modo de Mapa:", ["Táctico (Polígonos)", "Científico (Heatmap)"])
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
temp, hum, aqi, comps = get_live_data(slat, slon, api_key)

if temp:
    recos, status = get_protocols(temp, aqi)
    color_map = {"danger": "#FF1744", "warning": "#FF9100", "info": "#2979FF", "normal": "#00E676"}
    with protocol_container.container():
        st.markdown(f"""<div style="background:rgba(255,255,255,0.05); border-radius:10px; padding:10px; border-left:4px solid {color_map[status]};"><small style="color:{color_map[status]}">ESTATUS:</small><br>{'<br>'.join(recos)}</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📡 MONITOREO TÁCTICO", "📈 ANALÍTICA", "🔮 PRONÓSTICO 48H"])

# --- PESTAÑA 1: DASHBOARD ---
with tab1:
    if model and temp:
        data_packet = {'temp': temp, 'comps': comps}
        pred = predict_temp_advanced(model, scaler, data_packet)
        error = abs(temp - pred)
        k1, k2, k3, k4 = st.columns(4)
        temp_col = "#FF3D00" if temp >= 35 else "#FFF"
        # Colores AQI ajustados (2=Naranja, 3=Rojo, 4+=Púrpura)
        aqi_col = "#D500F9" if aqi >= 4 else ("#FF1744" if aqi == 3 else ("#FF9100" if aqi == 2 else "#00E676"))
        aqi_lbl = ["BUENO", "REGULAR", "MALO (ALERTA)", "MUY MALO", "PELIGROSO"][aqi-1] if 1<=aqi<=5 else "N/A"

        with k1: st.markdown(f"""<div class="css-card"><div class="metric-label">TEMP. ACTUAL</div><div class="big-metric" style="color:{temp_col}">{temp:.1f}°C</div></div>""", unsafe_allow_html=True)
        with k2: st.markdown(f"""<div class="css-card"><div class="metric-label">IA PREDICTIVA</div><div class="big-metric" style="color:#00E5FF">{pred:.1f}°C</div></div>""", unsafe_allow_html=True)
        with k3: st.markdown(f"""<div class="css-card"><div class="metric-label">DESVIACIÓN</div><div class="big-metric">{error:.1f}°C</div></div>""", unsafe_allow_html=True)
        with k4: st.markdown(f"""<div class="css-card"><div class="metric-label">CALIDAD AIRE</div><div class="big-metric" style="color:{aqi_col}">{aqi_lbl}</div></div>""", unsafe_allow_html=True)

        # --- MATRIX VIEW "LITE" (OPTIMIZADA) ---
        with st.expander("📋 Ver Resumen de Todas las Plantas (Matrix View)", expanded=False):
            matrix_data = []
            for c_name, c_coords in LOCATIONS.items():
                if c_name == selected_city:
                    # Solo mostramos datos reales de la ciudad seleccionada para no quemar la API
                    icon = "🔴" if status=="danger" else ("🟡" if status=="warning" else "🟢")
                    matrix_data.append({"Ubicación": c_name, "Temp": f"{temp}°C", "AQI": f"Nivel {aqi}", "Estado": icon})
                else:
                    matrix_data.append({"Ubicación": c_name, "Temp": "---", "AQI": "---", "Estado": "⚪"})
            st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)
            st.caption("Nota: Datos en tiempo real limitados a la ubicación activa para optimización.")

        st.markdown("---")
        c_pdf1, c_pdf2 = st.columns([3, 1])
        with c_pdf1: st.markdown("#### 📄 Reporte de Cumplimiento"); st.caption(f"Evidencia forense para: **{selected_city}**")
        with c_pdf2:
            pdf_data = create_pdf_download(selected_city, temp, aqi, pred, error, recos, comps)
            st.download_button(label="📥 DESCARGAR PDF", data=pdf_data, file_name=f"Reporte_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", type="primary")
        st.markdown("---")

        st.markdown("### 🗺️ RADAR METROPOLITANO")
        m = folium.Map(location=[25.69, -100.32], zoom_start=11, tiles="CartoDB dark_matter")
        heat_data = []
        for city, coords in LOCATIONS.items():
            ct, _, caqi, _ = get_live_data(coords[0], coords[1], api_key)
            if ct:
                intensity = caqi * 0.2
                heat_data.append([coords[0], coords[1], intensity])
                if layer_type == "Táctico (Polígonos)":
                    hex_color, _ = get_status_color(ct, caqi)
                    offset = 0.005
                    poly = [[coords[0]-offset, coords[1]-offset], [coords[0]+offset, coords[1]-offset], [coords[0]+offset, coords[1]+offset], [coords[0]-offset, coords[1]+offset]]
                    folium.Polygon(locations=poly, color=hex_color, fill=True, fill_color=hex_color, fill_opacity=0.3, tooltip=f"<b>{city}</b><br>Temp: {ct}°C").add_to(m)
                    folium.Marker(location=coords, icon=folium.DivIcon(html=f'<div style="color:#EEE;font-size:9pt;text-shadow:0 0 4px #000;font-weight:bold">{city}</div>')).add_to(m)
        if layer_type == "Científico (Heatmap)": HeatMap(heat_data, radius=25, blur=15, gradient={0.2:'blue',0.4:'lime',0.6:'orange',1:'red'}).add_to(m)
        st_folium(m, width="100%", height=500)
    
    # --- MANEJO DE ERRORES DE API ---
    else:
        st.error("⚠️ Error de Conexión con Sensores")
        st.info("El sistema no pudo recuperar datos en tiempo real. Posibles causas: Límite de API excedido o falla de red.")
        if not model: st.warning("Diagnóstico: Modelo de IA no cargado.")
        if not temp: st.warning("Diagnóstico: API OpenWeather no responde (Rate Limit).")

# --- PESTAÑA 2: ANALÍTICA ---
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

# --- PESTAÑA 3: PRONÓSTICO ---
with tab3:
    st.markdown("### 🔮 Pronóstico Extendido (48 Horas)")
    st.caption(f"Simulación avanzada para: **{selected_city}**")
    with st.spinner("Ejecutando simulación de IA..."):
        forecast_df = generate_cached_forecast(n_steps=48)
        if forecast_df is None: st.error("Error en pronóstico.")
        else:
            base = alt.Chart(forecast_df).encode(x=alt.X('timestamp:T', title="Hora"), tooltip=['timestamp', 'temperatura'])
            linea = base.mark_line(color="#00E5FF", point=True).encode(y=alt.Y('temperatura:Q', title="Temp (°C)", scale=alt.Scale(zero=False)))
            umbral = alt.Chart(pd.DataFrame({'u': [35.0]})).mark_rule(color="#FF3D00", strokeDash=[5,5]).encode(y='u:Q')
            st.altair_chart((linea + umbral).properties(height=400).interactive(), use_container_width=True)
            st.info("Nota: La simulación asume ciclos de tráfico (CO) sinusoidales para mayor realismo.")
