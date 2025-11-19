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

# --- 0. CONFIGURACIÓN ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Monterrey AI Sentinel", page_icon="🛡️", layout="wide")

# --- 1. ESTILOS ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #141e30); background-size: 400% 400%; animation: gradient 15s ease infinite; }
    @keyframes gradient { 0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;} }
    .css-card { background: rgba(255, 255, 255, 0.05); border-radius: 20px; backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); padding: 25px; }
    h1, h2, h3 { color: #ffffff !important; font-family: 'Helvetica Neue', sans-serif; }
    .big-metric { font-size: 38px; font-weight: 800; color: #FFF; }
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
    "Cadereyta": [25.5879, -99.9976]
}

# --- 3. FUNCIONES LÓGICAS ---
def calculate_heat_index(temp, humidity):
    """Calcula la Sensación Térmica (Heat Index) aproximada."""
    # Fórmula simplificada de Rothfusz para el reporte
    hi = 0.5 * (temp + 61.0 + ((temp-68.0)*1.2) + (humidity*0.094))
    if hi > 80: # Fórmula completa si hace calor
        hi = -42.379 + 2.04901523*temp + 10.14333127*humidity - .22475541*temp*humidity - .00683783*temp*temp - .05481717*humidity*humidity + .00122874*temp*temp*humidity + .00085282*temp*humidity*humidity - .00000199*temp*temp*humidity*humidity
    
    # Ajuste simple a Celsius si la fórmula da Fahrenheit (depende de la implementación, aquí asumimos input C y output C ajustado)
    # Para este MVP, usamos una aproximación directa:
    heat_index_c = temp + (0.33 * humidity / 10) if temp > 26 else temp
    return heat_index_c

def get_nom015_status(heat_index):
    """Determina el régimen de trabajo según NOM-015 estimado."""
    if heat_index < 28:
        return "RIESGO BAJO", "100% Trabajo / 0% Descanso", "Hidratacion normal."
    elif 28 <= heat_index < 30:
        return "PRECAUCION", "100% Trabajo (Vigilancia)", "Agua fresca disponible, monitoreo cada hora."
    elif 30 <= heat_index < 32:
        return "RIESGO MODERADO", "75% Trabajo / 25% Descanso", "Descanso 15 min/hora en sombra. Hidratacion obligatoria."
    elif 32 <= heat_index < 54:
        return "RIESGO ALTO", "50% Trabajo / 50% Descanso", "Descanso 30 min/hora. Rotacion de personal."
    else:
        return "PELIGRO EXTREMO", "0% Trabajo / 100% Descanso", "SUSPENSION DE ACTIVIDADES. Peligro inminente de golpe de calor."

def get_status_color(temp, aqi):
    if aqi >= 5: return "#D500F9", 4000  
    if aqi == 4: return "#FF1744", 3500  
    if temp >= 38: return "#FF3D00", 3000 
    if temp < 12:  return "#2979FF", 2500 
    if aqi == 3: return "#FF9100", 2500  
    return "#00E676", 1500              

def get_protocols(temp, aqi):
    protocols = []
    status = "normal"
    if temp >= 38:
        protocols.append("🔥 GOLPE DE CALOR: Hidratación obligatoria.")
        status = "danger"
    elif temp < 12:
        protocols.append("❄️ BAJAS TEMPERATURAS: Ropa térmica.")
        status = "info"
    if aqi >= 4:
        protocols.append("☣️ AIRE TÓXICO: Cubrebocas N95.")
        status = "danger"
    elif aqi == 3:
        protocols.append("😷 AIRE MODERADO: Reducir ejercicio.")
        if status != "danger": status = "warning"
    if not protocols: protocols.append("✅ OPERACIÓN NORMAL")
    return protocols, status

# --- MOTOR PDF PROFESIONAL ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI SENTINEL - REPORTE DE CUMPLIMIENTO AMBIENTAL', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()} - Generado por AI Sentinel System v7.0', 0, 0, 'C')

def create_pdf_download(city, temp, hum, aqi, pred, comps):
    pdf = PDFReport()
    pdf.add_page()
    
    # Título de Sección
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, f"1. DATOS GENERALES DE LA MEDICION", 1, 1, 'L', 1)
    pdf.set_font("Arial", '', 10)
    pdf.cell(100, 8, f"Ubicacion: {city}", 1)
    pdf.cell(0, 8, f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 1, 1)
    pdf.ln(5)
    
    # Sección NOM-015 (Calor)
    heat_index = calculate_heat_index(temp, hum)
    riesgo_nom, regimen, accion = get_nom015_status(heat_index)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(255, 200, 200) if heat_index > 30 else pdf.set_fill_color(220, 255, 220)
    pdf.cell(0, 8, f"2. ANALISIS DE ESTRES TERMICO (NOM-015-STPS)", 1, 1, 'L', 1)
    
    pdf.set_font("Arial", '', 10)
    pdf.cell(50, 8, "Temp. Bulbo Seco (Aire):", 1); pdf.cell(45, 8, f"{temp} C", 1)
    pdf.cell(50, 8, "Humedad Relativa:", 1); pdf.cell(0, 8, f"{hum}%", 1, 1)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(50, 8, "INDICE TERMICO (Est.):", 1); pdf.cell(45, 8, f"{heat_index:.1f} C", 1)
    pdf.cell(50, 8, "NIVEL DE RIESGO:", 1); pdf.cell(0, 8, f"{riesgo_nom}", 1, 1)
    pdf.ln(2)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.multi_cell(0, 6, f"REGIMEN SUGERIDO (Carga Media): {regimen}")
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 6, f"Accion Inmediata: {accion}")
    pdf.ln(5)

    # Sección Calidad Aire
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(230, 230, 250)
    pdf.cell(0, 8, f"3. CALIDAD DEL AIRE Y PREVISION", 1, 1, 'L', 1)
    pdf.set_font("Arial", '', 10)
    pdf.cell(50, 8, f"AQI Actual: Nivel {aqi}", 1); pdf.cell(0, 8, f"PM2.5: {comps.get('pm2_5')} ug/m3", 1, 1)
    pdf.cell(50, 8, "Prediccion IA (1h):", 1); pdf.cell(0, 8, f"{pred:.1f} C (Tendencia calculada por LSTM)", 1, 1)
    pdf.ln(15)
    
    # Firmas
    pdf.set_font("Arial", '', 10)
    pdf.cell(90, 10, "_"*30, 0, 0, 'C')
    pdf.cell(0, 10, "_"*30, 0, 1, 'C')
    pdf.cell(90, 5, "Firma Supervisor EHS", 0, 0, 'C')
    pdf.cell(0, 5, "Firma Gerencia de Planta", 0, 1, 'C')
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 7)
    pdf.multi_cell(0, 4, "NOTA: Este reporte utiliza datos satelitales y modelos de IA. Para cumplimiento legal estricto de NOM-015, se requieren mediciones en sitio con termometro de globo (TGBH). Este documento sirve como guia preventiva.")

    return pdf.output(dest='S').encode('latin-1')

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

# Nuevas funciones caché para pronóstico (igual que antes)
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

# --- 5. INTERFAZ ---
try:
    if "OPENWEATHER_KEY" in st.secrets: api_key = st.secrets["OPENWEATHER_KEY"]
    else: api_key = "352a68073cee67874db9a5892b8d1d8a"
except: api_key = "352a68073cee67874db9a5892b8d1d8a"

with st.sidebar:
    st.markdown("## 💧 AI SENTINEL")
    st.markdown("### `v7.1 // NOM-015`") 
    selected_city = st.selectbox("📍 UBICACIÓN OBJETIVO", list(LOCATIONS.keys()), key="city_sel_nom")
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

with tab1:
    if model and temp:
        data_packet = {'temp': temp, 'comps': comps}
        pred = predict_temp_advanced(model, scaler, data_packet)
        error = abs(temp - pred)
        k1, k2, k3, k4 = st.columns(4)
        temp_col = "#FF3D00" if temp >= 35 else "#FFF"
        aqi_col = "#D500F9" if aqi >= 5 else ("#FF1744" if aqi == 4 else ("#FF9100" if aqi == 3 else "#00E676"))
        aqi_lbl = ["BUENO", "ACEPTABLE", "MODERADO", "MALO", "PELIGROSO"][aqi-1] if 1<=aqi<=5 else "N/A"

        with k1: st.markdown(f"""<div class="css-card"><div class="metric-label">TEMP. ACTUAL</div><div class="big-metric" style="color:{temp_col}">{temp:.1f}°C</div></div>""", unsafe_allow_html=True)
        with k2: st.markdown(f"""<div class="css-card"><div class="metric-label">IA PREDICTIVA</div><div class="big-metric" style="color:#00E5FF">{pred:.1f}°C</div></div>""", unsafe_allow_html=True)
        with k3: st.markdown(f"""<div class="css-card"><div class="metric-label">DESVIACIÓN</div><div class="big-metric">{error:.1f}°C</div></div>""", unsafe_allow_html=True)
        with k4: st.markdown(f"""<div class="css-card"><div class="metric-label">CALIDAD AIRE</div><div class="big-metric" style="color:{aqi_col}">{aqi_lbl}</div></div>""", unsafe_allow_html=True)

        with st.expander("📋 Ver Resumen de Todas las Plantas (Matrix View)", expanded=False):
            matrix_data = []
            for c_name, c_coords in LOCATIONS.items():
                if c_name == selected_city:
                    matrix_data.append({"Ubicación": c_name, "Temp": f"{temp}°C", "AQI": f"Nivel {aqi}", "Estado": "🔴" if status=="danger" else ("🟡" if status=="warning" else "🟢")})
                else:
                    matrix_data.append({"Ubicación": c_name, "Temp": "--", "AQI": "--", "Estado": "⚪"})
            st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)

        st.markdown("---")
        c_pdf1, c_pdf2 = st.columns([3, 1])
        with c_pdf1: st.markdown("#### 📄 Reporte NOM-015 (Oficial)"); st.caption(f"Descargar reporte de cumplimiento normativo para: **{selected_city}**")
        with c_pdf2:
            # PASAMOS HUMEDAD A LA FUNCIÓN DE PDF
            pdf_data = create_pdf_download(selected_city, temp, hum, aqi, pred, comps)
            st.download_button(label="📥 DESCARGAR PDF", data=pdf_data, file_name=f"Reporte_NOM015_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", type="primary")
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


