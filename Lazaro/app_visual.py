import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime
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

# --- 0. CONFIGURACIÓN INICIAL ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Monterrey AI Sentinel", page_icon="💧", layout="wide")

# --- 1. ESTÉTICA "LIQUID GLASS" (CSS ANIMADO) ---
st.markdown("""
    <style>
    /* FONDO LÍQUIDO ANIMADO */
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #141e30);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    /* TARJETAS DE CRISTAL LÍQUIDO */
    .css-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .css-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* TIPOGRAFÍA */
    h1, h2, h3 { 
        color: #ffffff !important; 
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
        letter-spacing: 1px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    .big-metric { 
        font-size: 38px; 
        font-weight: 800; 
        color: #FFF;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }
    
    .metric-label { 
        font-size: 12px; 
        text-transform: uppercase; 
        color: rgba(255,255,255,0.7); 
        letter-spacing: 2px; 
        font-weight: 600;
    }
    
    /* BOTONES PERSONALIZADOS */
    .stButton>button {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 30px;
        backdrop-filter: blur(5px);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: rgba(255, 255, 255, 0.3);
        border-color: white;
        box-shadow: 0 0 15px rgba(255,255,255,0.5);
    }
    
    /* AJUSTES MAPA */
    iframe { 
        border-radius: 20px; 
        border: 1px solid rgba(255,255,255,0.1); 
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
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

# --- 3. FUNCIONES DE NEGOCIO ---
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

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI SENTINEL - REPORTE DE SITUACION', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def create_pdf_download(city, temp, aqi, pred, error, recos, comps):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Ubicacion: {city}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "METRICAS CLAVE", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(90, 10, f"Temperatura: {temp} C", border=1)
    pdf.cell(90, 10, f"IA Prediccion: {pred:.2f} C", border=1)
    pdf.ln()
    pdf.cell(90, 10, f"AQI: Nivel {aqi}", border=1)
    pdf.cell(90, 10, f"PM2.5: {comps.get('pm2_5')} ug/m3", border=1)
    pdf.ln(15)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "PROTOCOLOS ACTIVOS", ln=True)
    pdf.set_font("Arial", size=12)
    for rec in recos:
        clean_rec = rec.replace("🔥", "[CALOR]").replace("❄️", "[FRIO]").replace("☣️", "[PELIGRO]").replace("😷", "[AIRE]").replace("✅", "[OK]")
        pdf.cell(0, 10, f"- {clean_rec}", ln=True)
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
    temp = current_data_dict['temp']
    pm25 = current_data_dict['comps'].get('pm2_5', 0)
    co = current_data_dict['comps'].get('co', 0)
    no2 = current_data_dict['comps'].get('no2', 0)
    o3 = current_data_dict['comps'].get('o3', 0)
    current_vector = np.array([[temp, pm25, co, no2, o3]])
    input_sequence = np.repeat(current_vector, 24, axis=0) 
    input_sequence = input_sequence + np.random.normal(0, 0.1, input_sequence.shape)
    input_scaled = scaler.transform(input_sequence)
    input_reshaped = input_scaled.reshape(1, 24, 5)
    pred_scaled = model.predict(input_reshaped)
    dummy_row = np.zeros((1, 5))
    dummy_row[0, 0] = pred_scaled
    pred_final = scaler.inverse_transform(dummy_row)[0][0]
    return pred_final

def load_historical_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "historial_clima.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for c in ['temperatura', 'pm2_5', 'co', 'no2', 'o3']:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            return df
        except: pass
    return pd.DataFrame()

# --- 5. INTERFAZ ---
try:
    if "OPENWEATHER_KEY" in st.secrets:
        api_key = st.secrets["OPENWEATHER_KEY"]
    else:
        api_key = "352a68073cee67874db9a5892b8d1d8a"
except: api_key = "352a68073cee67874db9a5892b8d1d8a"

with st.sidebar:
    st.markdown("## 💧 AI SENTINEL")
    st.markdown("### `v5.5 // LIQUID`") 
    
    selected_city = st.selectbox("📍 UBICACIÓN OBJETIVO", list(LOCATIONS.keys()), key="city_sel_lqd")
    
    st.divider()
    
    st.subheader("🗺️ Capas de Visualización")
    layer_type = st.radio("Modo de Mapa:", ["Táctico (Círculos)", "Científico (Heatmap)"])
    
    st.divider()
    protocol_container = st.empty()
    st.divider()
    refresh_rate = st.slider("Frecuencia (s):", 60, 300, 60)
    live_mode = st.toggle("🔴 MODO VIGILANCIA", value=False)
    if live_mode:
        ph = st.empty()
        for i in range(refresh_rate, 0, -1):
            ph.caption(f"Escaneando en: {i}s")
            time.sleep(1)
        st.rerun()

model, scaler = load_ai_resources()
if selected_city in LOCATIONS: slat, slon = LOCATIONS[selected_city]
else: slat, slon = LOCATIONS["Monterrey (Centro)"]

temp, hum, aqi, comps = get_live_data(slat, slon, api_key)

if temp:
    recos, status = get_protocols(temp, aqi)
    color_map = {"danger": "#FF1744", "warning": "#FF9100", "info": "#2979FF", "normal": "#00E676"}
    with protocol_container.container():
        st.markdown(f"""<div style="background:rgba(255,255,255,0.05); border-radius:10px; padding:10px; border-left:4px solid {color_map[status]};"><small style="color:{color_map[status]}">ESTATUS DE SISTEMA:</small><br>{'<br>'.join(recos)}</div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📡 MONITOREO TÁCTICO", "📈 ANALÍTICA"])

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

        st.markdown("---")
        c_pdf1, c_pdf2 = st.columns([3, 1])
        with c_pdf1:
            st.markdown("#### 📄 Reporte de Cumplimiento")
            st.caption(f"Evidencia forense para: **{selected_city}**")
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
                if layer_type == "Táctico (Círculos)":
                    hex_color, radius = get_status_color(ct, caqi)
                    folium.Circle(location=coords, radius=radius, color=hex_color, fill=True, fill_color=hex_color, fill_opacity=0.4, tooltip=f"{city}: {ct}°C").add_to(m)
                    folium.Marker(location=coords, icon=folium.DivIcon(html=f'<div style="color:#EEE;font-size:8pt;text-shadow:0 0 4px #000">{city}</div>')).add_to(m)
        
        if layer_type == "Científico (Heatmap)":
            HeatMap(heat_data, radius=25, blur=15, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(m)

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
        
        cutoff = datetime.now() - pd.Timedelta(days=days)
        data = df[df['timestamp'] > cutoff].copy()
        
        if not data.empty:
            p_colors = {'temperatura': '#FF3D00', 'pm2_5': '#D500F9', 'co': '#B0BEC5', 'no2': '#795548', 'o3': '#00E5FF'}
            curr_color = p_colors.get(param, '#FFF')
            chart = alt.Chart(data).mark_area(line={'color': curr_color}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color=curr_color, offset=1), alt.GradientStop(color='rgba(0,0,0,0)', offset=0)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('timestamp:T', title=None), y=alt.Y(f'{param}:Q', scale=alt.Scale(zero=False), title=param.upper()), tooltip=['timestamp', param]).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
    else: st.info("Esperando datos del recolector...")