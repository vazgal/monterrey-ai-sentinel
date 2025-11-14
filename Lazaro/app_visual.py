import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import time
import folium
from streamlit_folium import st_folium
from tensorflow.keras.models import load_model
import joblib
import os
import altair as alt

# --- 0. SILENCIADOR DE ADVERTENCIAS ---
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

# --- 1. CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="Monterrey AI Sentinel", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .css-card {
        background-color: rgba(38, 39, 48, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    h1, h2, h3 { color: #FFF !important; }
    .big-metric { font-size: 26px; font-weight: bold; color: #FFF; }
    .metric-label { font-size: 10px; text-transform: uppercase; color: #bbb; letter-spacing: 1px;}
    
    /* Estilo para la caja de sugerencias en el sidebar */
    .suggestion-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATOS Y UBICACIONES ---
LOCATIONS = {
    "Monterrey (Centro)": [25.6866, -100.3161],
    "San Pedro": [25.6576, -100.4029],
    "Apodaca": [25.7818, -100.1906],
    "Guadalupe": [25.6774, -100.2597],
    "San Nicolás": [25.7413, -100.2953],
    "Santa Catarina": [25.6768, -100.4627],
    "Escobedo": [25.7969, -100.3260]
}

# --- 3. FUNCIONES DE BACKEND ---
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
        
        temp = w['main']['temp']
        hum = w['main']['humidity']
        aqi = a['list'][0]['main']['aqi']
        comps = a['list'][0]['components']
        
        return temp, hum, aqi, comps
    except:
        return None, None, None, None

def predict_temp(model, scaler, current_temp):
    tiempo = np.arange(24)
    input_fake = (10 * np.sin(tiempo * 0.02) + 15) + np.random.normal(0, 1.5, 24)
    input_fake[-1] = current_temp 
    input_scaled = scaler.transform(input_fake.reshape(-1, 1))
    return scaler.inverse_transform(model.predict(input_scaled.reshape(1, 24, 1)))[0][0]

def load_historical_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "historial_clima.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cols_to_numeric = ['temperatura', 'pm2_5', 'co', 'no2', 'o3']
            for col in cols_to_numeric:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except: return pd.DataFrame()
    return pd.DataFrame()

def get_protocols(temp, aqi):
    """Genera la lista de sugerencias para el Sidebar"""
    protocols = []
    status = "normal" # normal, warning, danger
    
    # Temperatura
    if temp >= 38:
        protocols.append("🔥 **GOLPE DE CALOR:** Hidratación cada 30 min.")
        protocols.append("🛑 **PARO TÉCNICO:** Reducir labores al sol.")
        status = "danger"
    elif temp >= 32:
        protocols.append("☀️ **ALTA TEMPERATURA:** Usar bloqueador y gorra.")
        if status != "danger": status = "warning"
    elif temp < 12:
        protocols.append("❄️ **BAJAS TEMPERATURAS:** Usar ropa térmica.")
        protocols.append("☕ **BEBIDAS CALIENTES:** Disponibles para personal.")
        status = "info"
        
    # Aire
    if aqi >= 4:
        protocols.append("☣️ **AIRE TÓXICO:** Uso obligatorio de N95.")
        protocols.append("🏠 **CONFINAMIENTO:** Cerrar ventanas/puertas.")
        status = "danger"
    elif aqi == 3:
        protocols.append("😷 **AIRE MODERADO:** Reducir esfuerzo físico.")
        if status != "danger": status = "warning"
        
    if not protocols:
        protocols.append("✅ **OPERACIÓN NORMAL:** Sin restricciones.")
        protocols.append("📋 **RUTINA:** Monitoreo estándar c/ hora.")
        
    return protocols, status

# --- 4. GESTIÓN DE API KEY (INVISIBLE) ---
if "OPENWEATHER_KEY" in st.secrets:
    api_key = st.secrets["OPENWEATHER_KEY"]
else:
    # Fallback para pruebas locales
    api_key = "352a68073cee67874db9a5892b8d1d8a"

# --- 5. INTERFAZ DE USUARIO ---

# --- SIDEBAR LIMPIO ---
with st.sidebar:
    st.title("🛡️ AI Sentinel")
    st.caption("Sistema de Comando")
    
    # Selector principal
    selected_city = st.selectbox("📍 Ubicación Activa:", list(LOCATIONS.keys()))
    
    st.divider()
    
    # ESPACIO RESERVADO PARA PROTOCOLOS (Se llenará después de obtener datos)
    st.subheader("📢 Protocolos Activos")
    protocol_container = st.container()
    
    st.divider()
    live_mode = st.toggle("Modo EN VIVO", value=False)
    if live_mode:
        st.toast("📡 Actualizando...", icon="🔄")
        time.sleep(60)
        st.rerun()

# Tabs
tab1, tab2 = st.tabs(["📡 Monitor Tiempo Real", "🔬 Laboratorio de Datos"])

# --- LÓGICA PRINCIPAL ---
model, scaler = load_ai_resources()

if model:
    # Obtener datos de la ciudad seleccionada
    slat, slon = LOCATIONS[selected_city]
    temp, hum, aqi, comps = get_live_data(slat, slon, api_key)
    
    # --- ACTUALIZAR SIDEBAR CON DATOS REALES ---
    if temp:
        recos, status = get_protocols(temp, aqi)
        with protocol_container:
            if status == "danger":
                st.error("\n\n".join(recos))
            elif status == "warning":
                st.warning("\n\n".join(recos))
            elif status == "info":
                st.info("\n\n".join(recos))
            else:
                st.success("\n\n".join(recos))
    else:
        with protocol_container:
            st.warning("Esperando datos...")
            
    # --- TAB 1: DASHBOARD ---
    with tab1:
        if temp:
            pred = predict_temp(model, scaler, temp)
            error = abs(temp - pred)
            
            k1, k2, k3, k4 = st.columns(4)
            aqi_lbl = ["Bueno", "Aceptable", "Moderado", "Malo", "Peligroso"][aqi-1] if 1<=aqi<=5 else "N/A"
            aqi_col = "#4CAF50" if aqi <=2 else ("#FFC107" if aqi==3 else "#FF5252")
            
            with k1: st.markdown(f"""<div class="css-card"><div class="metric-label">🌡️ {selected_city}</div><div class="big-metric">{temp:.1f}°C</div></div>""", unsafe_allow_html=True)
            with k2: st.markdown(f"""<div class="css-card"><div class="metric-label">🤖 IA Predictiva</div><div class="big-metric" style="color:#29B6F6">{pred:.1f}°C</div></div>""", unsafe_allow_html=True)
            with k3: st.markdown(f"""<div class="css-card"><div class="metric-label">📉 Desviación</div><div class="big-metric">{error:.1f}°C</div></div>""", unsafe_allow_html=True)
            with k4: st.markdown(f"""<div class="css-card"><div class="metric-label">🌫️ Calidad Aire</div><div class="big-metric" style="color:{aqi_col}">{aqi_lbl}</div></div>""", unsafe_allow_html=True)

            # Mapa
            st.markdown("### 🗺️ Radar Metropolitano")
            m = folium.Map(location=[25.72, -100.30], zoom_start=11, tiles="CartoDB dark_matter")
            
            for city, coords in LOCATIONS.items():
                ct, _, caqi, _ = get_live_data(coords[0], coords[1], api_key)
                if ct:
                    color = "green"
                    if ct >= 38: color = "red"
                    elif ct < 12: color = "#29B6F6"
                    elif caqi >= 4: color = "purple"
                    elif caqi == 3: color = "orange"
                    
                    # Radio dinámico: Si hay peligro, el círculo es más grande
                    rad = 3000 if (color in ["red", "purple"]) else 1500
                    
                    folium.Circle(location=coords, radius=rad, color=color, fill=True, fill_opacity=0.5, tooltip=f"{city}: {ct}°C").add_to(m)
            
            st_folium(m, width="100%", height=450)

# --- TAB 2: ANÁLISIS ---
with tab2:
    st.markdown("### 📊 Tendencias Ambientales")
    df = load_historical_csv()
    
    if df.empty:
        st.warning("⚠️ Sin datos históricos. Ejecuta 'recolector.py'.")
    else:
        c_filter, c_days = st.columns([1, 1])
        with c_days: days = st.slider("Días atrás:", 1, 30, 3)
        
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        last_data = df[df['timestamp'] > cutoff_date]
        
        if last_data.empty:
            st.info("No hay datos recientes.")
        else:
            cols = [c for c in ['temperatura', 'pm2_5', 'co', 'no2', 'o3'] if c in last_data.columns]
            with c_filter: param = st.selectbox("Parámetro:", cols, index=0)
            
            p_conf = {
                'temperatura': {'c': '#FF5252', 't': 'Temp (°C)', 'f': False},
                'pm2_5': {'c': '#E040FB', 't': 'PM2.5', 'f': True},
                'co': {'c': '#9E9E9E', 't': 'CO (Tráfico)', 'f': True},
                'no2': {'c': '#795548', 't': 'NO2 (Ind)', 'f': True},
                'o3': {'c': '#29B6F6', 't': 'Ozono', 'f': True}
            }
            cfg = p_conf.get(param, {'c':'white','t':param,'f':False})
            
            chart_data = last_data[['timestamp', param]].dropna()
            
            # Escala segura
            ymin, ymax = chart_data[param].min(), chart_data[param].max()
            dom = [ymin*0.9, ymax*1.1] if not pd.isna(ymin) else [0, 100]

            base = alt.Chart(chart_data).encode(
                x=alt.X('timestamp:T', title="Hora"),
                y=alt.Y(f'{param}:Q', scale=alt.Scale(domain=dom), title=cfg['t']),
                tooltip=['timestamp', param]
            )
            
            if cfg['f']:
                ch = base.mark_area(line={'color':cfg['c']}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color=cfg['c'], offset=1), alt.GradientStop(color='rgba(0,0,0,0)', offset=0)])).properties(height=350).interactive()
            else:
                ch = base.mark_line(color=cfg['c'], point=True).properties(height=350).interactive()
            
            st.altair_chart(ch, use_container_width=True)
            
            with st.expander("Ver Datos"): st.dataframe(last_data.sort_values("timestamp", ascending=False))