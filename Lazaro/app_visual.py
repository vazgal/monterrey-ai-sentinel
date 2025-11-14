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

# --- 1. CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="Monterrey AI Lab", page_icon="🧪", layout="wide")

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
    iframe { width: 100% !important; }
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
        # Clima
        w = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric").json()
        # Contaminación (AQI Completo)
        a = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}").json()
        
        temp = w['main']['temp']
        hum = w['main']['humidity']
        aqi = a['list'][0]['main']['aqi']
        comps = a['list'][0]['components'] # Diccionario con co, no2, o3, pm2_5
        
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
            # Asegurar que las columnas sean numéricas
            cols_to_numeric = ['temperatura', 'pm2_5', 'co', 'no2', 'o3']
            for col in cols_to_numeric:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            return pd.DataFrame()
    return pd.DataFrame()

# --- 4. INTERFAZ DE USUARIO ---

# Sidebar
with st.sidebar:
    st.title("🧪 Monterrey AI Lab")
    st.caption("Monitoreo Ambiental Avanzado")
    api_key = st.text_input("API Key", value="352a68073cee67874db9a5892b8d1d8a", type="password")
    st.divider()
    live_mode = st.toggle("Modo EN VIVO (Auto-Refresh)", value=False)
    if live_mode:
        st.toast("📡 Sincronizando datos...")
        time.sleep(60)
        st.rerun()

# Tabs
tab1, tab2 = st.tabs(["📡 Monitor Tiempo Real", "🔬 Laboratorio de Datos"])

# --- TAB 1: DASHBOARD PRINCIPAL ---
with tab1:
    col_sel, col_kpi = st.columns([1, 3])
    with col_sel:
        selected_city = st.selectbox("📍 Seleccionar Sector:", list(LOCATIONS.keys()))
    
    model, scaler = load_ai_resources()
    
    if model:
        slat, slon = LOCATIONS[selected_city]
        temp, hum, aqi, comps = get_live_data(slat, slon, api_key)
        
        if temp:
            pred = predict_temp(model, scaler, temp)
            error = abs(temp - pred)
            pm25 = comps.get('pm2_5', 0)
            
            k1, k2, k3, k4 = st.columns(4)
            aqi_lbl = ["Bueno", "Aceptable", "Moderado", "Malo", "Peligroso"][aqi-1] if 1<=aqi<=5 else "N/A"
            aqi_col = "#4CAF50" if aqi <=2 else ("#FFC107" if aqi==3 else "#FF5252")
            
            with k1: st.markdown(f"""<div class="css-card"><div class="metric-label">🌡️ {selected_city}</div><div class="big-metric">{temp:.1f}°C</div></div>""", unsafe_allow_html=True)
            with k2: st.markdown(f"""<div class="css-card"><div class="metric-label">🤖 IA Predictiva</div><div class="big-metric" style="color:#29B6F6">{pred:.1f}°C</div></div>""", unsafe_allow_html=True)
            with k3: st.markdown(f"""<div class="css-card"><div class="metric-label">📉 Desviación</div><div class="big-metric">{error:.1f}°C</div></div>""", unsafe_allow_html=True)
            with k4: st.markdown(f"""<div class="css-card"><div class="metric-label">🌫️ Calidad Aire</div><div class="big-metric" style="color:{aqi_col}">{aqi_lbl}</div></div>""", unsafe_allow_html=True)

            # --- SECCIÓN NUEVA: DETALLES DE GASES ---
            with st.expander(f"🔬 Ver desglose de contaminantes en {selected_city}"):
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("PM2.5 (Partículas)", f"{comps.get('pm2_5',0)}", "μg/m³")
                g2.metric("CO (Monóxido/Autos)", f"{comps.get('co',0)}", "μg/m³")
                g3.metric("NO2 (Nitrógeno/Ind)", f"{comps.get('no2',0)}", "μg/m³")
                g4.metric("O3 (Ozono/Sol)", f"{comps.get('o3',0)}", "μg/m³")

            # Mapa
            st.markdown("### 🗺️ Radar Metropolitano")
            m = folium.Map(location=[25.72, -100.30], zoom_start=11, tiles="CartoDB dark_matter")
            
            for city, coords in LOCATIONS.items():
                ct, _, caqi, _ = get_live_data(coords[0], coords[1], api_key)
                if ct:
                    color = "green"
                    if ct >= 38: color = "red"
                    elif ct < 12: color = "blue"
                    elif caqi >= 4: color = "purple"
                    elif caqi == 3: color = "orange"
                    
                    folium.Circle(location=coords, radius=1800, color=color, fill=True, fill_opacity=0.5, tooltip=f"{city}: {ct}°C").add_to(m)
                    folium.Marker(location=coords, icon=folium.DivIcon(html=f'<div style="color:white;font-size:9pt;text-shadow:0 0 3px #000">{city}</div>')).add_to(m)
            
            st_folium(m, width="100%", height=450)

# --- TAB 2: ANÁLISIS DE DATOS (ACTUALIZADO MULTI-GAS) ---
with tab2:
    st.markdown("### 📊 Análisis de Tendencias Ambientales")
    
    df = load_historical_csv()
    
    if df.empty:
        st.warning("⚠️ No hay datos válidos. Ejecuta 'recolector.py'.")
    else:
        # Filtros de Tiempo
        c_filter, c_days = st.columns([1, 1])
        with c_days:
            days = st.slider("Historial (Días):", 1, 30, 3)
        
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        last_data = df[df['timestamp'] > cutoff_date]
        
        if last_data.empty:
            st.info(f"No hay datos en los últimos {days} días.")
        else:
            # --- SELECTOR DE PARAMETRO ---
            available_columns = [col for col in ['temperatura', 'pm2_5', 'co', 'no2', 'o3'] if col in last_data.columns]
            
            # Diccionario de configuración visual para cada gas
            param_config = {
                'temperatura': {'color': '#FF5252', 'title': 'Temperatura (°C)', 'fill': False},
                'pm2_5':       {'color': '#E040FB', 'title': 'PM2.5 (Partículas Finas)', 'fill': True},
                'co':          {'color': '#9E9E9E', 'title': 'CO (Tráfico Vehicular)', 'fill': True},
                'no2':         {'color': '#795548', 'title': 'NO2 (Actividad Industrial)', 'fill': True},
                'o3':          {'color': '#29B6F6', 'title': 'O3 (Radiación/Ozono)', 'fill': True}
            }

            with c_filter:
                param_selected = st.selectbox("🔍 Seleccionar Parámetro a Analizar:", available_columns, index=0)

            # Configuración del gráfico seleccionado
            config = param_config.get(param_selected, {'color': 'white', 'title': param_selected, 'fill': False})
            
            # Limpiar datos para ese parámetro específico
            chart_data = last_data[['timestamp', param_selected]].dropna()
            
            if chart_data.empty:
                st.warning(f"No hay datos registrados para {param_selected} aún.")
            else:
                # Calcular dominio Y dinámico
                y_min = chart_data[param_selected].min()
                y_max = chart_data[param_selected].max()
                y_domain = [y_min * 0.9, y_max * 1.1] # Darle un poco de aire

                st.subheader(f"Tendencia: {config['title']}")
                
                base = alt.Chart(chart_data).encode(
                    x=alt.X('timestamp:T', title="Tiempo"),
                    y=alt.Y(f'{param_selected}:Q', scale=alt.Scale(domain=y_domain), title=config['title']),
                    tooltip=['timestamp', f'{param_selected}']
                )

                if config['fill']:
                    # Gráfico de Área (para gases)
                    chart = base.mark_area(
                        line={'color': config['color']},
                        color=alt.Gradient(
                            gradient='linear',
                            stops=[alt.GradientStop(color=config['color'], offset=1),
                                   alt.GradientStop(color='rgba(0,0,0,0)', offset=0)],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    ).properties(height=350).interactive()
                else:
                    # Gráfico de Línea (para temperatura)
                    chart = base.mark_line(color=config['color'], point=True).properties(height=350).interactive()

                st.altair_chart(chart, use_container_width=True)
            
            st.divider()
            with st.expander("📥 Descargar Datos Crudos"):
                st.dataframe(last_data.sort_values(by="timestamp", ascending=False))