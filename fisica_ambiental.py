import numpy as np
import math

def estimar_estabilidad_atmosferica(wind_speed, es_dia=True):
    """
    Estima la Clase de Estabilidad de Pasquill-Gifford (A-F).
    A = Muy inestable (Mucho sol, poco viento -> El humo sube recto)
    D = Neutral (Nublado o viento fuerte -> El humo se va de lado)
    F = Muy estable (Noche despejada, poco viento -> El humo se queda abajo)
    """
    # Simplificación para el MVP
    if wind_speed < 2: return "A" if es_dia else "F"
    if wind_speed < 5: return "B" if es_dia else "E"
    return "D" # Viento fuerte mezcla todo (Neutral)

def coeficientes_briggs(x, estabilidad):
    """
    Calcula sigma_y y sigma_z (dispersión) basado en la distancia (x)
    usando las fórmulas urbanas de Briggs.
    """
    # Coeficientes estándar para entorno Urbano
    if estabilidad in ["A", "B"]:
        sy = 0.32 * x * (1 + 0.0004 * x)**(-0.5)
        sz = 0.24 * x * (1 + 0.001 * x)**0.5
    elif estabilidad == "C":
        sy = 0.22 * x * (1 + 0.0004 * x)**(-0.5)
        sz = 0.20 * x
    elif estabilidad == "D":
        sy = 0.16 * x * (1 + 0.0004 * x)**(-0.5)
        sz = 0.14 * x * (1 + 0.0003 * x)**(-0.5)
    else: # E, F
        sy = 0.11 * x * (1 + 0.0004 * x)**(-0.5)
        sz = 0.08 * x * (1 + 0.0015 * x)**(-0.5)
    return sy, sz

def generar_pluma_toxica(lat_origen, lon_origen, viento_vel, viento_dir_grados, q_emision=1000):
    """
    Genera una nube de puntos (Heatmap Data) simulando la pluma de contaminación.
    Retorna: Lista de [lat, lon, intensidad]
    """
    pluma_data = []
    
    # 1. Configuración Física
    # Convertimos dirección meteorológica (de donde viene) a matemática (hacia donde va)
    # Viento del Norte (0°) sopla hacia el Sur (180°)
    dir_flujo_rad = math.radians((viento_dir_grados + 180) % 360)
    
    estabilidad = estimar_estabilidad_atmosferica(viento_vel)
    
    # Si no hay viento, asumimos una dispersión circular mínima para no dividir por cero
    u = max(viento_vel, 0.5) 

    # 2. Simulación (Iteramos distancia desde la fuente)
    # Simulamos 5 km a la redonda (5000 metros)
    for x in range(100, 5000, 100): # Cada 100 metros
        
        # Calcular dispersión a esta distancia
        sy, sz = coeficientes_briggs(x, estabilidad)
        
        # Ecuación Gaussiana (Eje central de la pluma, y=0)
        # Concentración en el centro
        C_center = (q_emision) / (2 * np.pi * u * sy * sz)
        
        # Si la concentración es muy baja, dejamos de calcular (la nube se disipó)
        if C_center < 0.1: break
        
        # 3. Generar ancho de la pluma (dispersión lateral)
        # Barremos desde el centro hacia los lados (eje Y local)
        ancho_efectivo = int(3 * sy) # 3 sigmas cubren 99% de la pluma
        paso_y = max(int(sy / 2), 10) # Resolución dinámica
        
        for y_local in range(-ancho_efectivo, ancho_efectivo, paso_y):
            
            # Concentración en este punto lateral
            C = C_center * np.exp( - (y_local**2) / (2 * sy**2) )
            
            if C < 0.5: continue # Filtrar puntos invisibles
            
            # 4. Rotación de Coordenadas (Geometría Analítica)
            # Convertimos (x, y_local) a (Latitud, Longitud)
            # x está alineado con el viento. y_local es perpendicular.
            
            # Rotar
            dx_rot = x * math.cos(dir_flujo_rad) - y_local * math.sin(dir_flujo_rad)
            dy_rot = x * math.sin(dir_flujo_rad) + y_local * math.cos(dir_flujo_rad)
            
            # Convertir metros a grados (Aprox para Monterrey: 111km/grado lat, 100km/grado lon)
            d_lat = dy_rot / 111132
            d_lon = dx_rot / (111320 * math.cos(math.radians(lat_origen)))
            
            new_lat = lat_origen + d_lat
            new_lon = lon_origen + d_lon
            
            # Normalizar intensidad para el mapa (0 a 1)
            intensidad = min(C / 50, 1.0) # Ajuste visual
            
            pluma_data.append([new_lat, new_lon, intensidad])
            
    return pluma_data
