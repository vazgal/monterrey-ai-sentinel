import numpy as np
import math

def estimar_estabilidad_atmosferica(wind_speed, es_dia=True):
    """
    Estima la estabilidad (Pasquill-Gifford).
    """
    if wind_speed < 2: return "A" if es_dia else "F"
    if wind_speed < 5: return "B" if es_dia else "E"
    return "D"

def coeficientes_briggs(x, estabilidad):
    """
    Coeficientes de dispersión urbana (Briggs).
    """
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
    Genera la pluma Gaussiana con ALTA SENSIBILIDAD.
    """
    pluma_data = []
    
    # Validación de seguridad
    if viento_vel is None or viento_dir_grados is None:
        return []

    # Convertir dirección meteorológica a trigonométrica
    # (Viento DEL Norte sopla HACIA el Sur)
    dir_flujo_rad = math.radians((viento_dir_grados + 180) % 360)
    
    estabilidad = estimar_estabilidad_atmosferica(viento_vel)
    
    # Evitar división por cero si el viento es nulo (usamos 0.5 m/s mínimo)
    u = max(viento_vel, 0.5) 

    # Iterar distancia (hasta 10km para asegurar que se vea algo)
    for x in range(50, 10000, 100): 
        
        sy, sz = coeficientes_briggs(x, estabilidad)
        
        # Ecuación Gaussiana (Centro)
        C_center = (q_emision) / (2 * np.pi * u * sy * sz)
        
        # UMBRAL REDUCIDO: Antes era 0.1, ahora 0.001 para ver plumas lejanas
        if C_center < 0.001: break
        
        # Barrido lateral
        ancho_efectivo = int(3.5 * sy)
        paso_y = max(int(sy / 1.5), 10) # Mayor resolución
        
        for y_local in range(-ancho_efectivo, ancho_efectivo, paso_y):
            
            C = C_center * np.exp( - (y_local**2) / (2 * sy**2) )
            
            # UMBRAL DE DIBUJO: Si la concentración es mínima, ignora
            # Bajamos de 0.5 a 0.05 para que dibuje incluso humo ligero
            if C < 0.05: continue 
            
            # Rotación de Coordenadas
            dx_rot = x * math.cos(dir_flujo_rad) - y_local * math.sin(dir_flujo_rad)
            dy_rot = x * math.sin(dir_flujo_rad) + y_local * math.cos(dir_flujo_rad)
            
            # Conversión a Lat/Lon (Aprox Mty)
            d_lat = dy_rot / 111132
            d_lon = dx_rot / (111320 * math.cos(math.radians(lat_origen)))
            
            new_lat = lat_origen + d_lat
            new_lon = lon_origen + d_lon
            
            # Intensidad visual para el mapa (0.0 a 1.0)
            # Ajustamos la escala para que 50 sea el máximo visual
            intensidad = min(C / 20, 1.0) 
            
            pluma_data.append([new_lat, new_lon, intensidad])
            
    return pluma_data
