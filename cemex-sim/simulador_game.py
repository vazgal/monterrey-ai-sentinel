from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.prefabs.window_panel import WindowPanel
import random
import math

# --- 1. CONFIGURACIÓN ---
app = Ursina()
window.title = "Digital Twin: Physics & Collision"
window.borderless = False
window.color = color.rgb(135, 206, 235)

# --- 2. VARIABLES ---
sim_state = {
    "production": 50,
    "rain": 0,
    "wind_speed": 10,
    "wind_dir": 90,
    "filter_health": 100
}
frame_count = 0 

# --- 3. AMBIENTE ---
DirectionalLight(parent=scene, y=50, z=50, shadows=True, rotation=(45, -45, 45))
AmbientLight(color=color.rgba(100, 100, 100, 0.5))
Sky()

# Suelo
ground = Entity(
    model='plane', scale=(500, 1, 500), 
    color=color.rgb(40, 40, 45), texture='white_cube', texture_scale=(100,100), 
    collider='box'
)

# --- 4. TU MODELO 3D (SÓLIDO Y ELEVADO) ---
try:
    factory = Entity(
        model='the_factory.glb',
        scale=1,
        # AJUSTE DE ALTURA: Si sigue enterrado, aumenta el segundo número (Y)
        position=(0, 16, 0), 
        rotation=(0, 0, 0),
        collider='mesh' # <--- ¡PAREDES SÓLIDAS ACTIVADAS!
    )
except:
    print("⚠️ Error cargando modelo. Usando cubo.")
    factory = Entity(model='cube', scale=(20,10,20), color=color.gray, position=(0,5,0), collider='box')

# --- 5. OBJETOS MOVIBLES ---

# A. EMISOR HUMO (ROJO) - Control con I, J, K, L
smoke_emitter = Entity(
    model='cube', color=color.red, scale=(0.5, 0.5, 0.5), visible=True,
    position=(0, 15, 0) 
)

# B. SENSOR VECINO (VERDE) - Control con FLECHAS
neighborhood_sensor = Entity(
    model='cube', 
    color=color.rgba(0, 255, 0, 50), # Verde transparente
    scale=(20, 10, 20),              
    position=(30, 5, 0),             
    collider='box' # Para detectar humo, no al jugador
)

# --- 6. FÍSICA ---
particles = []

def create_pollution(pos, intensity):
    p = Entity(model='sphere', position=pos, scale=random.uniform(0.5, 1.5),
               color=color.rgba(50, 50, 50, 150)) 
    p.velocity = Vec3(0, 1 + (intensity/10), 0)
    p.life = 600
    particles.append(p)

def update():
    global frame_count
    frame_count += 1
    
    # --- CONTROLES DE MOVIMIENTO DE OBJETOS ---
    spd = 20 * time.dt
    
    # 1. Mover Chimenea (Rojo) -> I, K, J, L, U, O
    if held_keys['i']: smoke_emitter.z += spd
    if held_keys['k']: smoke_emitter.z -= spd
    if held_keys['j']: smoke_emitter.x -= spd
    if held_keys['l']: smoke_emitter.x += spd
    if held_keys['u']: smoke_emitter.y += spd
    if held_keys['o']: smoke_emitter.y -= spd

    # 2. Mover Vecindario (Verde) -> FLECHAS
    if held_keys['up arrow']: neighborhood_sensor.z += spd
    if held_keys['down arrow']: neighborhood_sensor.z -= spd
    if held_keys['left arrow']: neighborhood_sensor.x -= spd
    if held_keys['right arrow']: neighborhood_sensor.x += spd
    if held_keys['page up']: neighborhood_sensor.y += spd
    if held_keys['page down']: neighborhood_sensor.y -= spd

    # 3. Ajuste fino de altura de Fábrica (Si sigue enterrada) -> N, M
    if held_keys['n']: factory.y += spd * 1
    if held_keys['m']: factory.y -= spd * 1

    # --- FÍSICA AMBIENTAL ---
    emission_factor = (sim_state['production'] / 50) * (2 - (sim_state['filter_health']/100))
    spawn_rate = max(1, int(10 - emission_factor * 3))
    
    if frame_count % spawn_rate == 0:
        offset = Vec3(random.uniform(-0.2,0.2), 0, random.uniform(-0.2,0.2))
        create_pollution(smoke_emitter.world_position + offset, emission_factor)

    rad = math.radians(sim_state['wind_dir'])
    wind_vec = Vec3(math.cos(rad), 0, math.sin(rad)) * (sim_state['wind_speed'] * 0.05)

    impact_count = 0
    for p in particles:
        p.position += p.velocity * time.dt
        p.position += wind_vec
        p.scale += Vec3(0.02, 0.02, 0.02)
        p.alpha -= 0.002
        
        if sim_state['rain'] > 0:
            p.velocity.y -= 0.05 
            p.color = color.rgba(100, 90, 70, 150)
        
        if p.intersects(neighborhood_sensor).hit:
            impact_count += 1
            p.color = color.rgba(255, 0, 0, 200)

        if p.alpha <= 0 or p.y < 0:
            destroy(p)
            particles.remove(p)

    # ALERTA VISUAL
    if impact_count > 0:
        neighborhood_sensor.color = color.rgba(255, 0, 0, 100)
        alert_text.text = "⚠️ IMPACTO EN VECINDARIO"
        alert_text.color = color.red
    else:
        neighborhood_sensor.color = color.rgba(0, 255, 0, 50)
        alert_text.text = "SISTEMA SEGURO"
        alert_text.color = color.green
        
    info_text.text = f"Pos Fábrica Y: {factory.y:.2f} | Chimenea: {smoke_emitter.position}"

# --- UI & JUGADOR ---
alert_text = Text(text="INICIANDO...", position=(0, 0.45), origin=(0,0), scale=2, background=True)
info_text = Text(text="Datos", position=(0, 0.40), origin=(0,0), scale=1)

instructions = Text(
    text="[WASD] Mover Jugador | [F] Volar/Caminar\n[FLECHAS] Mover Vecindario (Verde)\n[I,J,K,L] Mover Humo (Rojo)\n[N / M] Subir/Bajar Fábrica",
    position=(-0.85, -0.40), origin=(0,0), scale=0.8
)

# Sliders
sl_prod = Slider(text='Producción', min=0, max=120, default=50, dynamic=True)
sl_wind_s = Slider(text='Viento Vel', min=0, max=50, default=10, dynamic=True)
sl_wind_d = Slider(text='Viento Dir', min=0, max=360, default=90, dynamic=True)
sl_filt = Slider(text='Filtros', min=0, max=100, default=100, dynamic=True)
sl_rain = Slider(text='Lluvia', min=0, max=100, default=0, dynamic=True)

def update_p():
    sim_state['production'] = sl_prod.value
    sim_state['wind_speed'] = sl_wind_s.value
    sim_state['wind_dir'] = sl_wind_d.value
    sim_state['filter_health'] = sl_filt.value
    sim_state['rain'] = sl_rain.value

for s in [sl_prod, sl_wind_s, sl_wind_d, sl_filt, sl_rain]: s.on_value_changed = update_p

menu = WindowPanel(
    title='CONTROLES (ESC)',
    content=(sl_prod, sl_filt, Space(), sl_wind_s, sl_wind_d, sl_rain),
    position=(-0.7, 0.2), enabled=False
)

# JUGADOR (Altura ajustada y gravedad activada)
player = FirstPersonController(speed=15, y=10) 
player.cursor.visible = False
player.gravity = 1 # Empezamos caminando

def input(key):
    # Menú
    if key == 'escape':
        if menu.enabled:
            menu.enabled = False
            player.enabled = True
            mouse.locked = True
        else:
            menu.enabled = True
            player.enabled = False
            mouse.locked = False
            
    # Modo Vuelo / Caminar
    if key == 'f' and not menu.enabled:
        if player.gravity > 0:
            player.gravity = 0 # Vuelo
            print("Modo Vuelo Activado")
        else:
            player.gravity = 1 # Caminar
            print("Modo Caminar Activado")
            
    # Controles de vuelo (Subir/Bajar)
    if player.gravity == 0:
        if held_keys['space']: player.y += 10 * time.dt
        if held_keys['shift']: player.y -= 10 * time.dt

app.run()