from ursina import *

app = Ursina()

# Carga el modelo
try:
    print("📂 Cargando modelo...")
    # Cambia 'ciudad.glb' por el nombre real de tu archivo
    model = Entity(model='ciudad.glb') 
    
    print("-" * 30)
    print("✅ ¡MODELO CARGADO!")
    print("🎥 LISTA DE ANIMACIONES ENCONTRADAS:")
    
    # Imprimir todas las animaciones disponibles
    if hasattr(model, 'animations') and model.animations:
        for anim_name in model.animations:
            print(f"   ▶ '{anim_name}'")
    else:
        print("   ⚠️ No se detectaron animaciones o tienen un formato no estándar.")
        
    print("-" * 30)

except Exception as e:
    print(f"❌ Error: {e}")

# Cerramos la app inmediatamente, solo queríamos ver la consola
application.quit()