''' main.py '''
import os
# Configuración para Wayland + IBus
os.environ["QT_QPA_PLATFORM"] = "wayland"  # Fuerza Wayland
os.environ["QT_IM_MODULE"] = "ibus"        # Habilita IBus
os.environ["GTK_IM_MODULE"] = "ibus"       # Para compatibilidad GTK
os.environ["XMODIFIERS"] = "@im=ibus"      # Soporte para aplicaciones legacy

from app import init

if __name__ == '__main__':
    import sys
    sys.exit(init.run())
