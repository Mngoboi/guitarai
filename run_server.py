# -*- coding: utf-8 -*-
"""Punto de entrada para el ejecutable empaquetado (Mac/Windows).
Arranca el dashboard y abre el navegador. Robusto para PyInstaller.
Auto-update: al abrir con internet baja la última versión de app.py + motor.py
desde GitHub y la usa; si no hay internet, usa la versión empaquetada."""
import threading, time, socket, webbrowser, os, sys, urllib.request

REPO = "Mngoboi/guitarai"          # repo donde vive el código (público)
UPDATE_FILES = ["app.py", "motor.py"]

def self_update():
    """Descarga el último código y lo deja en una carpeta local que tiene prioridad."""
    try:
        cache = os.path.join(os.path.expanduser("~"), ".guitarai_update")
        os.makedirs(cache, exist_ok=True)
        fetched = {}
        for f in UPDATE_FILES:
            url = f"https://raw.githubusercontent.com/{REPO}/main/{f}"
            data = urllib.request.urlopen(url, timeout=8).read()
            if not data or len(data) < 200:
                return False
            fetched[f] = data
        # solo si bajaron TODOS (evita versiones mezcladas)
        for f, data in fetched.items():
            with open(os.path.join(cache, f), "wb") as fh:
                fh.write(data)
        sys.path.insert(0, cache)
        print("GuitarAI: actualizado desde GitHub")
        return True
    except Exception as e:
        print("GuitarAI: sin actualización (uso versión local):", e)
        return False

self_update()
try:
    from app import app, DASH_TOKEN
except Exception:
    from app import app; DASH_TOKEN = ""

def _free_port(default=5050):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", default)); s.close(); return default
    except OSError:
        s2 = socket.socket(); s2.bind(("127.0.0.1", 0)); p = s2.getsockname()[1]; s2.close(); return p

def main():
    port = _free_port()
    def open_browser():
        time.sleep(1.8)
        q = f"/?key={DASH_TOKEN}" if DASH_TOKEN else "/"
        webbrowser.open(f"http://127.0.0.1:{port}{q}")
    threading.Thread(target=open_browser, daemon=True).start()
    print(f"GuitarAI -> http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)

if __name__ == "__main__":
    main()
