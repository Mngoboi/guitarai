# -*- coding: utf-8 -*-
"""Punto de entrada para el ejecutable empaquetado (Mac/Windows).
Arranca el dashboard y abre el navegador. Robusto para PyInstaller."""
import threading, time, socket, webbrowser
from app import app

def _free_port(default=5050):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", default)); s.close(); return default
    except OSError:
        s2 = socket.socket(); s2.bind(("127.0.0.1", 0)); p = s2.getsockname()[1]; s2.close(); return p

def main():
    port = _free_port()
    def open_browser():
        time.sleep(1.8); webbrowser.open(f"http://127.0.0.1:{port}")
    threading.Thread(target=open_browser, daemon=True).start()
    print(f"GuitarAI -> http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)

if __name__ == "__main__":
    main()
