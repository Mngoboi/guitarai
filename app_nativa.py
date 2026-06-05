# -*- coding: utf-8 -*-
"""Lanzador NATIVO (Windows/Mac): abre el dashboard GuitarAI en una ventana propia.
Arranca el servidor Flask en un hilo y lo muestra con pywebview."""
import threading, time, socket
import webview
from app import app

def _free_port(default=5050):
    s=socket.socket();
    try:
        s.bind(("127.0.0.1",default)); s.close(); return default
    except OSError:
        s2=socket.socket(); s2.bind(("127.0.0.1",0)); p=s2.getsockname()[1]; s2.close(); return p

def main():
    port=_free_port()
    threading.Thread(target=lambda: app.run(host="127.0.0.1",port=port,threaded=True,
                                            use_reloader=False), daemon=True).start()
    time.sleep(1.0)
    webview.create_window("GuitarAI — Generador Clone Hero",
                          f"http://127.0.0.1:{port}", width=620, height=820)
    webview.start()

if __name__=="__main__":
    main()
