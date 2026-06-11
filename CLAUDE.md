# GuitarAI — guía local (El Pantano)

- **Qué es:** MP3 → canción completa de Clone Hero (chart + karaoke whisper + video).
- **Puerto:** 5050 · **LaunchAgent:** ai.guitarai.web/bot/ngrok/awake (legados)
- **Arranque local:** `./venv/bin/python app.py` (venv = **python3.11**, NO 3.14: torch 2.2.2 no tiene wheel para 3.14).
- **Recrear venv (receta probada 2026-06-09):** `python3.11 -m venv venv` → `pip install -r requisitos.txt` → `pip install python-telegram-bot` → aeneas: `pip install "setuptools==59.8.0"` y luego `AENEAS_WITH_CEW=False pip install --no-build-isolation aeneas` (su `numpy.distutils` choca con setuptools moderno). ~5-7 GB. requisitos.txt NO lista telegram ni aeneas.
- **Nota clave:** ⚠️ Expuesto a internet vía ngrok; acceso por tokens/allowed.txt. venv regenerado 2026-06-09 tras limpieza de disco (dist sigue borrado, regenerable).

## Reglas locales
- Manda la **Constitución global**: `~/.claude/CLAUDE.md` (aditivo, doble-check si el producto está "terminado", Mac+Win, .exe se pregunta).
- Entregables → `Dropbox/todo claude/` en su subcarpeta. El código fuente NO se mueve de aquí.
- Estado completo e historial → memoria del proyecto (`project_*.md`) y `~/.claude/registro-pantano.md`.
