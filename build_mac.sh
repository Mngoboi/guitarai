#!/bin/bash
# Compila GuitarAI.app para Mac con PyInstaller (usa el venv del proyecto).
cd "$(dirname "$0")"
./venv/bin/pyinstaller --noconfirm --windowed --name GuitarAI \
  --collect-all demucs --collect-all whisper --collect-all librosa \
  --collect-all moviepy --collect-all imageio_ffmpeg --collect-all soundfile \
  --collect-all lazy_loader --collect-submodules sklearn \
  run_server.py
echo "Resultado en dist/GuitarAI.app"
