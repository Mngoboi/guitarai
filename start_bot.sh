#!/bin/bash
cd "$(dirname "$0")"
# Esperar a que haya conectividad con Telegram antes de arrancar (evita fallo de DNS al boot)
for i in $(seq 1 120); do
  if /usr/bin/curl -s --max-time 5 "https://api.telegram.org" >/dev/null 2>&1; then
    echo "$(date '+%H:%M:%S') red OK, arrancando bot..."; break
  fi
  echo "$(date '+%H:%M:%S') esperando red... ($i)"; sleep 5
done
export GUITARAI_TOKEN="$(cat .token)"
exec ./venv/bin/python bot.py
