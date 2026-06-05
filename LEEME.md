# 🎸 GuitarAI — Generador de canciones para Clone Hero

Sube un **MP3**, una **foto**, escribe **nombre + artista** y genera una carpeta lista
para Clone Hero con: chart difícil (triples/dobles/simples + sustains + Star Power),
notas que siguen la **voz** cuando canta y la **instrumental** cuando no, **karaoke**
automático y un **video que late con el ritmo**.

## Archivos del proyecto
- `motor.py` — el motor (todo el pipeline). Se puede usar solo.
- `app.py` — dashboard **web** (localhost / dominio).
- `app_nativa.py` — **app nativa** (ventana propia Win/Mac).
- `venv/` — entorno con todas las dependencias.

## Cómo usarlo

### 1) Web local (recomendado)
```bash
cd "/Users/macbook/GuitarAI"
./venv/bin/python app.py
```
Abre en el navegador: **http://127.0.0.1:5050**

### 2) App nativa (ventana propia)
```bash
./venv/bin/python app_nativa.py
```

### 3) Desde terminal (sin interfaz)
```bash
./venv/bin/python motor.py "miaudio.mp3" "Nombre Cancion" "Artista" "foto.jpg"
```

La carpeta sale en `Salida CloneHero/<Artista> - <Cancion>/` con:
`notes.chart`, `song.ini`, `song.mp3`, `album.png`, `video.mp4`.
Cópiala a `Clone Hero/songs/`.

## Tiempos
Cada canción tarda **~6-10 min** (separar voz + transcribir letra + video).
Es normal: usa demucs (IA de separación) y Whisper (IA de transcripción) en CPU.

## La letra (karaoke)
Se transcribe automáticamente de la voz aislada. **No es perfecta** (es canto):
revisa/corrige las palabras si quieres. Los tiempos quedan bien.

## Pasar a Windows (.exe)
El `.exe` se compila **en Windows** (no desde Mac). Allá, con Python instalado:
```
pip install -r requisitos.txt
pip install pyinstaller
pyinstaller --noconfirm --windowed --name GuitarAI ^
  --collect-all demucs --collect-all whisper --collect-all librosa ^
  --collect-all moviepy --collect-all imageio_ffmpeg app_nativa.py
```
(En Mac se empaqueta igual pero con `:` en vez de `^`.)

## Pasar a un dominio (web pública) — más adelante
- Servir `app.py` con gunicorn detrás de nginx.
- Como cada canción tarda minutos, conviene una **cola de trabajos** (Redis + RQ/Celery)
  para no bloquear. Lo vemos cuando llegues a esa fase.

## Límites de Clone Hero (no de este programa)
- No hay efectos (fuegos/rayos) que reaccionen a tus notas: el formato no lo permite.
  El "efecto de acumulación" real es el **Star Power** (ya incluido).
- Si quieres luces/efectos reactivos de verdad, el juego **YARG** sí los soporta.
