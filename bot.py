# -*- coding: utf-8 -*-
"""
Bot de Telegram para GuitarAI.
Flujo: el usuario manda un MP3 -> nombre y artista -> (foto opcional) -> recibe la carpeta .zip.
Token: variable de entorno GUITARAI_TOKEN  o  primer argumento al ejecutar.
Ejecutar:  ./venv/bin/python bot.py  <TOKEN>
"""
import os, sys, io, zipfile, asyncio, tempfile, traceback, functools
from urllib.parse import quote, urlparse, parse_qs
from telegram import Update
from telegram.ext import (Application, CommandHandler, MessageHandler, filters,
                          ContextTypes)
from motor import generar_cancion

TOKEN = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GUITARAI_TOKEN", "")).strip()
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "Salida CloneHero")
os.makedirs(OUT, exist_ok=True)
JOB_LOCK = asyncio.Lock()   # una canción a la vez

def _link_publico(folder):
    """Link de descarga directa para cuando el .zip no cabe en Telegram (>50 MB).
    Base/token desde env (GUITARAI_PUBLIC_URL/DASH_TOKEN) o enlace_publico.txt."""
    base = os.environ.get("GUITARAI_PUBLIC_URL", "").strip()
    key = os.environ.get("DASH_TOKEN", "").strip()
    try:
        txt = open(os.path.join(BASE, "enlace_publico.txt"), encoding="utf-8").read().strip()
        if txt:
            u = urlparse(txt)
            if not base and u.scheme: base = f"{u.scheme}://{u.netloc}"
            if not key: key = parse_qs(u.query).get("key", [""])[0]
    except Exception:
        pass
    if not base:
        return None
    link = f"{base}/descargar_chart/{quote(folder)}"
    if key: link += f"?key={quote(key)}"
    return link

# ---- Lista blanca (autorización por ID de Telegram) ----
ALLOW_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "allowed.txt")
def _read_allowed():
    ids, admins = set(), set()
    try:
        for line in open(ALLOW_FILE, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            uid = int(parts[0]); ids.add(uid)
            if "admin" in parts[1:]:
                admins.add(uid)
    except (FileNotFoundError, ValueError):
        pass
    return ids, admins
def is_allowed(uid):
    ids, _ = _read_allowed()
    return (not ids) or (uid in ids)   # archivo vacío/inexistente = abierto a todos
def is_admin(uid):
    return uid in _read_allowed()[1]
async def _guard(update):
    uid = update.effective_user.id
    if is_allowed(uid):
        return True
    await update.message.reply_text(
        f"🔒 No estás autorizado para usar este bot.\nTu ID de Telegram es: `{uid}`\n"
        "Pídele al dueño que te agregue.", parse_mode="Markdown")
    return False

WELCOME = (
    "🎸 *GuitarAI* — genero tu canción para Clone Hero.\n\n"
    "1) Mándame un *audio MP3*.\n"
    "2) Te pido *Nombre - Artista*.\n"
    "3) (Opcional) una *foto* para la carátula y el video.\n"
    "4) Te devuelvo la carpeta lista (.zip).\n\n"
    "⏱ Tarda varios minutos (usa IA). Manda el MP3 para empezar."
)

def _zip_folder(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, os.path.dirname(folder)))
    buf.seek(0); return buf

async def cmd_id(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    estado = "✅ autorizado" if is_allowed(uid) else "🔒 NO autorizado"
    await update.message.reply_text(f"Tu ID de Telegram es: `{uid}`\n({estado})", parse_mode="Markdown")

async def cmd_add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_admin(uid):
        await update.message.reply_text("Solo el dueño puede agregar usuarios.")
        return
    if not ctx.args or not ctx.args[0].lstrip("-").isdigit():
        await update.message.reply_text("Uso: /add <id>")
        return
    nuevo = int(ctx.args[0])
    with open(ALLOW_FILE, "a", encoding="utf-8") as f:
        f.write(f"{nuevo}\n")
    await update.message.reply_text(f"✅ Agregado el ID {nuevo} a la lista blanca.")

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update): return
    ctx.user_data.clear(); ctx.user_data["step"] = "audio"
    await update.message.reply_markdown(WELCOME)

async def on_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update): return
    a = update.message.audio or update.message.document
    if not a:
        return
    if a.file_size and a.file_size > 20*1024*1024:
        await update.message.reply_text("⚠️ El MP3 pesa más de 20 MB y Telegram no me deja descargarlo. Mándame uno más liviano.")
        return
    d = tempfile.mkdtemp()
    mp3 = os.path.join(d, "audio.mp3")
    f = await a.get_file(); await f.download_to_drive(mp3)
    ctx.user_data["mp3"] = mp3; ctx.user_data["step"] = "meta"
    await update.message.reply_text("✅ Audio recibido.\nAhora escribe: *nombre cancion - nombre artista*", parse_mode="Markdown")

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update): return
    step = ctx.user_data.get("step")
    txt = (update.message.text or "").strip()
    if step == "meta":
        if "-" in txt:
            name, artist = [p.strip() for p in txt.split("-", 1)]
        else:
            name, artist = txt, "Desconocido"
        ctx.user_data["name"] = name or "Mi Cancion"
        ctx.user_data["artist"] = artist or "Desconocido"
        ctx.user_data["step"] = "photo"
        await update.message.reply_text("📷 Mándame una *foto* para la carátula y el video, o escribe *saltar*.", parse_mode="Markdown")
    elif step == "photo" and txt.lower() in ("saltar", "skip", "no"):
        await _generar(update, ctx, None)
    else:
        await update.message.reply_text("Manda /start para empezar y envíame un MP3.")

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _guard(update): return
    if ctx.user_data.get("step") != "photo":
        await update.message.reply_text("Primero mándame un MP3 con /start.")
        return
    ph = update.message.photo[-1]
    d = tempfile.mkdtemp(); photo = os.path.join(d, "foto.jpg")
    f = await ph.get_file(); await f.download_to_drive(photo)
    await _generar(update, ctx, photo)

async def _generar(update: Update, ctx: ContextTypes.DEFAULT_TYPE, photo):
    mp3 = ctx.user_data.get("mp3")
    if not mp3:
        await update.message.reply_text("No tengo el audio. Manda /start de nuevo.")
        return
    name = ctx.user_data.get("name", "Mi Cancion"); artist = ctx.user_data.get("artist", "Desconocido")
    status = await update.message.reply_text("⏳ En cola...")
    progress = {"pct": 0, "msg": "En cola..."}
    def cb(msg, pct): progress.update(pct=pct, msg=msg)
    loop = asyncio.get_event_loop()

    async def poll():
        last = None
        while True:
            await asyncio.sleep(5)
            line = f"⏳ {progress['pct']}% — {progress['msg']}"
            if line != last:
                last = line
                try: await status.edit_text(line)
                except Exception: pass

    async with JOB_LOCK:
        poller = asyncio.create_task(poll())
        try:
            # OJO: generar_cancion(name, artist, ...) — el mp3 va como mp3_path (NO posicional).
            # make_video=False: el bot no recibe fuente de video; el video se agrega desde la web.
            job = functools.partial(generar_cancion, name, artist,
                                    mp3_path=mp3, photo_path=photo, out_root=OUT,
                                    make_lyrics=True, progress=cb)
            res = await loop.run_in_executor(None, job)
        except Exception as e:
            poller.cancel()
            await status.edit_text("❌ Error: " + str(e))
            traceback.print_exc(); return
        poller.cancel()

    await status.edit_text("📦 Empaquetando...")
    folder = res["folder"]
    buf = _zip_folder(folder)
    size_mb = buf.getbuffer().nbytes / 1048576
    n = res.get("notas", {})
    caption = (f"✅ {name} — {artist}\n"
               f"BPM {res.get('bpm','?')} · {res.get('dur','?')}s · {res.get('palabras',0)} palabras\n"
               f"Notas — Exp {n.get('ExpertSingle',0)} · Hard {n.get('HardSingle',0)} · "
               f"Med {n.get('MediumSingle',0)} · Easy {n.get('EasySingle',0)}\n"
               f"Descomprime y copia la carpeta a Clone Hero/songs/")
    link = _link_publico(os.path.basename(folder))
    LIMITE_MB = 48   # margen bajo el tope real de 50 MB del bot de Telegram
    if size_mb > LIMITE_MB:
        # demasiado grande para Telegram -> mandar el LINK de descarga directa (no fallar callado)
        extra = (f"\n\n📦 El .zip pesa {size_mb:.0f} MB y supera el límite de Telegram (50 MB).\n"
                 + (f"Descárgalo directo aquí 👇\n{link}" if link
                    else "Está guardado: ábrelo desde la página de GuitarAI → 📁 Mis charts."))
        await status.edit_text("✅ Listo (archivo grande, te paso el link)")
        await update.message.reply_text(caption + extra, disable_web_page_preview=True)
    else:
        try:
            await status.edit_text("📦 Enviando...")
            await update.message.reply_document(document=buf,
                                                filename=os.path.basename(folder)+".zip", caption=caption)
        except Exception as e:
            # el envío falló (tamaño/red) -> caer al link, nunca quedar callado
            traceback.print_exc()
            extra = (f"\n\n⚠️ No pude enviarte el .zip por Telegram ({e}).\n"
                     + (f"Descárgalo aquí 👇\n{link}" if link
                        else "Está guardado: ábrelo desde la página de GuitarAI → 📁 Mis charts."))
            await update.message.reply_text(caption + extra, disable_web_page_preview=True)
    ctx.user_data.clear(); ctx.user_data["step"] = "audio"

def main():
    if not TOKEN:
        print("Falta el token. Uso: ./venv/bin/python bot.py <TOKEN>  (o variable GUITARAI_TOKEN)"); sys.exit(1)
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ayuda", start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("add", cmd_add))
    app.add_handler(MessageHandler(filters.AUDIO | filters.Document.AUDIO, on_audio))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    print("🤖 GuitarAI bot corriendo. Ctrl+C para detener.")
    app.run_polling()

if __name__ == "__main__":
    main()
