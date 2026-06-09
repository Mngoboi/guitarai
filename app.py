# -*- coding: utf-8 -*-
"""Dashboard web GuitarAI — sube MP3 + foto, escribe nombre/artista, genera la carpeta."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # evita "OMP Error #15" (choque OpenMP)
import threading, uuid, zipfile, io, traceback
from flask import Flask, request, jsonify, send_file, Response
from motor import generar_cancion

BASE=os.path.dirname(os.path.abspath(__file__))
UPLOAD=os.path.join(BASE,"_uploads"); OUT=os.path.join(BASE,"Salida CloneHero")
os.makedirs(UPLOAD,exist_ok=True); os.makedirs(OUT,exist_ok=True)
app=Flask(__name__)
JOBS={}   # id -> {pct,msg,done,error,result}

# ===== Acceso por token (principal + tokens únicos revocables en tokens.txt) =====
DASH_TOKEN=os.environ.get("DASH_TOKEN","clonehero-aurora-4127")
TOKENS_FILE=os.path.join(BASE,"tokens.txt")
def _valid_tokens():
    toks={DASH_TOKEN}
    try:
        for line in open(TOKENS_FILE,encoding="utf-8"):
            line=line.split("#")[0].strip()
            if line: toks.add(line)
    except FileNotFoundError:
        pass
    return toks
LOGIN_HTML="""<!doctype html><meta charset=utf-8><title>GuitarAI</title>
<style>body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
background:linear-gradient(135deg,#1a1a4e,#9b3b78,#ff9a5a);font-family:-apple-system,Segoe UI,sans-serif}
form{background:rgba(16,9,28,.6);backdrop-filter:blur(12px);padding:36px;border-radius:16px;
border:1px solid #ffffff33;text-align:center;color:#fff;box-shadow:0 20px 60px #0007}
input{padding:11px 14px;border-radius:10px;border:1px solid #ffffff44;background:#0008;color:#fff;font-size:15px;width:220px}
button{margin-top:14px;width:100%;padding:11px;border:0;border-radius:10px;font-weight:700;cursor:pointer;
background:linear-gradient(90deg,#f7971e,#ffd200);color:#201500}
h2{margin:0 0 18px}</style>
<form method=get action=/><h2>🎸 GuitarAI</h2>
<input name=key type=password placeholder="clave de acceso" autofocus>
<button>Entrar</button>__ERR__</form>"""

@app.before_request
def _auth():
    if request.path.startswith("/static/"): return
    toks=_valid_tokens()
    if request.cookies.get("dash") in toks: return
    if request.args.get("key") in toks: return
    err="<p style='color:#ffb3b3;font-size:12px;margin:12px 0 0'>clave incorrecta</p>" if request.args.get("key") else ""
    return Response(LOGIN_HTML.replace("__ERR__",err),mimetype="text/html",status=401)

@app.after_request
def _setcookie(resp):
    k=request.args.get("key")
    if k and k in _valid_tokens():
        resp.set_cookie("dash",k,max_age=60*60*24*30,samesite="Lax")
    return resp

def run_job(jid, opts):
    def prog(msg,pct): JOBS[jid].update(msg=msg,pct=pct)
    try:
        r=generar_cancion(progress=prog, out_root=OUT, **opts)
        JOBS[jid].update(done=True,result=r,pct=100,msg="¡Listo!")
    except Exception as e:
        JOBS[jid].update(done=True,error=str(e),trace=traceback.format_exc())

@app.route("/")
def index(): return Response(HTML,mimetype="text/html")

@app.route("/generar",methods=["POST"])
def generar():
    f=request.form
    name=f.get("name","Mi Cancion").strip() or "Mi Cancion"
    artist=f.get("artist","Desconocido").strip() or "Desconocido"
    make_chart=f.get("chart","1")=="1"
    make_lyrics=f.get("lyrics","1")=="1"
    make_video=f.get("video","0")=="1"
    video_source=f.get("video_source","mp4")
    youtube_url=f.get("youtube","").strip() or None
    letra=f.get("letra","").strip() or None
    jid=uuid.uuid4().hex[:8]; d=os.path.join(UPLOAD,jid); os.makedirs(d,exist_ok=True)
    def save(field,nm):
        ff=request.files.get(field)
        if ff and ff.filename:
            p=os.path.join(d,nm+os.path.splitext(ff.filename)[1]); ff.save(p); return p
        return None
    mp3=save("mp3","audio"); photo=save("photo","foto"); video_mp4=save("video_mp4","video_in")
    # validaciones
    if not (make_chart or make_video):
        return jsonify(error="Elige al menos Chart o Video"),400
    if make_chart and not mp3 and not youtube_url:
        return jsonify(error="Para el chart: sube un MP3 o pon un link de YouTube"),400
    if make_video and video_source=="mp4" and not video_mp4:
        return jsonify(error="Para el video: sube un MP4 (o cambia a YouTube)"),400
    if make_video and video_source=="youtube" and not youtube_url:
        return jsonify(error="Para el video: pon el link de YouTube"),400
    opts=dict(name=name,artist=artist,photo_path=photo,mp3_path=mp3,youtube_url=youtube_url,
              video_mp4=video_mp4,video_source=video_source,make_chart=make_chart,
              make_lyrics=make_lyrics,make_video=make_video,letra=letra)
    JOBS[jid]=dict(pct=0,msg="En cola...",done=False,error=None,result=None)
    threading.Thread(target=run_job,args=(jid,opts),daemon=True).start()
    return jsonify(job=jid)

@app.route("/estado/<jid>")
def estado(jid):
    j=JOBS.get(jid)
    if not j: return jsonify(error="job no existe"),404
    return jsonify(j)

@app.route("/descargar/<jid>")
def descargar(jid):
    j=JOBS.get(jid)
    if not j or not j.get("result"): return "No listo",404
    folder=j["result"]["folder"]
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for root,_,files in os.walk(folder):
            for fn in files:
                fp=os.path.join(root,fn); z.write(fp,os.path.relpath(fp,os.path.dirname(folder)))
    buf.seek(0)
    return send_file(buf,mimetype="application/zip",as_attachment=True,
                     download_name=os.path.basename(folder)+".zip")

HTML=r"""<!doctype html><html lang=es><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>GuitarAI — Generador Clone Hero</title>
<style>
*{box-sizing:border-box;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
body{margin:0;color:#fff;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px;overflow:hidden}

/* ===== ESCENA: imágenes de Naruto ===== */
.scene{position:fixed;inset:0;z-index:0;overflow:hidden;background:#05070e}
.bg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center 38%;
  opacity:0;transition:opacity 1.6s ease;animation:kb 30s ease-in-out infinite alternate}
body:not(.m-shippuden) .bg.sunset{opacity:1}
body.m-shippuden .bg.night{opacity:1}
@keyframes kb{from{transform:scale(1.03)}to{transform:scale(1.13)}}
/* viñeta suave: integra la imagen y deja leer el formulario */
.vig{position:absolute;inset:0;z-index:1;pointer-events:none;
  background:radial-gradient(125% 90% at 50% 16%,transparent 34%,rgba(3,5,12,.55) 100%),
   linear-gradient(180deg,rgba(3,5,12,.22),rgba(3,5,12,.04) 28%,rgba(3,5,12,.5))}
@media (prefers-reduced-motion:reduce){.bg{animation:none}}

/* ===== TARJETA / FORM ===== */
.card{position:relative;z-index:10;background:rgba(8,11,20,.32);backdrop-filter:blur(20px) saturate(1.25);
  -webkit-backdrop-filter:blur(20px) saturate(1.25);
  border:1px solid rgba(255,255,255,.18);border-radius:20px;padding:28px;max-width:480px;width:100%;
  box-shadow:0 24px 80px #000b,inset 0 1px 0 rgba(255,255,255,.12);max-height:92vh;overflow:auto}
h1{margin:0 0 4px;font-size:26px;background:linear-gradient(90deg,#ffd200,#ff8f5e);-webkit-background-clip:text;background-clip:text;color:transparent}
p.sub{margin:0 0 22px;color:#e9d6c8;font-size:13px}
label{display:block;font-size:13px;margin:14px 0 6px;color:#f0e2d6}
input[type=text],input[type=file],textarea{width:100%;padding:11px;border-radius:10px;border:1px solid #ffffff22;background:rgba(8,5,16,.72);color:#fff;font-size:14px;font-family:inherit}
textarea{resize:vertical;min-height:64px}
.row{display:flex;gap:12px}.row>div{flex:1}
.chk{display:flex;align-items:center;gap:8px;margin-top:14px;font-size:13px;color:#f0e2d6}
button{margin-top:22px;width:100%;padding:14px;border:0;border-radius:12px;font-size:16px;font-weight:700;cursor:pointer;background:linear-gradient(90deg,#f7971e,#ffd200);color:#201500}
button:disabled{opacity:.5;cursor:wait}
.bar{height:10px;background:#0e0c1f;border-radius:6px;overflow:hidden;margin-top:18px;display:none}
.bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#f7971e,#ffd200);transition:width .4s}
#msg{font-size:12px;color:#ffd9a8;margin-top:8px;min-height:16px}
#res{margin-top:18px;padding:14px;background:rgba(8,5,16,.72);border-radius:10px;font-size:13px;display:none}
a.dl{display:inline-block;margin-top:10px;color:#ffd200;font-weight:700;text-decoration:none}
.modes{display:flex;gap:8px;margin-bottom:16px}
.mode{flex:1;margin:0;padding:9px;font-size:13px;border-radius:10px;background:rgba(255,255,255,.08);
  color:#cfe0ff;border:1px solid #ffffff22;cursor:pointer;transition:.2s}
.mode.on{background:linear-gradient(90deg,#ff8a1f,#ffd24d);color:#201500;border-color:#ffd98a}
body.m-shippuden .card{border-color:#ff6fae44;box-shadow:0 24px 70px #000a,0 0 40px #ff2a6a22}
body.m-shippuden h1{background:linear-gradient(90deg,#ff5ea3,#b18bff);-webkit-background-clip:text;background-clip:text}
</style></head><body>
<div class="scene" id="scene">
  <img class="bg sunset" src="/static/img/classic.jpg" alt="">
  <img class="bg night" src="/static/img/shippuden.jpg" alt="">
  <div class="vig"></div>
</div>
<div class=card>
<div class="modes"><button type=button class="mode on" data-m="classic">🍃 Clásico</button><button type=button class="mode" data-m="shippuden">🔥 Shippuden</button></div>
<h1>🍥 GuitarAI</h1><p class=sub>Sube un MP3, una foto y genera tu canción para Clone Hero (chart difícil, voz dinámica, karaoke y video que late).</p>
<form id=f>
<div class=row><div><label>Nombre de la canción</label><input type=text name=name placeholder="nombre cancion"></div>
<div><label>Artista</label><input type=text name=artist placeholder="nombre artista"></div></div>

<label>🎵 Audio: sube un MP3…</label><input type=file name=mp3 accept=".mp3,audio/*">
<label>…o pega un link de YouTube (sirve para audio y/o video)</label>
<input type=text name=youtube placeholder="https://youtube.com/watch?v=...">

<label>🖼️ Foto para la carátula — opcional</label><input type=file name=photo accept="image/*">

<label>📝 Letra (opcional) — pégala y la calzo con la voz</label>
<textarea name=letra rows=3 placeholder="Pega aquí la letra (una línea = una frase). Vacío = la saco de la voz."></textarea>

<div style="margin-top:14px;font-weight:700;color:#ffd9a8">¿Qué generar?</div>
<label class=chk><input type=checkbox name=chart value=1 checked> 🎸 Chart (notas)</label>
<label class=chk><input type=checkbox name=lyrics value=1 checked> 🎤 Karaoke (letra en el chart)</label>
<label class=chk><input type=checkbox name=video value=1 id=vchk onchange="document.getElementById('vbox').style.display=this.checked?'block':'none'"> 🎬 Video de fondo</label>

<div id=vbox style="display:none;margin:6px 0 0 22px;padding:8px;border-left:2px solid #ffffff22">
  <label class=chk><input type=radio name=video_source value=mp4 checked onchange="vmode()"> Mi MP4 (desde mi compu)</label>
  <input type=file name=video_mp4 accept="video/mp4,video/*" id=vfile>
  <label class=chk style="margin-top:8px"><input type=radio name=video_source value=youtube onchange="vmode()"> Desde el link de YouTube de arriba</label>
</div>

<button type=submit id=btn>Generar</button>
</form>
<div class=bar id=bar><i id=fill></i></div><div id=msg></div>
<div id=res></div>
</div>
<script>
// ===== escena: modo + aviso de imágenes =====
(function(){
  const hint=document.getElementById('hint');
  const refreshHint=()=>{
    if(!hint)return;
    const sel=document.body.classList.contains('m-shippuden')?'.ch.sh':'.ch.cl';
    const imgs=[...document.querySelectorAll(sel)];
    const any=imgs.some(i=>i.naturalWidth>0 && i.style.display!=='none');
    hint.style.display=any?'none':'block';
  };
  document.querySelectorAll('.ch').forEach(i=>{i.addEventListener('load',refreshHint);i.addEventListener('error',refreshHint);});
  const setMode=m=>{document.body.classList.toggle('m-shippuden',m==='shippuden');
    document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
    try{localStorage.setItem('narutoMode',m)}catch(e){}
    refreshHint();};
  document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>setMode(b.dataset.m));
  let saved='classic';try{saved=localStorage.getItem('narutoMode')||'classic'}catch(e){}
  setMode(saved);
})();
const KEY=new URLSearchParams(location.search).get('key')||'';
const Q=KEY?('?key='+encodeURIComponent(KEY)):'';
const f=document.getElementById('f'),btn=document.getElementById('btn'),bar=document.getElementById('bar'),
fill=document.getElementById('fill'),msg=document.getElementById('msg'),res=document.getElementById('res');
f.onsubmit=async e=>{e.preventDefault();btn.disabled=true;res.style.display='none';bar.style.display='block';
msg.textContent='Subiendo...';fill.style.width='3%';
const HDR={'ngrok-skip-browser-warning':'1'};
const r=await fetch('/generar'+Q,{method:'POST',headers:HDR,body:new FormData(f)});
const j=await r.json();if(j.error){msg.textContent='Error: '+j.error;btn.disabled=false;return;}
const id=j.job;const poll=setInterval(async()=>{
 const s=await(await fetch('/estado/'+id+Q,{headers:HDR})).json();
 fill.style.width=(s.pct||0)+'%';msg.textContent=s.msg||'';
 if(s.done){clearInterval(poll);btn.disabled=false;
  if(s.error){msg.textContent='Error: '+s.error;return;}
  const R=s.result, hizo=(R.hizo||[]);
  let html='✅ <b>Listo</b> ('+hizo.join(' + ')+')<br>';
  if(hizo.includes('chart')){ const n=R.notas||{};
    html+='BPM '+R.bpm+' · '+R.dur+'s'+(R.palabras?(' · '+R.palabras+' palabras'):'')+
      '<br>Siguió: '+(R.guitarra?'🎸 guitarra/melodía (90/10)':'🎤 voz (sin guitarra)')+
      '<br>Notas — Exp '+n.ExpertSingle+' · Hard '+n.HardSingle+' · Med '+n.MediumSingle+' · Easy '+n.EasySingle+'<br>'; }
  html+='<span style=color:#9ad;font-size:11px>'+R.folder+'</span>'+
    '<br><a class=dl href="/descargar/'+id+Q+'">⬇ Descargar carpeta (.zip)</a>';
  res.style.display='block'; res.innerHTML=html;
 }},1500);
};
function vmode(){ var yt=document.querySelector('input[name=video_source]:checked');
  document.getElementById('vfile').style.display=(yt&&yt.value==='youtube')?'none':'block'; }
</script></body></html>"""

if __name__=="__main__":
    import os
    port=int(os.environ.get("PORT","5050"))
    print(f"GuitarAI dashboard -> http://127.0.0.1:{port}")
    app.run(host="0.0.0.0",port=port,threaded=True)
