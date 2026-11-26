import os, sys, json, queue, threading, time, ctypes
from collections import deque
from pathlib import Path

if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import tkinter as tk
import customtkinter as ctk
import updater
from version import __version__

DATA = Path(os.environ.get("APPDATA", str(Path.home()))) / "VCTranslator"
DATA.mkdir(parents=True, exist_ok=True)
SETTINGS = DATA / "settings.json"
LANGS = {"Auto-detect": None, "Russian": "ru", "Arabic": "ar", "Persian": "fa", "Turkish": "tr",
         "Spanish": "es", "French": "fr", "German": "de", "Portuguese": "pt", "Italian": "it",
         "Polish": "pl", "Ukrainian": "uk", "Japanese": "ja", "Korean": "ko", "Chinese": "zh", "Hindi": "hi"}
# quality -> (model, longest chunk in seconds, beam size, description)
QUALITY = {"Fast": ("base", 4.0, 1, "Small model, short chunks. Quickest, least accurate."),
           "Balanced": ("small", 6.0, 3, "Good mix of speed and accuracy."),
           "Best": ("medium", 10.0, 5, "Most accurate. ~1.5 GB download, best with the GPU on.")}
RATE = 16000
OVW, OVH, KEY = 640, 360, "#010203"
ACC, CARD, BD, BG, MUT, IN = "#7c5cff", "#141720", "#242836", "#0b0d12", "#868c9d", "#1b1f2a"
DEFAULT = {"src": "", "lang": "Auto-detect", "quality": "Balanced", "hide": True,
           "where": "Chat window", "lock": True, "gpu": True}

def load_cfg():
    try: return {**DEFAULT, **json.loads(SETTINGS.read_text("utf-8"))}
    except Exception: return dict(DEFAULT)

def list_sources():
    out = []
    try:  # apps that currently have an audio session (like "App: Discord")
        import psutil
        from pycaw.pycaw import AudioUtilities
        seen = {}
        for ses in AudioUtilities.GetAllSessions():
            pid = ses.ProcessId
            if not pid or pid == os.getpid(): continue
            try: name = psutil.Process(pid).name()
            except Exception: continue
            seen[pid] = name[:-4] if name.lower().endswith(".exe") else name
        names = list(seen.values())
        for pid, n in sorted(seen.items(), key=lambda kv: kv[1].lower()):
            out.append((f"App: {n}" + (f" (pid {pid})" if names.count(n) > 1 else ""), {"kind": "app", "pid": pid}))
    except Exception:
        pass
    import pyaudiowpatch as pa
    p = pa.PyAudio()
    try:
        def dev(d):
            return {"kind": "dev", "index": d["index"], "maxInputChannels": d["maxInputChannels"],
                    "defaultSampleRate": d["defaultSampleRate"]}
        for d in p.get_loopback_device_info_generator():
            out.append((f"Everything playing on {d['name'].replace(' [Loopback]', '')}", dev(d)))
        w = p.get_host_api_info_by_type(pa.paWASAPI)["index"]
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] == w and d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice"):
                out.append((f"Input: {d['name']}", dev(d)))
    finally:
        p.terminate()
    return out

def to16k(x, rate):
    if rate == RATE or len(x) == 0: return x
    if rate % RATE == 0:
        k = rate // RATE; return x[: len(x) // k * k].reshape(-1, k).mean(1)
    n = int(len(x) * RATE / rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

CUDA_DIR = DATA / "cuda"

def cuda_libs_ready():
    return any(CUDA_DIR.glob("cublas64_12.dll")) and any(CUDA_DIR.glob("cudnn64_9.dll"))

def use_cuda_libs():
    if CUDA_DIR.exists():
        os.add_dll_directory(str(CUDA_DIR))
        os.environ["PATH"] = str(CUDA_DIR) + os.pathsep + os.environ["PATH"]

def fetch_cuda_libs(post):
    import urllib.request, zipfile, shutil
    CUDA_DIR.mkdir(parents=True, exist_ok=True)
    for pkg in ("nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
        post("status", f"Downloading GPU libraries ({pkg}) - one time, about 1 GB in total…")
        info = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30))
        wheel = next(f for f in info["urls"] if f["filename"].endswith(".whl") and "win_amd64" in f["filename"])
        tmp = DATA / (pkg + ".whl")
        urllib.request.urlretrieve(wheel["url"], tmp)
        with zipfile.ZipFile(tmp) as z:
            for n in z.namelist():
                if n.lower().endswith(".dll"):
                    with z.open(n) as src, open(CUDA_DIR / Path(n).name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        tmp.unlink()

def load_model(name, use_gpu, post):
    import ctranslate2
    from faster_whisper import WhisperModel
    plan = []
    if use_gpu:
        try:
            if ctranslate2.get_cuda_device_count() > 0:
                if not cuda_libs_ready(): fetch_cuda_libs(post)
                use_cuda_libs()
                plan += [("cuda", "float16"), ("cuda", "int8_float16")]
            else:
                post("status", "No NVIDIA GPU found, using the CPU.")
        except Exception as e:
            post("status", f"GPU setup skipped ({e}), using the CPU.")
    plan.append(("cpu", "int8"))
    post("status", f"Loading the {name} model (the first time it downloads)…")
    last = None
    for dev, ct in plan:
        try:
            m = WhisperModel(name, device=dev, compute_type=ct, download_root=str(DATA / "models"))
            list(m.transcribe(np.zeros(RATE, dtype=np.float32), beam_size=1)[0])  # warm-up proves the GPU libs work
            return m, ("GPU" if dev == "cuda" else "CPU")
        except Exception as e:
            last = e
    raise last

class Engine:
    def __init__(self, post):
        self.post, self.run, self.gen = post, False, 0
        self.model = self.key_loaded = None; self.where = "CPU"

    def start(self, dev, lang, quality, hide, gpu):
        self.gen += 1; self.run = True
        self.dev, self.lang, self.hide, self.gpu = dev, lang, hide, gpu
        self.mname, self.maxlen, self.beam, _ = QUALITY[quality]
        if dev["kind"] == "app": self.ch, self.rate = 2, 48000
        else: self.ch, self.rate = int(dev["maxInputChannels"]), int(dev["defaultSampleRate"])
        self.audio_q, self.utt_q = queue.Queue(), queue.Queue(maxsize=6)
        threading.Thread(target=self._boot, args=(self.gen,), daemon=True).start()

    def stop(self): self.run = False

    def alive(self, g): return self.run and g == self.gen

    def _boot(self, g):
        try:
            key = (self.mname, self.gpu)
            if self.key_loaded != key:
                self.model, self.where = load_model(self.mname, self.gpu, self.post); self.key_loaded = key
            if not self.alive(g): return
            threading.Thread(target=self._segment, args=(g,), daemon=True).start()
            threading.Thread(target=self._transcribe, args=(g,), daemon=True).start()
            self._capture(g)
        except Exception as e:
            self.post("error", f"{type(e).__name__}: {e}")

    def _capture(self, g):
        if self.dev["kind"] == "app":  # one app only (Windows process loopback)
            import proctap
            Tap = getattr(proctap, "ProcessAudioCapture", None) or getattr(proctap, "ProcTap", None)
            if Tap is None:
                raise RuntimeError("proctap has no capture class: " + str([n for n in dir(proctap) if not n.startswith("_")]))
            pid, cb = self.dev["pid"], (lambda pcm, frames: self.audio_q.put(bytes(pcm)))
            try:
                try: tap = Tap(pid, on_data=cb)
                except TypeError: tap = Tap(pid=pid, on_data=cb)
                tap.start()
            except Exception as e:
                raise RuntimeError(f"{e} | proctap offers: {[n for n in dir(proctap) if not n.startswith('_')]}")
            self.post("status", f"Listening to the app… ({self.where})")
            try:
                while self.alive(g): time.sleep(0.1)
            finally:
                for fn in ("stop", "close"):
                    try: getattr(tap, fn)()
                    except Exception: pass
            return
        import pyaudiowpatch as pa
        p = pa.PyAudio()
        def cb(data, frames, t, status):
            self.audio_q.put(data); return (None, pa.paContinue)
        s = p.open(format=pa.paInt16, channels=self.ch, rate=self.rate, frames_per_buffer=self.rate // 10,
                   input=True, input_device_index=self.dev["index"], stream_callback=cb)
        self.post("status", f"Listening… ({self.where})")
        try:
            while self.alive(g): time.sleep(0.1)
        finally:
            s.stop_stream(); s.close(); p.terminate()

    def _pcm(self, raw):
        if self.dev["kind"] == "app" and len(raw) % 4 == 0:  # per-app capture delivers float32
            f = np.frombuffer(raw, dtype=np.float32)
            if len(f) == 0 or (np.isfinite(f).all() and float(np.abs(f).max()) <= 8.0):
                return f.astype(np.float32)
        n = len(raw) // 2 * 2
        return np.frombuffer(raw[:n], dtype=np.int16).astype(np.float32) / 32768.0

    def _segment(self, g):
        buf, pre, silence, speaking = [], deque(maxlen=3), 0.0, False
        while self.alive(g):
            try: raw = self.audio_q.get(timeout=0.3)
            except queue.Empty: continue
            x = self._pcm(raw)
            if self.ch > 1: x = x[: len(x) // self.ch * self.ch].reshape(-1, self.ch).mean(1)
            x = to16k(x, self.rate)
            rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
            if rms > 0.006:
                if not speaking: buf, speaking = list(pre), True
                buf.append(x); silence = 0.0
            elif speaking:
                buf.append(x); silence += len(x) / RATE
            else:
                pre.append(x)
            total = sum(len(b) for b in buf) / RATE
            if speaking and (silence >= 0.6 or total >= self.maxlen):
                if total > 0.8:
                    try: self.utt_q.put_nowait(np.concatenate(buf))
                    except queue.Full: pass
                buf, speaking, silence = [], False, 0.0

    def _transcribe(self, g):
        while self.alive(g):
            try: a = self.utt_q.get(timeout=0.3)
            except queue.Empty: continue
            try:
                segs, info = self.model.transcribe(a, task="translate", language=self.lang, beam_size=self.beam,
                                                   vad_filter=True, condition_on_previous_text=False,
                                                   without_timestamps=True)
                text = " ".join(s.text.strip() for s in segs).strip()
            except Exception as e:
                self.post("status", f"Transcribe error: {e}"); continue
            if len(text) < 2: continue
            if self.hide and info.language == "en": continue
            self.post("line", (text, info.language or "??"))

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title(f"VC Translator {__version__}"); self.geometry("450x900"); self.minsize(420, 600)
        self.configure(fg_color=BG)
        self.cfg, self.uiq = load_cfg(), queue.Queue()
        self.engine = Engine(lambda k, v: self.uiq.put((k, v)))
        self.sources, self.running, self.overlay, self.lines, self.pending = [], False, None, [], None
        self.build(); self.refresh_sources()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(150, self.pump)
        updater.check(lambda r: self.uiq.put(("update", r)))

    def card(self, title):
        f = ctk.CTkFrame(self.body, corner_radius=14, fg_color=CARD, border_width=1, border_color=BD)
        f.pack(fill="x", pady=6)
        ctk.CTkLabel(f, text=title, text_color=MUT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
        return f

    def lab(self, p, text, dim=False):
        ctk.CTkLabel(p, text=text, anchor="w", justify="left", wraplength=360,
                     text_color=MUT if dim else None, font=("Segoe UI", 12 if dim else 14)
                     ).pack(anchor="w", padx=14, pady=(2, 4) if dim else (8, 2))

    def build(self):
        c = self.cfg
        self.body = ctk.CTkScrollableFrame(self, fg_color=BG); self.body.pack(fill="both", expand=True)
        head = ctk.CTkFrame(self.body, fg_color="transparent"); head.pack(fill="x", pady=(4, 6))
        ctk.CTkLabel(head, text="VC Translator", font=("Segoe UI", 22, "bold")).pack(side="left")
        self.pill = ctk.CTkLabel(head, text="● Off", text_color=MUT, fg_color=CARD, corner_radius=12, width=64)
        self.pill.pack(side="right")

        a = self.card("AUDIO"); self.lab(a, "Voice chat source")
        row = ctk.CTkFrame(a, fg_color="transparent"); row.pack(fill="x", padx=14, pady=(0, 6))
        self.src = ctk.CTkOptionMenu(row, values=["…"], fg_color=IN, button_color=IN, dynamic_resizing=False)
        self.src.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="↻", width=36, fg_color=IN, hover_color=BD, command=self.refresh_sources).pack(side="left", padx=(8, 0))
        self.lab(a, "“App: …” hears only that app. Start the game and join voice chat first, then press ↻ so it shows up. “Everything playing on…” hears the whole PC.", True)

        t = self.card("TRANSLATION"); self.lab(t, "Teammates speak")
        self.lang = ctk.CTkOptionMenu(t, values=list(LANGS), fg_color=IN, button_color=IN)
        self.lang.set(c["lang"] if c["lang"] in LANGS else "Auto-detect"); self.lang.pack(fill="x", padx=14)
        self.lab(t, "Auto-detect works out each line on its own. Pin a language if short callouts get misread. Everything is translated into English.", True)
        self.hide = ctk.CTkSwitch(t, text="Hide lines already in English", progress_color=ACC)
        self.hide.pack(anchor="w", padx=14, pady=8)
        if c["hide"]: self.hide.select()
        self.gpu = ctk.CTkSwitch(t, text="Use NVIDIA GPU (faster; downloads ~1 GB once)", progress_color=ACC)
        self.gpu.pack(anchor="w", padx=14, pady=(0, 8))
        if c["gpu"]: self.gpu.select()
        self.lab(t, "Quality")
        self.qual = ctk.CTkSegmentedButton(t, values=list(QUALITY), selected_color=ACC, selected_hover_color=ACC, command=self.on_quality)
        self.qual.set(c["quality"]); self.qual.pack(fill="x", padx=14)
        self.qdesc = ctk.CTkLabel(t, text="", text_color=MUT, font=("Segoe UI", 12), anchor="w"); self.qdesc.pack(anchor="w", padx=14, pady=(2, 4))
        self.on_quality(c["quality"])
        self.lab(t, "Show translations in")
        self.where = ctk.CTkSegmentedButton(t, values=["Chat window", "Game overlay"], selected_color=ACC, selected_hover_color=ACC, command=self.on_where)
        self.where.set(c["where"]); self.where.pack(fill="x", padx=14)
        self.lock = ctk.CTkSwitch(t, text="Lock overlay (clicks go to the game)", progress_color=ACC, command=self.apply_lock)
        self.lock.pack(anchor="w", padx=14, pady=(12, 14))
        if c["lock"]: self.lock.select()

        self.go = ctk.CTkButton(self.body, text="▷  Start listening", height=46, fg_color=ACC, font=("Segoe UI", 15, "bold"), command=self.toggle)
        self.go.pack(fill="x", pady=(10, 4))
        self.status = ctk.CTkLabel(self.body, text="Ready. Press Start when you’re in a match.", text_color=MUT, anchor="w", wraplength=400, justify="left")
        self.status.pack(anchor="w", padx=2, pady=2)
        self.box = ctk.CTkTextbox(self.body, height=230, font=("Segoe UI", 16), fg_color=CARD, wrap="word", border_width=1, border_color=BD)
        self.box.pack(fill="x", pady=8)
        foot = ctk.CTkFrame(self.body, fg_color="transparent"); foot.pack(fill="x")
        self.ver = ctk.CTkLabel(foot, text=f"v{__version__}", text_color=MUT); self.ver.pack(side="left")
        self.ubtn = ctk.CTkButton(foot, text="Check for updates", width=140, fg_color=IN, hover_color=BD, command=self.check_now)
        self.ubtn.pack(side="right")

    def on_quality(self, v): self.qdesc.configure(text=QUALITY[v][3])

    def refresh_sources(self):
        try: self.sources = list_sources()
        except Exception as e: self.sources = []; self.status.configure(text=f"Couldn’t list audio devices: {e}")
        names = [n for n, _ in self.sources] or ["No audio devices found"]
        self.src.configure(values=names)
        self.src.set(self.cfg["src"] if self.cfg["src"] in names else names[0])

    def save(self):
        self.cfg.update(src=self.src.get(), lang=self.lang.get(), quality=self.qual.get(),
                        hide=bool(self.hide.get()), gpu=bool(self.gpu.get()), where=self.where.get(), lock=bool(self.lock.get()))
        try: SETTINGS.write_text(json.dumps(self.cfg), "utf-8")
        except Exception: pass

    def toggle(self):
        if self.running:
            self.engine.stop(); self.set_running(False); self.status.configure(text="Stopped."); return
        dev = dict(self.sources).get(self.src.get())
        if not dev: self.status.configure(text="Pick an audio source first."); return
        self.save(); self.set_running(True)
        self.engine.start(dev, LANGS[self.lang.get()], self.qual.get(), bool(self.hide.get()), bool(self.gpu.get()))

    def set_running(self, v):
        self.running = v
        self.go.configure(text="■  Stop listening" if v else "▷  Start listening", fg_color="#e5484d" if v else ACC)
        self.pill.configure(text="● On" if v else "● Off", text_color="#2fbf71" if v else MUT)

    def add_line(self, item):
        text, lang = item if isinstance(item, tuple) else (item, "??")
        self.box.insert("1.0", f"[{lang.upper()}] {text}\n\n")
        self.lines = (self.lines + [(time.time(), text, lang)])[-30:]

    def on_where(self, v):
        if v == "Game overlay": self.show_overlay()
        elif self.overlay: self.overlay.destroy(); self.overlay = None

    def show_overlay(self):
        if self.overlay: return
        o = tk.Toplevel(self); o.overrideredirect(True); o.attributes("-topmost", True)
        o.configure(bg=KEY); o.attributes("-transparentcolor", KEY)
        x, y = self.cfg.get("ovx"), self.cfg.get("ovy")
        if x is None: x, y = (self.winfo_screenwidth() - OVW) // 2, 80
        o.geometry(f"{OVW}x{OVH}+{x}+{y}")
        self.cv = tk.Canvas(o, width=OVW, height=OVH, bg=KEY, highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.cv.bind("<ButtonPress-1>", self._ds); self.cv.bind("<B1-Motion>", self._dm); self.cv.bind("<ButtonRelease-1>", self._de)
        self._sig = None; self.overlay = o; self.after(250, self.apply_lock)

    def _ds(self, e): self._dx, self._dy = e.x_root - self.overlay.winfo_x(), e.y_root - self.overlay.winfo_y()
    def _dm(self, e): self.overlay.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")
    def _de(self, e):
        self.cfg["ovx"], self.cfg["ovy"] = self.overlay.winfo_x(), self.overlay.winfo_y(); self.save()

    def rrect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1+r,y1, x1+r,y1, x2-r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y1+r, x2,y2-r, x2,y2-r, x2,y2,
               x2-r,y2, x2-r,y2, x1+r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y2-r, x1,y1+r, x1,y1+r, x1,y1]
        return self.cv.create_polygon(pts, smooth=True, **kw)

    def apply_lock(self):
        if not self.overlay: return
        try:
            u = ctypes.windll.user32
            hwnd = u.GetParent(self.overlay.winfo_id()) or self.overlay.winfo_id()
            st = u.GetWindowLongW(hwnd, -20) | 0x80000 | 0x80
            st = st | 0x20 if self.lock.get() else st & ~0x20
            u.SetWindowLongW(hwnd, -20, st)
        except Exception: pass

    def refresh_overlay(self):
        if not self.overlay: return
        now, locked = time.time(), bool(self.lock.get())
        live = [(t, l) for ts, t, l in self.lines if now - ts < min(14, 4 + 0.07 * len(t))][-5:]
        sig = (tuple(live), locked)
        if sig == getattr(self, "_sig", None): return
        self._sig = sig
        cv = self.cv; cv.delete("all"); y = 8
        if not locked:  # visible, draggable area plus the hint
            self.rrect(2, 2, OVW - 2, OVH - 2, 14, fill="#12141b", outline=ACC, width=2)
            cv.create_text(20, 24, text="Drag to position. Lock the overlay in the app when done.", anchor="w", fill="white", font=("Segoe UI", 11, "bold"))
            y = 52
        for text, lang in live:
            chip = self.rrect(16, y + 9, 52, y + 31, 6, fill="#2b2644")
            cv.create_text(34, y + 20, text=lang.upper()[:3], fill="#a58bff", font=("Segoe UI", 9, "bold"))
            tid = cv.create_text(64, y + 8, text=text, anchor="nw", fill="white", font=("Segoe UI", 15, "bold"), width=OVW - 94)
            x2, y2 = cv.bbox(tid)[2:]
            bottom = max(y2 + 8, y + 40)
            cv.tag_lower(self.rrect(8, y, x2 + 14, bottom, 12, fill="#16181f"), chip)
            y = bottom + 6

    def check_now(self):
        self.ubtn.configure(text="Checking…", state="disabled")
        updater.check(lambda r: self.uiq.put(("update", r)))

    def on_update(self, r):
        if r[0] == "new":
            self.pending = r; self.ver.configure(text=f"Update v{r[1]} available")
            self.ubtn.configure(text=f"Install v{r[1]}", state="normal", fg_color=ACC, command=self.install_update)
        else:
            self.ver.configure(text=f"v{__version__} · " + ("up to date" if r[0] == "none" else "couldn’t check"))
            self.ubtn.configure(text="Check for updates", state="normal", fg_color=IN, command=self.check_now)
            if r[0] == "err": self.status.configure(text=f"Update check: {r[1]}")

    def install_update(self):
        self.ubtn.configure(text="Downloading…", state="disabled")
        threading.Thread(target=updater.apply, args=(self.pending[2], lambda m: self.uiq.put(("status", "Update failed: " + m))), daemon=True).start()

    def pump(self):
        try:
            while True:
                k, v = self.uiq.get_nowait()
                if k == "line": self.add_line(v)
                elif k == "status": self.status.configure(text=v)
                elif k == "error": self.status.configure(text="Error: " + v); self.set_running(False)
                elif k == "update": self.on_update(v)
        except queue.Empty: pass
        self.refresh_overlay(); self.after(150, self.pump)

    def close(self):
        self.save(); self.engine.stop(); self.destroy(); os._exit(0)

if __name__ == "__main__":
    App().mainloop()
