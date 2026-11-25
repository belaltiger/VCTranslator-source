import json, os, subprocess, tempfile, threading, urllib.request
from version import __version__, UPDATE_REPO

def _vt(v):
    return tuple(int(x) for x in v.lstrip("vV").split(".") if x.isdigit())

def check(cb):
    """cb gets ("new", version, url) / ("none",) / ("err", message)."""
    def run():
        try:
            if UPDATE_REPO.startswith("YOUR_"):
                cb(("err", "Set UPDATE_REPO in version.py first")); return
            req = urllib.request.Request(
                f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest",
                headers={"User-Agent": "VCTranslator"})
            d = json.load(urllib.request.urlopen(req, timeout=10))
            tag = d["tag_name"]
            if _vt(tag) > _vt(__version__):
                for a in d.get("assets", []):
                    if a["name"].lower().endswith("setup.exe"):
                        cb(("new", tag.lstrip("vV"), a["browser_download_url"])); return
            cb(("none",))
        except Exception as e:
            cb(("err", str(e)))
    threading.Thread(target=run, daemon=True).start()

def apply(url, on_error):
    try:
        path = os.path.join(tempfile.gettempdir(), "VCTranslator-Setup.exe")
        urllib.request.urlretrieve(url, path)
        subprocess.Popen([path, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"])
        os._exit(0)
    except Exception as e:
        on_error(str(e))
