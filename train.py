"""Wanderpin scanner trainer — runs on GitHub's servers every hour.

Each run:
  1. finds street photos on Mapillary's coverage map (CC BY-SA) and plate photos on Wikimedia Commons,
     favouring the countries with the fewest examples so far; 16 downloads run side by side
  2. cuts out the whole scene (4 views from 360° photos), poles, signs, guardrails, street lamps,
     road posts and plates, plus a shifted/zoomed copy of each (so it learns from more than one framing)
  3. turns each cut-out into the model's 512-number "embedding" and adds it to the collection
     (a small copy of each cut-out is kept too, for the weekly fine-tuning)
  4. retrains every country-style classifier, the "which area of the country?" classifier and every
     "is it really there?" check; for each one it tries a straight-line classifier and a small neural
     network and keeps whichever does better on places it has never seen
  5. saves the results to the wp_scan_heads table, which every player's scanner reads

Once a week (when there's enough), it also fine-tunes the last 2 layers of the image model itself on
the collection, and publishes them only if they beat the current ones on unseen places.

Secrets (GitHub → Settings → Secrets and variables → Actions):
  MAPILLARY_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""
import base64, io, json, math, os, random, sqlite3, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime, timezone

import numpy as np
import requests
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- settings ----------------
SB_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
MLY_TOKEN = os.environ.get("MAPILLARY_TOKEN", "")
MINUTES = float(os.environ.get("MINUTES") or 40)              # time spent collecting photos each run
THREADS = int(os.environ.get("THREADS") or 16)                # downloads running side by side
PER_TILE = int(os.environ.get("PER_TILE") or 6)               # photos taken from each map square per trip (different drives first)
TILE_BUDGET = int(os.environ.get("TILE_BUDGET") or 1800)      # map squares per run (Mapillary allows 50,000 a day)
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS") or 800)   # cap per country per style (older ones get swapped out)
MAX_PLATE = int(os.environ.get("MAX_PLATE") or 600)
MAX_PER_CHECK = int(os.environ.get("MAX_PER_CHECK") or 4000)  # cap per yes/no answer for the "is it really there?" checks
FT_EVERY_DAYS = float(os.environ.get("FT_EVERY_DAYS") or 7)   # fine-tune the image model this often…
FT_MIN = int(os.environ.get("FT_MIN") or 30000)               # …once this many cut-outs with pictures are collected
FT_MINUTES = float(os.environ.get("FT_MINUTES") or 90)        # time spent fine-tuning
FT_MAX = int(os.environ.get("FT_MAX") or 60000)               # cut-outs used for fine-tuning
FT_GAIN = float(os.environ.get("FT_GAIN") or 0.015)           # must beat the current layers by this much to be published
FORCE_FT = os.environ.get("FORCE_FT", "") in ("1", "true")
SEED = os.environ.get("SEED")
MLY = os.environ.get("MLY_BASE", "https://graph.mapillary.com")
TILES = os.environ.get("MLY_TILES", "https://tiles.mapillary.com/maps/vtp/mly1_public/2")
COMMONS = os.environ.get("COMMONS_API", "https://commons.wikimedia.org/w/api.php")
BASE_NAME = "wanderpin-vision-2.onnx"                         # in this repo (models/) and uploaded to Supabase for the players
BASE_PATH = os.path.join(HERE, "models", BASE_NAME)
PLACES_URL = os.environ.get("PLACES_URL", "https://cdn.jsdelivr.net/npm/all-the-cities@3.1.0/cities.pbf")
BORDERS_URL = os.environ.get("BORDERS_URL", "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_50m_admin_0_countries.geojson")
CACHE = os.environ.get("CACHE_DIR", "cache")                 # kept between runs by GitHub
BACKUP = os.environ.get("BACKUP", "") in ("1", "true")        # also copy the collection's numbers to Supabase Storage this run
UA = {"User-Agent": "WanderpinTrainer/2.0 (https://github.com/roostercodes/wanderpin-trainer)"}

COVERAGE = ("US CA MX GT DO PR CW CO EC PE BO CL AR UY BR GB IE FR ES PT IT DE NL BE LU CH AT DK NO SE FI IS EE LV LT PL CZ SK HU "
            "SI HR RS ME AL MK BG RO GR MT AD FO UA RU TR JP KR TW HK PH TH MY SG ID KH LA IN LK BD BT MN KG KZ IL JO AE QA ZA BW "
            "SZ LS KE UG RW GH SN NG TN AU NZ").split()
NAMES = {"US": "the United States", "CA": "Canada", "MX": "Mexico", "GT": "Guatemala", "DO": "the Dominican Republic", "PR": "Puerto Rico", "CW": "Curaçao",
         "CO": "Colombia", "EC": "Ecuador", "PE": "Peru", "BO": "Bolivia", "CL": "Chile", "AR": "Argentina", "UY": "Uruguay", "BR": "Brazil",
         "GB": "the United Kingdom", "IE": "Ireland", "FR": "France", "ES": "Spain", "PT": "Portugal", "IT": "Italy", "DE": "Germany", "NL": "the Netherlands",
         "BE": "Belgium", "LU": "Luxembourg", "CH": "Switzerland", "AT": "Austria", "DK": "Denmark", "NO": "Norway", "SE": "Sweden", "FI": "Finland",
         "IS": "Iceland", "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania", "PL": "Poland", "CZ": "the Czech Republic", "SK": "Slovakia", "HU": "Hungary",
         "SI": "Slovenia", "HR": "Croatia", "RS": "Serbia", "ME": "Montenegro", "AL": "Albania", "MK": "North Macedonia", "BG": "Bulgaria", "RO": "Romania",
         "GR": "Greece", "MT": "Malta", "AD": "Andorra", "FO": "the Faroe Islands", "UA": "Ukraine", "RU": "Russia", "TR": "Turkey", "JP": "Japan",
         "KR": "South Korea", "TW": "Taiwan", "HK": "Hong Kong", "PH": "the Philippines", "TH": "Thailand", "MY": "Malaysia", "SG": "Singapore",
         "ID": "Indonesia", "KH": "Cambodia", "LA": "Laos", "IN": "India", "LK": "Sri Lanka", "BD": "Bangladesh", "BT": "Bhutan", "MN": "Mongolia",
         "KG": "Kyrgyzstan", "KZ": "Kazakhstan", "IL": "Israel", "JO": "Jordan", "AE": "the United Arab Emirates", "QA": "Qatar", "ZA": "South Africa",
         "BW": "Botswana", "SZ": "Eswatini", "LS": "Lesotho", "KE": "Kenya", "UG": "Uganda", "RW": "Rwanda", "GH": "Ghana", "SN": "Senegal",
         "NG": "Nigeria", "TN": "Tunisia", "AU": "Australia", "NZ": "New Zealand"}
STYLE_CATS = ["scene", "pole", "sign", "guardrail", "lamp", "bollard", "plate"]
DET_CATS = ["det_pole", "det_sign", "det_guardrail", "det_lamp", "det_bollard"]
CROP_PX = {"scene": 224, "plate": 192}                       # size of the kept copy of each cut-out (others: 160)
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)
def gh_error(msg): print("::error::" + str(msg).replace("\n", " | ")[:900], flush=True)   # shows on the run page (and to the API)
def gh_note(msg): print("::notice::" + str(msg).replace("\n", " | ")[:900], flush=True)
http = requests.Session(); http.headers.update(UA)
http.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=THREADS * 2))


# ---------------- Supabase ----------------
def sb_headers(extra=None):
    # new sb_secret_ keys go only on the apikey header; old service_role keys (eyJ…) also go on Authorization
    h = {"apikey": SB_KEY} if SB_KEY.startswith("sb_") else {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"}
    h.update(extra or {}); return h
def storage_get(path):
    r = http.get(f"{SB_URL}/storage/v1/object/wp-models/{path}", headers=sb_headers(), timeout=300)
    if r.status_code in (400, 404): return None
    r.raise_for_status(); return r.content
def storage_put(path, data, ctype="application/octet-stream"):
    r = http.post(f"{SB_URL}/storage/v1/object/wp-models/{path}", headers=sb_headers({"content-type": ctype, "x-upsert": "true"}), data=data, timeout=600)
    if not r.ok: raise RuntimeError(f"upload {path} failed: {r.status_code} {r.text[:200]}")
def public_size(path):
    try:
        r = http.head(f"{SB_URL}/storage/v1/object/public/wp-models/{path}", timeout=30, allow_redirects=True)
        return int(r.headers.get("content-length", -1)) if r.ok else None
    except requests.RequestException: return -1
def save_head(hid, data):
    r = http.post(f"{SB_URL}/rest/v1/wp_scan_heads", headers=sb_headers({"content-type": "application/json", "Prefer": "resolution=merge-duplicates"}),
                  data=json.dumps([{"id": hid, "data": data, "updated_at": datetime.now(timezone.utc).isoformat()}]), timeout=60)
    if not r.ok: raise RuntimeError(f"saving {hid} failed: {r.status_code} {r.text[:200]} (did you run the newest setup SQL?)")
def read_head(hid):
    try:
        r = http.get(f"{SB_URL}/rest/v1/wp_scan_heads", params={"id": f"eq.{hid}", "select": "data"}, headers=sb_headers(), timeout=30)
        rows = r.json() if r.ok else []
        return rows[0]["data"] if rows else None
    except Exception: return None


# ---------------- downloads, kept between runs ----------------
def cached_download(url, name):
    path = os.path.join(CACHE, name); os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(path): return open(path, "rb").read()
    r = http.get(url, timeout=300)
    if not r.ok: return None
    with open(path, "wb") as f: f.write(r.content)
    return r.content
def state_get():
    try: return json.load(open(os.path.join(CACHE, "state.json")))
    except Exception: return {}
def state_set(**kw):
    s = state_get(); s.update(kw); os.makedirs(CACHE, exist_ok=True)
    json.dump(s, open(os.path.join(CACHE, "state.json"), "w"))

# ---------------- live progress for the in-game page ----------------
RUN_KEY = os.environ.get("GITHUB_RUN_ID") or f"local-{int(T0)}"
def report(status, stage, **f):
    row = {"run_key": RUN_KEY, "status": status, "stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), "minutes": round((time.time() - T0) / 60, 1)}
    row.update(f)
    try:
        r = http.post(f"{SB_URL}/rest/v1/wp_scan_runs?on_conflict=run_key", headers=sb_headers({"content-type": "application/json", "Prefer": "resolution=merge-duplicates"}), data=json.dumps([row]), timeout=30)
        if not r.ok and not getattr(report, "warned", False): report.warned = True; gh_error(f"Couldn't update live progress: {r.status_code} {r.text[:150]} (run the newest setup SQL)")
    except Exception as e: log("progress update failed:", e)


# ---------------- the model ----------------
def prep(img, box=None):
    if box is not None:
        x, y, w, h = box; img = img.crop((int(x), int(y), int(x + w), int(y + h)))
    c = img.resize((224, 224), Image.BICUBIC) if img.size != (224, 224) else img
    a = (np.asarray(c, np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1)

class Model:
    """The players' image model (base) plus, when published, the fine-tuned last layers (branch)."""
    def __init__(self, meta):
        import onnxruntime as ort
        self.ort = ort
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
        self.so = so
        self.s = ort.InferenceSession(BASE_PATH, so, providers=["CPUExecutionProvider"])
        self.outs = [o.name for o in self.s.get_outputs()]
        self.inp, self.meta = self.s.get_inputs()[0].name, meta
        self.boll = meta["concepts"].index(next(c for c in meta["concepts"] if c["id"] == "bollard"))
        self.branch, self.ver = None, 0
    def load_branch(self, blob, ver):
        self.branch = self.ort.InferenceSession(blob, self.so, providers=["CPUExecutionProvider"]); self.ver = ver
    def run(self, arrays, want_tok=False):
        """arrays: list of 3×224×224 float arrays. Returns emb N×512, bollard prob N, emb2 N×512 (or None), tok (if asked)."""
        embs, boll, e2s, toks = [], [], [], []
        for i in range(0, len(arrays), 32):
            X = np.stack(arrays[i:i + 32]).astype(np.float32)
            logits, emb, tok = self.s.run(["logits", "emb", "tok"], {self.inp: X})
            embs.append(emb)
            lg = [np.max(logits[:, c["p"]], axis=1) for c in self.meta["concepts"]] + [logits[:, p] for p in self.meta["bg"]]
            L = np.stack(lg, 1); L = np.exp(L - L.max(1, keepdims=True)); L /= L.sum(1, keepdims=True)
            boll.append(L[:, self.boll])
            if self.branch is not None: e2s.append(self.branch.run(None, {"tok": tok})[0])
            if want_tok: toks.append(tok.astype(np.float16))
        if not embs: return np.zeros((0, 512), np.float32), np.zeros(0), None, None
        return (np.concatenate(embs), np.concatenate(boll), np.concatenate(e2s) if e2s else None,
                np.concatenate(toks) if want_tok else None)


# ---------------- places and borders ----------------
def decode_places(buf):
    want, out = set(COVERAGE), {}
    p, n, last_lat, last_lon = 0, len(buf), 0, 0
    def vint():
        nonlocal p
        v = s = 0
        while True:
            b = buf[p]; p += 1; v |= (b & 0x7f) << s; s += 7
            if not b & 0x80: return v
    def svint():
        v = vint(); return -((v + 1) >> 1) if v & 1 else v >> 1
    while p < n:
        end = vint() + p; cc, pop, name = "", 0, ""
        while p < end:
            key = vint(); tag, wt = key >> 3, key & 7
            if tag == 10: last_lon += svint()
            elif tag == 11: last_lat += svint()
            elif tag == 9: pop = vint()
            elif wt == 0: vint()
            elif wt == 2:
                l = vint()
                if tag == 3: cc = bytes(buf[p:p + min(l, 2)]).decode("ascii", "ignore")
                elif tag == 2: name = bytes(buf[p:p + l]).decode("utf-8", "ignore")
                p += l
            elif wt == 5: p += 4
            elif wt == 1: p += 8
            else: p = end
        p = end
        if cc in want: out.setdefault(cc, []).append((last_lat / 1e5, last_lon / 1e5, pop, name))
    return out

def load_borders(data):
    from shapely.geometry import shape, Point
    from shapely.prepared import prep as sprep
    shapes = {}
    for f in data["features"]:
        a2 = f["properties"].get("ISO_A2_EH") or f["properties"].get("ISO_A2")
        if a2 in COVERAGE: shapes[a2] = shape(f["geometry"])
    prepped = {k: sprep(v) for k, v in shapes.items()}
    def country_at(lat, lng, hint=None):
        pt = Point(lng, lat)
        if hint in prepped and prepped[hint].contains(pt): return hint
        for k, s in prepped.items():
            if s.contains(pt): return k
        return None
    def random_point(cc):
        g = shapes.get(cc)
        if g is None: return None
        x0, y0, x1, y1 = g.bounds
        for _ in range(200):
            x, y = random.uniform(x0, x1), random.uniform(y0, y1)
            if prepped[cc].contains(Point(x, y)): return y, x
        return None
    return country_at, random_point


# ---------------- Mapillary vector tiles ----------------
def _varint(u, o):
    v = s = 0
    while True:
        b = u[o[0]]; o[0] += 1; v |= (b & 0x7f) << s; s += 7
        if not b & 0x80: return v
def _walk(u, start, end, on):
    o = [start]
    while o[0] < end:
        key = _varint(u, o); f, wt = key >> 3, key & 7
        if wt == 0: on(f, wt, _varint(u, o))
        elif wt == 2:
            l = _varint(u, o); s = o[0]; o[0] += l; on(f, wt, (s, s + l))
        elif wt == 5: on(f, wt, (o[0], o[0] + 4)); o[0] += 4
        elif wt == 1: on(f, wt, (o[0], o[0] + 8)); o[0] += 8
        else: break
def _geom(u, s, e):
    o = [s]; x = y = 0; pts = []
    while o[0] < e:
        ci = _varint(u, o); cid, cnt = ci & 7, ci >> 3
        if cid == 7: continue
        for _ in range(cnt):
            dx, dy = _varint(u, o), _varint(u, o)
            x += -((dx + 1) >> 1) if dx & 1 else dx >> 1
            y += -((dy + 1) >> 1) if dy & 1 else dy >> 1
            pts.append((x, y))
    return pts

def mvt_layers(u):
    """Decode a Mapillary vector tile: {layer name: (extent, [(properties, [(x, y), ...]), ...])}."""
    layers = {}
    def layer(f, wt, v):
        if f != 3 or wt != 2: return
        st = {"name": "", "extent": 4096, "keys": [], "vals": [], "feats": []}
        def lf(f2, wt2, v2):
            if f2 == 1 and wt2 == 2: st["name"] = bytes(u[v2[0]:v2[1]]).decode("utf-8", "ignore")
            elif f2 == 5 and wt2 == 0: st["extent"] = v2
            elif f2 == 3 and wt2 == 2: st["keys"].append(bytes(u[v2[0]:v2[1]]).decode("utf-8", "ignore"))
            elif f2 == 4 and wt2 == 2:
                val = [None]
                def vf(f3, wt3, v3):
                    if f3 == 1 and wt3 == 2: val[0] = bytes(u[v3[0]:v3[1]]).decode("utf-8", "ignore")
                    elif f3 == 2 and wt3 == 5: val[0] = float(np.frombuffer(bytes(u[v3[0]:v3[1]]), "<f4")[0])
                    elif f3 == 3 and wt3 == 1: val[0] = float(np.frombuffer(bytes(u[v3[0]:v3[1]]), "<f8")[0])
                    elif f3 in (4, 5) and wt3 == 0: val[0] = v3
                    elif f3 == 6 and wt3 == 0: val[0] = -((v3 + 1) >> 1) if v3 & 1 else v3 >> 1
                    elif f3 == 7 and wt3 == 0: val[0] = bool(v3)
                _walk(u, v2[0], v2[1], vf); st["vals"].append(val[0])
            elif f2 == 2 and wt2 == 2:
                ft = {"tags": [], "geom": None}
                def ff(f3, wt3, v3):
                    if f3 == 2 and wt3 == 2:
                        o = [v3[0]]
                        while o[0] < v3[1]: ft["tags"].append(_varint(u, o))
                    elif f3 == 4 and wt3 == 2: ft["geom"] = v3
                _walk(u, v2[0], v2[1], ff); st["feats"].append(ft)
        _walk(u, v[0], v[1], lf)
        feats = []
        for ft in st["feats"]:
            t = ft["tags"]; props = {st["keys"][t[i]]: st["vals"][t[i + 1]] for i in range(0, len(t) - 1, 2) if t[i] < len(st["keys"]) and t[i + 1] < len(st["vals"])}
            feats.append((props, _geom(u, *ft["geom"]) if ft["geom"] else []))
        layers[st["name"]] = (st["extent"], feats)
    _walk(u, 0, len(u), layer)
    return layers

def mvt_box(b64):
    """Bounding box (0..1) of a Mapillary detection shape."""
    u = base64.b64decode(b64)
    for extent, feats in mvt_layers(u).values():
        pts = [p for _, g in feats for p in g]
        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            return min(xs) / extent, min(ys) / extent, max(xs) / extent, max(ys) / extent
    return None

def tile_of(lat, lng, z=14):
    n = 2 ** z; x = int((lng + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y
def tile_ll(z, x, y, fx, fy):
    n = 2 ** z; lng = (x + fx) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + fy) / n)))); return lat, lng

def det_cat(v):
    if "utility-pole" in v: return "pole"
    if "traffic-sign" in v and "back" not in v: return "sign"
    if "guard-rail" in v: return "guardrail"
    if "street-light" in v: return "lamp"
    if "bollard" in v or "delineator" in v: return "bollard"
    return None

class TokenError(SystemExit): pass
def get(url, params=None, tries=3, timeout=30):
    for k in range(tries):
        try:
            r = http.get(url, params=params, timeout=timeout)
            if r.status_code in (401, 403) and "mapillary" in url: raise TokenError("Mapillary refused MAPILLARY_TOKEN. Copy the Client Token (MLY|…) again into the GitHub secret.")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (k + 1)); continue
            return r
        except requests.RequestException:
            time.sleep(1.5 * (k + 1))
    return None

def fetch_image(url):
    r = get(url, timeout=40)
    if r is None or not r.ok: return None
    try:
        im = Image.open(io.BytesIO(r.content)); im.draft("RGB", (2048, 2048)); return im.convert("RGB")
    except Exception: return None

TILE = {"left": {}, "used": 0, "empty": set()}   # map square → photos not used yet; squares known to be empty (kept between runs)
def tile_candidates(cc, lat, lng, country_at, seen):
    """Photos in the ~2 km map square around a point, inside the country: one per drive (sequence) first,
    then more from the same drives. The square is remembered, so later trips there cost nothing."""
    x, y = tile_of(lat, lng); key = f"14/{x}/{y}"
    if key in TILE["empty"]: return [], key   # free: known to have no photos in this country
    if key not in TILE["left"]:
        if TILE["used"] >= TILE_BUDGET: return None, key
        TILE["used"] += 1; TILE["left"][key] = []
        r = get(f"{TILES}/14/{x}/{y}", {"access_token": MLY_TOKEN}, timeout=20)
        if r is None or not r.ok or not r.content: return [], key
        raw = r.content
        if raw[:2] == b"\x1f\x8b": import gzip; raw = gzip.decompress(raw)   # some tile servers send it still zipped
        try: ext, feats = mvt_layers(raw).get("image", (4096, []))
        except Exception: return [], key
        by_seq = {}
        for props, pts in feats:
            if not pts or "id" not in props: continue
            by_seq.setdefault(props.get("sequence_id") or props["id"], []).append((str(props["id"]), pts[0], props))
        seqs = list(by_seq.values()); random.shuffle(seqs)
        for g in seqs: random.shuffle(g)
        order = [g[i] for i in range(max((len(g) for g in seqs), default=0)) for g in seqs if i < len(g)]   # round-robin over drives
        left = []
        for iid, (px, py), props in order[:400]:
            la, ln = tile_ll(14, x, y, px / ext, py / ext)
            if country_at(la, ln, cc) != cc: continue
            left.append({"id": iid, "lat": la, "lng": ln, "pano": bool(props.get("is_pano"))})
        TILE["left"][key] = left
        if not left: TILE["empty"].add(key)
    left = TILE["left"][key]; out = []
    while left and len(out) < PER_TILE:
        c = left.pop(0)
        if c["id"] not in seen: out.append(c)
    return out, key

def mly_search(cc, lat, lng, country_at, seen):
    """Fallback when the run's map squares are used up: Mapillary's (slower) search."""
    for d in (0.004, 0.015, 0.05):
        r = get(f"{MLY}/images", {"access_token": MLY_TOKEN, "fields": "id,computed_geometry,geometry,is_pano",
                                  "bbox": f"{lng - d:.5f},{lat - d:.5f},{lng + d:.5f},{lat + d:.5f}", "limit": 20})
        if r is None or not r.ok: continue
        out = []
        for i in r.json().get("data", []):
            g = (i.get("computed_geometry") or i.get("geometry") or {}).get("coordinates")
            if not g or str(i["id"]) in seen or country_at(g[1], g[0], cc) != cc: continue
            out.append({"id": str(i["id"]), "lat": g[1], "lng": g[0], "pano": bool(i.get("is_pano"))})
        if out: random.shuffle(out); return out[:PER_TILE], f"s{round(lat, 2)}_{round(lng, 2)}"
    return [], ""

def jitter(box, W, H, lo=1.15, hi=1.6):
    x, y, w, h = box; side = min(max(w, h) * random.uniform(lo, hi), W, H)
    cx = x + w / 2 + random.uniform(-0.15, 0.15) * w; cy = y + h / 2 + random.uniform(-0.15, 0.15) * h
    return (max(0, min(W - side, cx - side / 2)), max(0, min(H - side, cy - side / 2)), side, side)

def mly_photo(c):
    """Download one Mapillary photo + its labeled objects, and decide the cut-outs.
    Returns dict(cuts=[(tag, 224 array, small jpeg or None)], ...) — the heavy work happens here, in the download threads."""
    r = get(f"{MLY}/{c['id']}", {"access_token": MLY_TOKEN, "fields": "id,thumb_1024_url,thumb_2048_url,is_pano"})
    if r is None or not r.ok: return None
    info = r.json(); pano = bool(info.get("is_pano") or c.get("pano"))
    url = (info.get("thumb_2048_url") if pano else None) or info.get("thumb_1024_url")
    if not url: return None
    img = fetch_image(url)
    if img is None: return None
    W, H = img.size; s = min(W, H); boxes = []
    if pano and W >= 1.8 * H:      # 360° photo: 4 views around the horizon, each 90° wide
        side = W / 4; y0 = max(0, H / 2 - side / 2); off = random.uniform(0, side)
        for k in range(4):
            x = (off + k * side) % W
            if x + side > W: x = W - side
            boxes.append(("scene", (x, y0, side, min(side, H))))
    else:
        boxes.append(("scene", ((W - s) / 2, (H - s) / 2, s, s)))
        z = s * random.uniform(0.7, 0.85)   # a zoomed-in second look
        boxes.append(("scene", (random.uniform(0, W - z), random.uniform(max(0, (H - z) * 0.2), (H - z) * 0.8), z, z)))
    objs, per = [], {}
    r = get(f"{MLY}/{c['id']}/detections", {"access_token": MLY_TOKEN, "fields": "value,geometry"})
    if r is not None and r.ok:
        dets = r.json().get("data", []); random.shuffle(dets)
        for d in dets:
            v = d.get("value", ""); bb = d.get("geometry") and mvt_box(d["geometry"])
            if not bb: continue
            x, y, w, h = bb[0] * W, bb[1] * H, (bb[2] - bb[0]) * W, (bb[3] - bb[1]) * H
            if v.startswith("object") or v.startswith("construction--barrier"): objs.append((x, y, w, h))
            cat = det_cat(v)
            if not cat or w < 14 or h < 14: continue
            per[cat] = per.get(cat, 0) + 1
            if per[cat] > 3: continue
            boxes.append((cat, jitter((x, y, w, h), W, H, 1.25, 1.25)))
            boxes.append((cat, jitter((x, y, w, h), W, H)))            # a second, shifted framing of the same thing
        made = 0
        for _ in range(20):
            if made >= 3: break
            side = s * random.uniform(0.15, 0.4); x, y = random.uniform(0, W - side), random.uniform(0, H - side)
            if not any(x < b[0] + b[2] and x + side > b[0] and y < b[1] + b[3] and y + side > b[1] for b in objs):
                boxes.append(("neg", (x, y, side, side))); made += 1
    if not pano:
        t = round(s * 0.35)
        for x in (0, (W - t) / 2, W - t): boxes.append(("bollard?", (x, H - t - H * 0.08, t, t)))
    cuts = []
    for tag, (x, y, w, h) in boxes:
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        keep = None
        if tag in STYLE_CATS:
            px = CROP_PX.get(tag, 160); small = crop.resize((px, px), Image.BICUBIC) if crop.size[0] > px else crop
            b = io.BytesIO(); small.save(b, "JPEG", quality=82); keep = b.getvalue()
        cuts.append((tag, prep(crop), keep))
    return {"id": c["id"], "lat": c["lat"], "lng": c["lng"], "cuts": cuts, "pano": pano}

def commons_files(cc, want):
    name, files = NAMES.get(cc, cc), []
    base = {"origin": "*", "format": "json", "action": "query", "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 640}
    for cat in (f"Vehicle registration plates of {name}", f"License plates of {name}"):
        r = get(COMMONS, dict(base, generator="categorymembers", gcmtype="file", gcmlimit=60, gcmtitle="Category:" + cat))
        if r is not None and r.ok:
            for pg in (r.json().get("query") or {}).get("pages", {}).values():
                ii = (pg.get("imageinfo") or [{}])[0]
                if "jpeg" in ii.get("mime", "") or "png" in ii.get("mime", ""): files.append(ii.get("thumburl") or ii.get("url"))
        if len(files) < want:
            r = get(COMMONS, {"origin": "*", "format": "json", "action": "query", "list": "categorymembers", "cmtype": "subcat", "cmlimit": 30, "cmtitle": "Category:" + cat})
            subs = ((r.json().get("query") or {}).get("categorymembers", []) if r is not None and r.ok else [])
            random.shuffle(subs)
            for sc in subs[:8]:
                r2 = get(COMMONS, dict(base, generator="categorymembers", gcmtype="file", gcmlimit=20, gcmtitle=sc["title"]))
                if r2 is not None and r2.ok:
                    for pg in (r2.json().get("query") or {}).get("pages", {}).values():
                        ii = (pg.get("imageinfo") or [{}])[0]
                        if "jpeg" in ii.get("mime", "") or "png" in ii.get("mime", ""): files.append(ii.get("thumburl") or ii.get("url"))
        if files: break
    random.shuffle(files)
    return [f for f in files if f][:want]

def plate_job(cc, seen):
    out = []
    for url in commons_files(cc, 12):
        if url in seen: continue
        img = fetch_image(url)
        if img is None: continue
        W, H = img.size; s = min(W, H)
        blurred = img.filter(ImageFilter.GaussianBlur(max(3, W / 90)))   # Street View blurs plates, so learn the blurred look too
        cuts = []
        for im2 in (img, blurred):
            for b in ((0, 0, W, H), jitter(((W - s) / 2, (H - s) / 2, s, s), W, H, 0.8, 1.0)):
                crop = im2.crop(tuple(int(v) for v in (b[0], b[1], b[0] + b[2], b[1] + b[3])))
                small = crop.resize((192, 192), Image.BICUBIC); bb = io.BytesIO(); small.save(bb, "JPEG", quality=82)
                cuts.append(("plate", prep(crop), bb.getvalue()))
        out.append({"url": url, "cuts": cuts, "g": f"p{abs(hash(url)) % 10**9}"})
    return out


# ---------------- the growing collection ----------------
def q8(e):
    return np.clip(np.round(e * 127 / max(1e-6, float(np.abs(e).max()))), -127, 127).astype(np.int8).tobytes()

class Store:
    """All examples in one SQLite file: label, spot, place, the base embedding, the fine-tuned embedding
    (when there is one) and a small JPEG of the cut-out (for fine-tuning and re-embedding)."""
    def __init__(self, path=None):
        self.path = path or os.path.join(CACHE, "collection.db"); os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fresh = not os.path.exists(self.path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("pragma journal_mode=wal"); self.db.execute("pragma synchronous=normal")
        self.db.execute("""create table if not exists ex (id integer primary key, cat text, y text, g text, src text, lat real, lng real,
                           e blob, e2 blob, v2 integer default 0, img blob, at real)""")
        self.db.execute("create index if not exists ex_cy on ex(cat, y)")
        self.db.execute("create index if not exists ex_src on ex(src)")
        self.cnt = {}; self.seen = set()
        if fresh: self.restore()
        for cat, y, n in self.db.execute("select cat, y, count(*) from ex group by cat, y"): self.cnt[(cat, y)] = n
        self.seen = set(s for (s,) in self.db.execute("select distinct src from ex where src is not null"))
        log("collection so far:", self.sizes())
    def sizes(self):
        out = {}
        for (c, _), n in self.cnt.items(): out[c] = out.get(c, 0) + n
        return out
    def count(self, cat, label): return self.cnt.get((cat, label), 0)
    def cap(self, cat): return MAX_PER_CHECK if cat.startswith("det_") else MAX_PLATE if cat == "plate" else MAX_PER_CLASS
    def add(self, cat, label, emb, group="", src=None, ll=None, img=None, emb2=None, ver=0):
        row = (group, src, ll[0] if ll else None, ll[1] if ll else None, q8(emb), q8(emb2) if emb2 is not None else None, ver if emb2 is not None else 0, img, time.time())
        if self.count(cat, label) >= self.cap(cat):   # full: replace a random old one so the collection keeps freshening
            j = self.db.execute("select id from ex where cat=? and y=? order by random() limit 1", (cat, label)).fetchone()
            if j: self.db.execute("update ex set g=?, src=?, lat=?, lng=?, e=?, e2=?, v2=?, img=?, at=? where id=?", row + (j[0],))
        else:
            self.db.execute("insert into ex (cat, y, g, src, lat, lng, e, e2, v2, img, at) values (?,?,?,?,?,?,?,?,?,?,?)", (cat, label) + row)
            self.cnt[(cat, label)] = self.count(cat, label) + 1
        if src: self.seen.add(src)
    def commit(self): self.db.commit()
    def arrays(self, cat, ver=0, need_ll=False):
        """Embeddings, labels, spots and places for one category; with ver, the fine-tuned embeddings of that version."""
        col, cond = ("e2", "and v2=? and e2 is not null") if ver else ("e", "")
        q = f"select {col}, y, g, lat, lng from ex where cat=? {cond}" + (" and lat is not null" if need_ll else "")
        rows = self.db.execute(q, (cat, ver) if ver else (cat,)).fetchall()
        if not rows: return None
        E = np.frombuffer(b"".join(r[0] for r in rows), np.int8).reshape(len(rows), -1).astype(np.float32)
        return E, np.array([r[1] for r in rows]), np.array([r[2] or f"r{i}" for i, r in enumerate(rows)]), np.array([[r[3] or 0, r[4] or 0] for r in rows], np.float32)
    def with_images(self, cats):
        return self.db.execute(f"select count(*) from ex where img is not null and cat in ({','.join('?' * len(cats))})", cats).fetchone()[0]
    # ---- backup of the numbers (not the pictures) to Supabase, in parts under the 50 MB file limit ----
    def backup(self):
        man = {}
        for cat in STYLE_CATS + DET_CATS:
            rows = self.db.execute("select e, y, g, src, lat, lng from ex where cat=?", (cat,)).fetchall()
            if not rows: continue
            parts, step = [], 60000
            for i in range(0, len(rows), step):
                ch = rows[i:i + step]; buf = io.BytesIO()
                np.savez_compressed(buf, e=np.frombuffer(b"".join(r[0] for r in ch), np.int8).reshape(len(ch), -1), y=np.array([r[1] for r in ch], dtype="U8"),
                                    g=np.array([r[2] or "" for r in ch], dtype="U24"), src=np.array([r[3] or "" for r in ch], dtype="U64"),
                                    ll=np.array([[r[4] if r[4] is not None else np.nan, r[5] if r[5] is not None else np.nan] for r in ch], np.float32))
                name = f"train2/{cat}.{len(parts)}.npz"; storage_put(name, buf.getvalue()); parts.append(name)
            man[cat] = parts
        storage_put("train2/manifest.json", json.dumps(man).encode(), "application/json")
        log("collection numbers backed up to Supabase:", {k: len(v) for k, v in man.items()}, "parts")
    def restore(self):
        """A brand-new cache: start from the Supabase backup (new format, else the old one) so nothing is lost."""
        man = None
        try: blob = storage_get("train2/manifest.json"); man = json.loads(blob) if blob else None
        except Exception as e: log("no new-style backup:", e)
        n = 0
        for cat in STYLE_CATS + DET_CATS:
            blobs = [storage_get(p) for p in man.get(cat, [])] if man else []
            if not man:
                local = os.path.join(CACHE, "train", f"{cat}.npz")
                blobs = [open(local, "rb").read() if os.path.exists(local) else storage_get(f"train/{cat}.npz")]
            for blob in blobs:
                if not blob: continue
                z = np.load(io.BytesIO(blob), allow_pickle=False)
                E, Y = z["e"], [str(v) for v in z["y"]]
                G = [str(v) for v in z["g"]] if "g" in z.files else [f"old{i}" for i in range(len(Y))]
                S = [str(v) or None for v in z["src"]] if "src" in z.files and len(z["src"]) == len(Y) else [None] * len(Y)
                LL = z["ll"] if "ll" in z.files else np.full((len(Y), 2), np.nan, np.float32)
                self.db.executemany("insert into ex (cat, y, g, src, lat, lng, e, at) values (?,?,?,?,?,?,?,?)",
                                    [(cat, Y[i], G[i], S[i], None if np.isnan(LL[i][0]) else float(LL[i][0]), None if np.isnan(LL[i][1]) else float(LL[i][1]), E[i].astype(np.int8).tobytes(), 0) for i in range(len(Y))])
                n += len(Y)
        self.db.commit()
        if n: log(f"restored {n} examples from the saved collection")


# ---------------- learning ----------------
def split(y, groups, test=0.2):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    idx = np.arange(len(y)); classes = np.unique(y)
    # photos from the same spot look alike, so the test photos come from spots the model never trained on
    if groups is not None and len(set(groups)) >= 10:
        return next(GroupShuffleSplit(n_splits=1, test_size=test, random_state=0).split(idx, y, groups))
    strat = y if min(np.unique(y, return_counts=True)[1]) >= 2 and math.ceil(test * len(y)) >= len(classes) else None
    return train_test_split(idx, test_size=test, random_state=0, stratify=strat)

def scores(P, cl, yte):
    order = np.argsort(-P, 1)
    return float(np.mean(cl[order[:, 0]] == yte)), float(np.mean([yte[i] in cl[order[i, :3]] for i in range(len(yte))]))

def fit_mlp(X, y, hidden, seed=0, epochs=None):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1e-3, batch_size=256, learning_rate_init=1e-3, max_iter=epochs or 200,
                        early_stopping=epochs is None, validation_fraction=0.1, n_iter_no_change=8, random_state=seed)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); clf.fit(X, y)
    return clf

def train_head(E, y, groups=None, try_mlp=True, Cs=(0.5, 2.0, 8.0)):
    """A straight-line classifier and a small neural network; keeps whichever is right more often on unseen spots."""
    from sklearn.linear_model import LogisticRegression
    labels, counts = np.unique(y, return_counts=True)
    keep = set(labels[counts >= 4])
    m = np.array([v in keep for v in y]); E, y = E[m], y[m]
    groups = np.array(groups)[m] if groups is not None else None
    classes = sorted(keep)
    if len(classes) < 2: return None
    X = E.astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    tr, te = split(y, groups)
    if len(tr) > 30000: tr = np.random.RandomState(0).choice(tr, 30000, replace=False)   # choosing is done on a sample, to save time
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        t1, t3 = scores(clf.predict_proba(Xte), clf.classes_, yte)
        if best is None or t1 > best["acc"]: best = {"kind": "lr", "C": C, "acc": t1, "acc3": t3}
    lr_acc = best["acc"]
    if try_mlp and len(ytr) >= 300:
        for hidden in (128, 256):
            try:
                clf = fit_mlp(Xtr * 10, ytr, hidden)   # ×10: inputs on a friendlier scale for the network
                t1, t3 = scores(clf.predict_proba(Xte * 10), clf.classes_, yte)
                if t1 > best["acc"] + 0.005: best = {"kind": "mlp", "hidden": hidden, "epochs": max(10, clf.n_iter_), "acc": t1, "acc3": t3}
            except Exception as e: log("network failed:", e)
    if best["kind"] == "lr":
        clf = LogisticRegression(C=best["C"], max_iter=2000, class_weight="balanced").fit(X, y)
        W, b, cl = clf.coef_, clf.intercept_, [str(c) for c in clf.classes_]
        if len(cl) == 2 and W.shape[0] == 1:   # two classes: write it as a softmax over [0, z]
            W = np.vstack([np.zeros_like(W[0]), W[0]]); b = np.array([0.0, b[0]])
        h = {"kind": "lr", "classes": cl, "W": W, "b": b}
    else:
        clf = fit_mlp(X * 10, y, best["hidden"], epochs=best["epochs"])
        W1, W2 = clf.coefs_[0].T * 10, clf.coefs_[1].T; b1, b2 = clf.intercepts_
        cl = [str(c) for c in clf.classes_]
        if len(cl) == 2 and W2.shape[0] == 1: W2 = np.vstack([np.zeros_like(W2[0]), W2[0]]); b2 = np.array([0.0, b2[0]])
        h = {"kind": "mlp", "classes": cl, "W1": W1, "b1": b1, "W": W2, "b": b2}
    h.update({"acc": round(best["acc"], 3), "acc3": round(best["acc3"], 3), "lr_acc": round(lr_acc, 3), "n": int(len(y))})
    return h

def q_rows(W):
    s = np.maximum(np.abs(W).max(1), 1e-9) / 127
    return base64.b64encode(np.round(W / s[:, None]).astype(np.int8).tobytes()).decode(), [float(f"{v:.7g}") for v in s]

def pack_head(h, src=None, extra=None):
    w, s = q_rows(h["W"])
    out = {"classes": h["classes"], "d": int(h["W1"].shape[1] if h["kind"] == "mlp" else h["W"].shape[1]), "w": w, "s": s,
           "b": [round(float(v), 5) for v in h["b"]], "acc": h["acc"], "acc3": h["acc3"], "n": h["n"], "at": datetime.now(timezone.utc).isoformat(), "by": "server"}
    if h["kind"] == "mlp":
        w1, s1 = q_rows(h["W1"])
        out.update({"kind": "mlp", "hd": int(h["W1"].shape[0]), "w1": w1, "s1": s1, "b1": [round(float(v), 5) for v in h["b1"]]})
    if src: out["src"] = src
    if extra: out.update(extra)
    return out

def predict_packed(p, X):
    """Same maths as the browser (int8 weights), to double-check what gets published."""
    def deq(w, s, r): return np.frombuffer(base64.b64decode(w), np.int8).reshape(r, -1).astype(np.float32) * np.array(s, np.float32)[:, None]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    if p.get("kind") == "mlp": X = np.maximum(0, X @ deq(p["w1"], p["s1"], p["hd"]).T + np.array(p["b1"]))
    Z = X @ deq(p["w"], p["s"], len(p["classes"])).T + np.array(p["b"])
    Z = np.exp(Z - Z.max(1, keepdims=True)); return Z / Z.sum(1, keepdims=True)

def area_cells(ll, y, places):
    """Split each country's photos into areas (k-means on where they were taken), named after the biggest town in each."""
    from sklearn.cluster import KMeans
    lab = np.array(y, dtype=object); meta = {}
    for cc in np.unique(y):
        m = np.where(y == cc)[0]; pts = ll[m]
        k = int(min(6, len(m) // 100))
        if k < 2:
            for i in m: lab[i] = f"{cc}.0"
            c = pts.mean(0) if len(pts) else np.zeros(2); meta[f"{cc}.0"] = [round(float(c[0]), 3), round(float(c[1]), 3), ""]
            continue
        P = np.stack([pts[:, 0], pts[:, 1] * np.cos(np.radians(pts[:, 0]))], 1)   # roughly equal-area
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(P)
        towns = [t for t in places.get(cc, []) if t[2] >= 20000]
        best = {}
        for t in towns:
            tp = np.array([t[0], t[1] * math.cos(math.radians(t[0]))])
            j = int(np.argmin(((km.cluster_centers_ - tp) ** 2).sum(1)))
            if t[2] > best.get(j, (0, ""))[0]: best[j] = (t[2], t[3])
        for j in range(k):
            mm = m[km.labels_ == j]
            for i in mm: lab[i] = f"{cc}.{j}"
            c = pts[km.labels_ == j].mean(0)
            meta[f"{cc}.{j}"] = [round(float(c[0]), 3), round(float(c[1]), 3), best.get(j, (0, ""))[1]]
    return lab.astype(str), meta

def learn(store, ver, places, deadline=None):
    """Retrain everything on the whole collection and publish it."""
    summary, detail = [], {}
    jobs = STYLE_CATS + ["area"] + DET_CATS
    for cat in jobs:
        if deadline and time.time() > deadline: summary.append((cat, "skipped (out of time)", 0)); continue
        src_cat = "scene" if cat == "area" else cat
        use = 0
        if ver and cat in STYLE_CATS + ["area"]:
            a2 = store.arrays(src_cat, ver, need_ll=cat == "area")
            if a2 is not None and len(a2[1]) >= 200: use = ver
        A = store.arrays(src_cat, use, need_ll=cat == "area")
        if A is None or len(A[1]) < 12: summary.append((cat, "not enough examples yet", 0 if A is None else len(A[1]))); continue
        E, y, g, ll = A
        extra = None
        if cat == "area":
            if len(y) < 300: summary.append((cat, "not enough places yet", len(y))); continue
            y, cells = area_cells(ll, y, places); extra = {"cells": cells}
        try: h = train_head(E, y, g, try_mlp=not cat.startswith("det_"), Cs=(2.0,) if cat == "area" else (0.5, 2.0, 8.0))
        except Exception as e: summary.append((cat, f"couldn't learn yet ({e})", len(y))); log(cat, "failed:", e); traceback.print_exc(limit=2); continue
        if not h: summary.append((cat, "needs at least 2 groups", len(y))); continue
        packed = pack_head(h, src=f"b{use}" if use else None, extra=extra)
        try: save_head(cat, packed)
        except Exception as e: gh_error(f"Saving {cat} failed: {e}"); summary.append((cat, f"save failed: {e}", h["n"])); continue
        kind = "network" if h["kind"] == "mlp" else "line"
        summary.append((cat, f"{len(h['classes'])} groups, {round(h['acc'] * 100)}% right, {round(h['acc3'] * 100)}% top-3 ({kind}{', tuned layers' if use else ''})", h["n"]))
        detail[cat] = {"acc": h["acc"], "acc3": h["acc3"], "n": h["n"], "classes": len(h["classes"]), "kind": h["kind"], "lr_acc": h["lr_acc"], "tuned": bool(use)}
        log("saved", cat, summary[-1][1])
    return summary, detail


# ---------------- fine-tuning the last layers of the image model (weekly) ----------------
def ensure_torch():
    try: import torch; return torch
    except ImportError: pass
    log("installing PyTorch (CPU) for fine-tuning…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx==1.17.0"])   # for making the int8 copy
    import torch; return torch

def finetune(store, model, places):
    """Teach the last 2 layers what tells countries apart, keeping them close to the original (so nothing else breaks).
    Publishes only if a fresh classifier on the new embeddings beats one on the current embeddings, on unseen spots."""
    torch = ensure_torch(); torch.set_num_threads(os.cpu_count() or 2)
    sys.path.insert(0, HERE)
    from branch import Branch
    t_start = time.time()
    # 1) a balanced sample of cut-outs with pictures
    per = max(40, FT_MAX // max(1, len([1 for (c, _), n in store.cnt.items() if c in STYLE_CATS and n >= 8])))
    rows = []
    for cat in STYLE_CATS:
        for (c, label), n in store.cnt.items():
            if c != cat or n < 8: continue
            rows += store.db.execute("select id, cat, y, g, img from ex where cat=? and y=? and img is not null order by random() limit ?", (cat, label, per)).fetchall()
    rows = [r for r in rows if r[4]]
    if len(rows) < max(500, FT_MIN // 20): return {"skipped": "not enough cut-outs with pictures", "n": len(rows)}
    random.shuffle(rows); rows = rows[:FT_MAX]
    cats = sorted(set(r[1] for r in rows)); labels = {c: sorted(set(r[2] for r in rows if r[1] == c)) for c in cats}
    N = len(rows); log(f"fine-tuning on {N} cut-outs, {len(cats)} kinds")
    report("running", "fine-tuning: reading cut-outs", examples=store.sizes())
    # 2) the frozen part of the model, once for every cut-out
    tok = np.zeros((N, 50, 512), np.float16); base = np.zeros((N, 512), np.float32)
    keep_branch, model.branch = model.branch, None
    for i in range(0, N, 256):
        arr = [prep(Image.open(io.BytesIO(r[4])).convert("RGB")) for r in rows[i:i + 256]]
        e, _, _, t = model.run(arr, want_tok=True); tok[i:i + len(arr)] = t; base[i:i + len(arr)] = e
        if i % 5120 == 0: log(f"  frozen part: {i}/{N}")
    model.branch = keep_branch
    groups = np.array([r[3] or f"r{r[0]}" for r in rows]); y_all = np.array([r[2] for r in rows]); c_all = np.array([r[1] for r in rows])
    from sklearn.model_selection import GroupShuffleSplit
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=1).split(np.arange(N), y_all, groups))
    # 3) train
    br = Branch(); br.load_np(dict(np.load(os.path.join(HERE, "models", "branch_init.npz"))))
    heads = torch.nn.ModuleDict({c: torch.nn.Linear(512, len(labels[c])) for c in cats})
    yi = np.array([labels[c_all[i]].index(y_all[i]) for i in range(N)])
    opt = torch.optim.AdamW([{"params": br.parameters(), "lr": 3e-5, "weight_decay": 0.05}, {"params": heads.parameters(), "lr": 2e-3, "weight_decay": 1e-4}])
    T = torch.from_numpy(tok); B = torch.from_numpy(base); Y = torch.from_numpy(yi)
    budget = time.time() + FT_MINUTES * 60; step = 0; best = (-1, None); epoch = 0   # the clock starts after the preparation
    def evaluate():
        br.eval(); out = np.zeros((len(va), 512), np.float32)
        with torch.no_grad():
            for i in range(0, len(va), 512): out[i:i + 512] = br(T[va[i:i + 512]].float()).numpy()
        br.train(); return out
    while epoch == 0 or (time.time() < budget and epoch < 12):
        epoch += 1; perm = np.random.permutation(tr); br.train()
        for i in range(0, len(perm), 96):
            b = perm[i:i + 96]; x = T[b].float(); e = br(x)
            loss = 0.5 * (1 - (e * B[b]).sum(1)).mean() * 10        # stay close to the original embedding
            for c in set(c_all[b]):
                m = np.where(c_all[b] == c)[0]
                loss = loss + torch.nn.functional.cross_entropy(heads[c](e[m] * 20), Y[b][m]) * len(m) / len(b)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if time.time() > budget: break
        E2 = evaluate(); right = 0
        with torch.no_grad():
            for c in cats:
                m = np.where(c_all[va] == c)[0]
                if len(m): right += int((heads[c](torch.from_numpy(E2[m] * 20)).argmax(1).numpy() == yi[va[m]]).sum())
        acc = right / max(1, len(va))
        log(f"  epoch {epoch}: {round(acc * 100, 1)}% right on unseen spots")
        report("running", f"fine-tuning: round {epoch}", examples=store.sizes())
        if acc > best[0]: best = (acc, {k: v.detach().clone() for k, v in br.state_dict().items()})
    br.load_state_dict(best[1]); br.eval()
    # 4) fair comparison: fresh straight-line classifiers on current vs new embeddings, per kind
    with torch.no_grad():
        E2 = np.zeros((N, 512), np.float32)
        for i in range(0, N, 512): E2[i:i + 512] = br(T[i:i + 512].float()).numpy()
    cur = base
    if keep_branch is not None:   # the layers the players have now
        cur = np.concatenate([keep_branch.run(None, {"tok": tok[i:i + 256].astype(np.float32)})[0] for i in range(0, N, 256)])
    from sklearn.linear_model import LogisticRegression
    gains, wsum = [], 0
    for c in cats:
        m_tr = tr[c_all[tr] == c]; m_va = va[c_all[va] == c]
        if len(set(y_all[m_tr])) < 2 or len(m_va) < 20: continue
        acc = []
        for E in (cur, E2):
            clf = LogisticRegression(C=2.0, max_iter=1500, class_weight="balanced").fit(E[m_tr], y_all[m_tr])
            acc.append(float(np.mean(clf.predict(E[m_va]) == y_all[m_va])))
        w = math.sqrt(len(m_va)); gains.append((acc[1] - acc[0]) * w); wsum += w
        log(f"  {c}: {round(acc[0] * 100, 1)}% → {round(acc[1] * 100, 1)}%")
    gain = sum(gains) / max(1e-9, wsum)
    res = {"gain": round(gain, 4), "epochs": epoch, "n": N, "minutes": round((time.time() - t_start) / 60, 1)}
    if gain < FT_GAIN:
        log(f"fine-tuned layers not better enough ({round(gain * 100, 1)} points); keeping the current ones"); res["published"] = False; return res
    # 5) publish: small int8 ONNX file, then re-embed every cut-out that has a picture
    ver = (model.ver or 0) + 1
    path = os.path.join(CACHE, f"branch-v{ver}.onnx"); fp = path + ".fp32"
    import inspect
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}   # the classic exporter on every torch version
    torch.onnx.export(br, torch.from_numpy(tok[:2].astype(np.float32)), fp, input_names=["tok"], output_names=["emb2"],
                      dynamic_axes={"tok": {0: "n"}, "emb2": {0: "n"}}, opset_version=17, **kw)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(fp, path, op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt8)
    blob = open(path, "rb").read()
    model.load_branch(blob, ver)
    chk = model.branch.run(None, {"tok": tok[:64].astype(np.float32)})[0]
    log("int8 copy agrees with the trained layers:", round(float(np.mean(np.sum(chk * E2[:64], 1))), 4), "(cosine)")
    storage_put(f"branch-v{ver}.onnx", blob)
    report("running", "fine-tuning: updating the collection", examples=store.sizes())
    ids = [r[0] for r in store.db.execute("select id from ex where img is not null and cat in (%s)" % ",".join("?" * len(STYLE_CATS)), STYLE_CATS)]
    done = 0
    for i in range(0, len(ids), 256):
        chunk = store.db.execute("select id, img from ex where id in (%s)" % ",".join("?" * len(ids[i:i + 256])), ids[i:i + 256]).fetchall()
        arr = [prep(Image.open(io.BytesIO(r[1])).convert("RGB")) for r in chunk]
        _, _, e2, _ = model.run(arr)
        store.db.executemany("update ex set e2=?, v2=? where id=?", [(q8(e2[k]), ver, chunk[k][0]) for k in range(len(chunk))])
        done += len(chunk)
        if done % 10240 < 256: log(f"  re-embedded {done}/{len(ids)}"); store.commit()
    store.commit()
    res.update({"published": True, "ver": ver})
    return res


# ---------------- main ----------------
def main():
    if SEED: random.seed(int(SEED)); np.random.seed(int(SEED))
    for k, v in {"SUPABASE_URL": SB_URL, "SUPABASE_SERVICE_ROLE_KEY": SB_KEY, "MAPILLARY_TOKEN": MLY_TOKEN}.items():
        if not v: raise SystemExit(f"Missing secret {k}. Add it under GitHub → Settings → Secrets and variables → Actions.")
    meta = json.load(open(os.path.join(HERE, "meta.json")))
    # the players download the image model from Supabase; keep it in step with this repo
    size = public_size(BASE_NAME)   # None = not there, -1 = couldn't check
    if size is None or (size != -1 and size != os.path.getsize(BASE_PATH)):
        log("uploading the image model for the players…"); storage_put(BASE_NAME, open(BASE_PATH, "rb").read()); gh_note("Uploaded the image model to Supabase Storage.")
    model = Model(meta)
    tuned = read_head("model") or {}
    if tuned.get("ver"):
        blob = cached_download(f"{SB_URL}/storage/v1/object/public/wp-models/branch-v{tuned['ver']}.onnx", f"branch-v{tuned['ver']}.onnx")
        if blob: model.load_branch(blob, tuned["ver"]); log(f"using fine-tuned layers v{tuned['ver']}")
    log("getting cities and borders…")
    places = decode_places(cached_download(PLACES_URL, "cities.pbf"))
    country_at, random_point = load_borders(json.loads(cached_download(BORDERS_URL, "borders.geojson")))
    store = Store()
    try: TILE["empty"] = set(json.load(open(os.path.join(CACHE, "empty_tiles.json"))))
    except Exception: pass
    run_started = datetime.now(timezone.utc).isoformat()
    report("running", "starting", at=run_started, photos=0, plates=0, visits=0, examples=store.sizes())

    # once a week: fine-tune the image model's last layers instead of collecting
    st = state_get(); ft_res = None
    due = (time.time() - st.get("ft_at", 0)) > FT_EVERY_DAYS * 86400
    have = store.with_images(STYLE_CATS)
    # a fine-tune can also be asked for by committing a new line to finetune_request.txt
    req = ""
    try: req = open(os.path.join(HERE, "finetune_request.txt")).read().strip()
    except OSError: pass
    asked = bool(req) and req != st.get("ft_req")
    if FORCE_FT or asked or (due and have >= FT_MIN):
        state_set(ft_at=time.time(), ft_req=req)
        try:
            ft_res = finetune(store, model, places); log("fine-tuning:", ft_res)
        except Exception as e:
            gh_error(f"Fine-tuning failed: {e}"); traceback.print_exc(); ft_res = {"error": str(e)[:200]}
        if ft_res.get("error") or ft_res.get("skipped"): state_set(ft_at=time.time() - (FT_EVERY_DAYS - 1) * 86400)   # try again tomorrow
        elif ft_res.get("published") is False and ft_res.get("gain", 0) > 0: state_set(ft_at=time.time() - max(0, FT_EVERY_DAYS - 3) * 86400)   # close: try again in 3 days, with more photos
        report("running", "learning", examples=store.sizes())
        summary, detail = learn(store, model.ver, places)   # heads first (on the new layers, if published)…
        if ft_res.get("published"):                          # …then tell the players to switch
            save_head("model", {"ver": model.ver, "url": f"branch-v{model.ver}.onnx", "gain": ft_res["gain"], "at": datetime.now(timezone.utc).isoformat(),
                                "n": ft_res["n"], "base": BASE_NAME})
            gh_note(f"Published fine-tuned layers v{model.ver} (+{round(ft_res['gain'] * 100, 1)} points).")
        finish(store, run_started, 0, {}, summary, detail, ft_res)
        return

    deadline = T0 + MINUTES * 60
    stats = {cc: {"photos": 0, "plates": 0} for cc in COVERAGE}
    pool = ThreadPoolExecutor(THREADS)
    inflight, busy = {}, {}
    visits = 0; last_report = last_commit = time.time()
    plate_seen = set(store.seen)

    def need(cc): return store.count("scene", cc) + store.count("pole", cc) + store.count("sign", cc) + 30 * busy.get(cc, 0)
    def visit(cc):
        """One trip: up to 4 map squares around towns or random spots, PER_TILE photos from each."""
        got = []
        for _ in range(4):
            towns = places.get(cc, [])
            if towns and random.random() < 0.6:
                lat, lng = (random.choice(towns) if random.random() < 0.5 else random.choices(towns, weights=[math.sqrt(t[2] + 1) for t in towns])[0])[:2]
                lat += random.uniform(-0.02, 0.02); lng += random.uniform(-0.02, 0.02)
            else:
                pt = random_point(cc)
                if not pt: continue
                lat, lng = pt
            cands, tile = tile_candidates(cc, lat, lng, country_at, store.seen)
            if cands is None: cands, tile = mly_search(cc, lat, lng, country_at, store.seen)
            for c in cands:
                try:
                    res = mly_photo(c)
                    if res: res["g"] = f"{cc}{tile}"; got.append(res)
                except TokenError: raise
                except Exception as e: log(f"{cc}: a photo failed ({e})")
            if len(got) >= PER_TILE: break
        return got

    while True:
        now = time.time()
        while now < deadline and len(inflight) < THREADS:
            order = sorted(COVERAGE, key=lambda c: (need(c), random.random()))
            cc = order[0] if random.random() < 0.7 else random.choice(order[:15])
            if store.count("plate", cc) < MAX_PLATE // 4 and not any(k == ("plate", cc) for k in inflight.values()) and random.random() < 0.3:
                inflight[pool.submit(plate_job, cc, plate_seen)] = ("plate", cc)
            else:
                inflight[pool.submit(visit, cc)] = ("visit", cc)
            busy[cc] = busy.get(cc, 0) + 1
        if not inflight: break
        done, _ = wait(list(inflight), timeout=30, return_when=FIRST_COMPLETED)
        for fut in done:
            kind, cc = inflight.pop(fut); busy[cc] -= 1
            try: res = fut.result()
            except TokenError: raise
            except Exception as e: log(f"{cc}: skipped ({e})"); continue
            if kind == "plate":
                for p in res:
                    e, _, e2, _ = model.run([c[1] for c in p["cuts"]])
                    for k, (tag, _, jpg) in enumerate(p["cuts"]):
                        store.add("plate", cc, e[k], group=p["g"], img=jpg, src=p["url"] if k == 0 else None, emb2=None if e2 is None else e2[k], ver=model.ver)
                    plate_seen.add(p["url"]); store.seen.add(p["url"]); stats[cc]["plates"] += 1
                continue
            visits += 1
            for ph in res:
                cuts = ph["cuts"]
                e, pb, e2, _ = model.run([c[1] for c in cuts])
                ll, g = (ph["lat"], ph["lng"]), ph["g"]
                first_scene = True
                for k, (tag, _, jpg) in enumerate(cuts):
                    e2k = None if e2 is None else e2[k]
                    if tag == "bollard?":
                        if pb[k] >= 0.6: store.add("bollard", cc, e[k], group=g, ll=ll, emb2=e2k, ver=model.ver); store.add("det_bollard", "yes", e[k], group=g)
                    elif tag == "neg":
                        for dc in DET_CATS: store.add(dc, "no", e[k], group=g)
                    elif tag == "scene":
                        store.add("scene", cc, e[k], group=g, src=ph["id"] if first_scene else None, ll=ll, img=jpg, emb2=e2k, ver=model.ver); first_scene = False
                    else:
                        store.add(tag, cc, e[k], group=g, ll=ll, img=jpg, emb2=e2k, ver=model.ver); store.add("det_" + tag, "yes", e[k], group=g)
                stats[cc]["photos"] += 1
        now = time.time()
        if now - last_commit > 120: store.commit(); last_commit = now
        if now - last_report > 120:   # every 2 minutes: tell the game how it's going
            last_report = now
            p = sum(v["photos"] for v in stats.values())
            report("running", "collecting", photos=p, plates=sum(v["plates"] for v in stats.values()), visits=visits, examples=store.sizes())
            log(f"{visits} visits, {p} photos ({round(p / max(1, (now - T0) / 60), 1)}/min); collection: {store.sizes()}")
        if now >= deadline and inflight:
            # stop starting new trips; give the running ones a little while
            if now > deadline + 90:
                for f in inflight: f.cancel()
                break
    pool.shutdown(wait=False, cancel_futures=True)
    store.commit()
    try: json.dump(sorted(TILE["empty"])[-300000:], open(os.path.join(CACHE, "empty_tiles.json"), "w"))
    except Exception as e: log("couldn't save empty squares:", e)
    log(f"map squares: {TILE['used']} downloaded this run, {len(TILE['empty'])} known empty")
    if BACKUP:
        try: store.backup()
        except Exception as e: gh_error(f"Backing up the collection failed: {e}")
    report("running", "learning", photos=sum(v["photos"] for v in stats.values()), plates=sum(v["plates"] for v in stats.values()), visits=visits, examples=store.sizes())
    summary, detail = learn(store, model.ver, places)
    finish(store, run_started, visits, stats, summary, detail, None)

def finish(store, run_started, visits, stats, summary, detail, ft_res):
    photos = sum(v["photos"] for v in stats.values()); plates = sum(v["plates"] for v in stats.values())
    run = {"at": datetime.now(timezone.utc).isoformat(), "minutes": round((time.time() - T0) / 60, 1), "visits": visits,
           "photos": photos, "plates": plates, "heads": {c: s for c, s, _ in summary}, "examples": store.sizes(), "with_pictures": store.with_images(STYLE_CATS)}
    if ft_res: run["finetune"] = ft_res; detail = dict(detail, finetune=ft_res)
    detail = dict(detail, _pictures=run["with_pictures"])   # shown on the progress page (fine-tuning starts at FT_MIN)
    with open(os.path.join(HERE, "stats.json"), "w") as f: json.dump(run, f, indent=1)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"## Wanderpin training run\n{photos} street photos and {plates} plate photos in {run['minutes']} min.\n\n")
            if ft_res: f.write(f"Fine-tuning: `{json.dumps(ft_res)}`\n\n")
            f.write("| What | Examples | Result |\n|---|---|---|\n")
            for c, s, n in summary: f.write(f"| {c} | {n} | {s} |\n")
    log("done:", json.dumps(run))
    try: store.db.execute("pragma wal_checkpoint(truncate)"); store.db.close()   # one tidy file for GitHub's cache
    except Exception as e: log("closing the collection:", e)
    # the finished row for the progress page in the game
    report("done", "done", at=run_started, visits=visits, photos=photos, plates=plates, heads=detail, examples=store.sizes())
    bad = [c for c, s, _ in summary if s.startswith("save failed")]
    if bad and len(bad) == len([1 for _, s, _ in summary if not s.startswith(("not enough", "needs", "couldn", "skipped"))]): raise SystemExit("Nothing could be saved to Supabase (see errors above).")

if __name__ == "__main__":
    try: main()
    except SystemExit as e:
        if e.code not in (None, 0): gh_error(e.code)
        raise
    except Exception as e:
        gh_error(f"{type(e).__name__}: {e} @ " + " < ".join(f"{f.name}:{f.lineno}" for f in traceback.extract_tb(e.__traceback__)[-3:]))
        raise
