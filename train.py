"""Wanderpin scanner trainer — runs on GitHub's servers every night.

Each run:
  1. downloads the scanner's image model from your Supabase Storage
  2. pulls fresh street photos from Mapillary (CC BY-SA) and plate photos from Wikimedia Commons,
     favouring the countries with the fewest examples so far
  3. cuts out poles, signs, guardrails, street lamps, road posts and plates, and turns each into
     the model's 512-number "embedding"
  4. adds them to the growing collection kept in Supabase Storage (wp-models/train/*.npz)
  5. retrains every country-style classifier and every "is it really there?" check on all of it
  6. saves the results to the wp_scan_heads table, which every player's scanner reads

Only numbers are kept, never the photos.
Secrets (GitHub → Settings → Secrets and variables → Actions):
  MAPILLARY_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""
import base64, io, json, math, os, random, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import requests
from PIL import Image, ImageFilter

# ---------------- settings ----------------
SB_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
MLY_TOKEN = os.environ.get("MAPILLARY_TOKEN", "")
MINUTES = float(os.environ.get("MINUTES") or 40)          # time spent collecting photos each run
PER_VISIT = int(os.environ.get("PER_VISIT") or 6)          # Mapillary photos per country visit
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS") or 600)  # cap per country per style (keeps files small and fair)
MAX_PER_CHECK = int(os.environ.get("MAX_PER_CHECK") or 4000) # cap per yes/no answer for the "is it really there?" checks
SEED = os.environ.get("SEED")
MLY = os.environ.get("MLY_BASE", "https://graph.mapillary.com")
COMMONS = os.environ.get("COMMONS_API", "https://commons.wikimedia.org/w/api.php")
MODEL_URL = os.environ.get("MODEL_URL") or f"{SB_URL}/storage/v1/object/public/wp-models/wanderpin-vision.onnx"
PLACES_URL = os.environ.get("PLACES_URL", "https://cdn.jsdelivr.net/npm/all-the-cities@3.1.0/cities.pbf")
BORDERS_URL = os.environ.get("BORDERS_URL", "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_50m_admin_0_countries.geojson")
UA = {"User-Agent": "WanderpinTrainer/1.0 (https://github.com/roostercodes/wanderpin-trainer)"}

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
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)
def gh_error(msg): print("::error::" + str(msg).replace("\n", " | ")[:900], flush=True)   # shows on the run page (and to the API)
http = requests.Session(); http.headers.update(UA)


# ---------------- Supabase ----------------
def sb_headers(extra=None):
    # new sb_secret_ keys go only on the apikey header; old service_role keys (eyJ…) also go on Authorization
    h = {"apikey": SB_KEY} if SB_KEY.startswith("sb_") else {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"}
    h.update(extra or {}); return h
def storage_get(path):
    r = http.get(f"{SB_URL}/storage/v1/object/wp-models/{path}", headers=sb_headers(), timeout=120)
    if r.status_code in (400, 404): return None
    r.raise_for_status(); return r.content
def storage_put(path, data, ctype="application/octet-stream"):
    r = http.post(f"{SB_URL}/storage/v1/object/wp-models/{path}", headers=sb_headers({"content-type": ctype, "x-upsert": "true"}), data=data, timeout=300)
    if not r.ok: raise RuntimeError(f"upload {path} failed: {r.status_code} {r.text[:200]}")
def save_head(hid, data):
    r = http.post(f"{SB_URL}/rest/v1/wp_scan_heads", headers=sb_headers({"content-type": "application/json", "Prefer": "resolution=merge-duplicates"}),
                  data=json.dumps([{"id": hid, "data": data, "updated_at": datetime.now(timezone.utc).isoformat()}]), timeout=60)
    if not r.ok: raise RuntimeError(f"saving {hid} failed: {r.status_code} {r.text[:200]} (did you run the newest setup SQL?)")


# ---------------- the model ----------------
class Model:
    def __init__(self, blob, meta):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
        self.s = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
        outs = [o.name for o in self.s.get_outputs()]
        if len(outs) < 2: raise SystemExit("The model in Supabase Storage is the old version. Upload the newest wanderpin-vision.onnx to wp-models.")
        self.inp, self.outs, self.meta = self.s.get_inputs()[0].name, outs, meta
        self.boll = next(c for c in meta["concepts"] if c["id"] == "bollard")
    @staticmethod
    def prep(img, box):
        x, y, w, h = box
        c = img.crop((int(x), int(y), int(x + w), int(y + h))).resize((224, 224), Image.BICUBIC)
        a = (np.asarray(c, np.float32) / 255 - MEAN) / STD
        return a.transpose(2, 0, 1)
    def run(self, items):
        """items: list of (PIL image, box). Returns (embeddings N×512, bollard probability N)."""
        embs, boll = [], []
        for i in range(0, len(items), 32):
            X = np.stack([self.prep(im, b) for im, b in items[i:i + 32]])
            logits, emb = self.s.run(self.outs, {self.inp: X})
            embs.append(emb)
            lg = [np.max(logits[:, c["p"]], axis=1) for c in self.meta["concepts"]] + [logits[:, p] for p in self.meta["bg"]]
            L = np.stack(lg, 1); L = np.exp(L - L.max(1, keepdims=True)); L /= L.sum(1, keepdims=True)
            boll.append(L[:, self.meta["concepts"].index(self.boll)])
        return np.concatenate(embs), np.concatenate(boll)


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
        end = vint() + p; cc, pop = "", 0
        while p < end:
            key = vint(); tag, wt = key >> 3, key & 7
            if tag == 10: last_lon += svint()
            elif tag == 11: last_lat += svint()
            elif tag == 9: pop = vint()
            elif wt == 0: vint()
            elif wt == 2:
                l = vint()
                if tag == 3: cc = bytes(buf[p:p + min(l, 2)]).decode("ascii", "ignore")
                p += l
            elif wt == 5: p += 4
            elif wt == 1: p += 8
            else: p = end
        p = end
        if cc in want: out.setdefault(cc, []).append((last_lat / 1e5, last_lon / 1e5, pop))
    return out

def load_borders(data):
    from shapely.geometry import shape, Point
    from shapely.prepared import prep
    shapes = {}
    for f in data["features"]:
        a2 = f["properties"].get("ISO_A2_EH") or f["properties"].get("ISO_A2")
        if a2 in COVERAGE: shapes[a2] = shape(f["geometry"])
    prepped = {k: prep(v) for k, v in shapes.items()}
    def country_at(lat, lng):
        pt = Point(lng, lat)
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


# ---------------- Mapillary detection shapes (vector tile) ----------------
def mvt_box(b64):
    u = base64.b64decode(b64); xs, ys = [], []; extent = 4096
    def varint(o):
        v = s = 0
        while True:
            b = u[o[0]]; o[0] += 1; v |= (b & 0x7f) << s; s += 7
            if not b & 0x80: return v
    def walk(start, end, on):
        o = [start]
        while o[0] < end:
            key = varint(o); f, wt = key >> 3, key & 7
            if wt == 0: on(f, wt, varint(o))
            elif wt == 2:
                l = varint(o); s = o[0]; o[0] += l; on(f, wt, (s, s + l))
            elif wt == 5: o[0] += 4
            elif wt == 1: o[0] += 8
            else: break
    def feature(f, wt, v):
        if f != 4 or wt != 2: return
        o = [v[0]]; x = y = 0
        while o[0] < v[1]:
            ci = varint(o); cid, cnt = ci & 7, ci >> 3
            if cid == 7: continue
            for _ in range(cnt):
                dx, dy = varint(o), varint(o)
                x += -((dx + 1) >> 1) if dx & 1 else dx >> 1
                y += -((dy + 1) >> 1) if dy & 1 else dy >> 1
                xs.append(x); ys.append(y)
    def layer(f, wt, v):
        nonlocal extent
        if f == 5 and wt == 0: extent = v
        if f == 2 and wt == 2: walk(v[0], v[1], feature)
    walk(0, len(u), lambda f, wt, v: walk(v[0], v[1], layer) if f == 3 and wt == 2 else None)
    if not xs: return None
    return min(xs) / extent, min(ys) / extent, max(xs) / extent, max(ys) / extent

def det_cat(v):
    if "utility-pole" in v: return "pole"
    if "traffic-sign" in v and "back" not in v: return "sign"
    if "guard-rail" in v: return "guardrail"
    if "street-light" in v: return "lamp"
    if "bollard" in v or "delineator" in v: return "bollard"
    return None

def get(url, params=None, tries=3, timeout=30):
    for k in range(tries):
        try:
            r = http.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (k + 1)); continue
            return r
        except requests.RequestException:
            time.sleep(1.5 * (k + 1))
    return None

def mly_search(lat, lng):
    # Mapillary's search times out on big boxes in busy places, so start small and widen only if empty
    for d in (0.004, 0.015, 0.05):
        r = get(f"{MLY}/images", {"access_token": MLY_TOKEN, "fields": "id,thumb_1024_url,computed_geometry,geometry",
                                  "bbox": f"{lng - d:.5f},{lat - d:.5f},{lng + d:.5f},{lat + d:.5f}", "limit": 20})
        if r is None: continue
        if r.status_code in (401, 403): raise SystemExit("Mapillary refused MAPILLARY_TOKEN. Copy the Client Token (MLY|…) again into the GitHub secret.")
        if r.ok:
            data = [i for i in r.json().get("data", []) if i.get("thumb_1024_url")]
            if data: return data
    return []

def fetch_image(url):
    r = get(url, timeout=40)
    if r is None or not r.ok: return None
    try: return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception: return None

def mly_photo(im):
    """Download one Mapillary photo + its labeled objects, and decide the crops."""
    img = fetch_image(im["thumb_1024_url"])
    if img is None: return None
    W, H = img.size; s = min(W, H)
    crops = [("scene", ((W - s) / 2, (H - s) / 2, s, s))]
    objs, per = [], {}
    r = get(f"{MLY}/{im['id']}/detections", {"access_token": MLY_TOKEN, "fields": "value,geometry"})
    if r is not None and r.ok:
        for d in r.json().get("data", []):
            v = d.get("value", ""); bb = d.get("geometry") and mvt_box(d["geometry"])
            if not bb: continue
            x, y, w, h = bb[0] * W, bb[1] * H, (bb[2] - bb[0]) * W, (bb[3] - bb[1]) * H
            if v.startswith("object") or v.startswith("construction--barrier"): objs.append((x, y, w, h))
            cat = det_cat(v)
            if not cat or w < 14 or h < 14: continue
            per[cat] = per.get(cat, 0) + 1
            if per[cat] > 2: continue
            side = min(max(w, h) * 1.25, W, H); cx, cy = x + w / 2, y + h / 2
            crops.append((cat, (max(0, min(W - side, cx - side / 2)), max(0, min(H - side, cy - side / 2)), side, side)))
        made = 0
        for _ in range(20):
            if made >= 3: break
            side = s * random.uniform(0.15, 0.4); x, y = random.uniform(0, W - side), random.uniform(0, H - side)
            if not any(x < b[0] + b[2] and x + side > b[0] and y < b[1] + b[3] and y + side > b[1] for b in objs):
                crops.append(("neg", (x, y, side, side))); made += 1
    t = round(s * 0.35)
    for x in (0, (W - t) / 2, W - t): crops.append(("bollard?", (x, H - t - H * 0.08, t, t)))
    return img, crops

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


# ---------------- the growing collection ----------------
class Store:
    def __init__(self):
        self.data = {}   # cat -> dict(e: list[int8 arrays], y: list[str], src: set)
        self.cnt = {}    # (cat, label) -> how many
    def load(self):
        for cat in STYLE_CATS + DET_CATS:
            blob = storage_get(f"train/{cat}.npz")
            d = {"e": [], "y": [], "g": [], "src": set()}
            if blob:
                z = np.load(io.BytesIO(blob), allow_pickle=False)
                d["e"] = list(z["e"]); d["y"] = [str(v) for v in z["y"]]; d["src"] = set(str(v) for v in z["src"])
                d["g"] = [str(v) for v in z["g"]] if "g" in z.files else [f"old{i}" for i in range(len(d["y"]))]
            self.data[cat] = d
            for y in d["y"]: self.cnt[(cat, y)] = self.cnt.get((cat, y), 0) + 1
        log("collection so far:", {k: len(v["y"]) for k, v in self.data.items()})
    def count(self, cat, label): return self.cnt.get((cat, label), 0)
    def add(self, cat, label, emb, src=None, group=""):
        d = self.data[cat]
        q = np.clip(np.round(emb * 127 / max(1e-6, float(np.abs(emb).max()))), -127, 127).astype(np.int8)
        cap = MAX_PER_CHECK if cat.startswith("det_") else MAX_PER_CLASS
        if self.count(cat, label) >= cap:   # full: replace a random old one so the collection keeps freshening
            idx = [i for i, y in enumerate(d["y"]) if y == label]
            j = random.choice(idx); d["e"][j] = q; d["g"][j] = group; return
        d["e"].append(q); d["y"].append(label); d["g"].append(group); self.cnt[(cat, label)] = self.count(cat, label) + 1
        if src: d["src"].add(src)
    def save(self):
        for cat, d in self.data.items():
            if not d["y"]: continue
            buf = io.BytesIO()
            np.savez_compressed(buf, e=np.stack(d["e"]), y=np.array(d["y"], dtype="U4"), g=np.array(d["g"], dtype="U24"), src=np.array(sorted(d["src"])[-200000:], dtype="U64"))
            storage_put(f"train/{cat}.npz", buf.getvalue())
        log("collection saved:", {k: len(v["y"]) for k, v in self.data.items()})


# ---------------- learning ----------------
def train_head(E, y, groups=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    labels, counts = np.unique(y, return_counts=True)
    keep = set(labels[counts >= 4])
    m = np.array([v in keep for v in y]); E, y = E[m], y[m]
    groups = np.array(groups)[m] if groups is not None else None
    classes = sorted(keep)
    if len(classes) < 2: return None
    X = E.astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    # honest score: hold back 20% the model never sees, try a few strengths, keep the best
    # photos from the same spot look alike, so the test photos come from spots the model never trained on
    if groups is not None and len(set(groups)) >= 10:
        from sklearn.model_selection import GroupShuffleSplit
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    else:
        strat = y if min(np.unique(y, return_counts=True)[1]) >= 2 and math.ceil(0.2 * len(y)) >= len(classes) else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=strat)
    best = None
    for C in (0.5, 2.0, 8.0):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        P = clf.predict_proba(Xte); order = np.argsort(-P, 1); cl = clf.classes_
        top1 = float(np.mean(cl[order[:, 0]] == yte)); top3 = float(np.mean([yte[i] in cl[order[i, :3]] for i in range(len(yte))]))
        if best is None or top1 > best[1]: best = (C, top1, top3)
    clf = LogisticRegression(C=best[0], max_iter=3000, class_weight="balanced").fit(X, y)
    W, b, cl = clf.coef_, clf.intercept_, [str(c) for c in clf.classes_]
    if len(cl) == 2 and W.shape[0] == 1:   # two classes: write it as a softmax over [0, z]
        W = np.vstack([np.zeros_like(W[0]), W[0]]); b = np.array([0.0, b[0]])
    return {"classes": cl, "W": W, "b": b, "acc": round(best[1], 3), "acc3": round(best[2], 3), "n": int(len(y)), "C": best[0]}

def pack_head(h):
    W = h["W"]; C, D = W.shape; s = np.maximum(np.abs(W).max(1), 1e-9) / 127
    q = np.round(W / s[:, None]).astype(np.int8)
    return {"classes": h["classes"], "d": int(D), "w": base64.b64encode(q.tobytes()).decode(), "s": [float(f"{v:.7g}") for v in s],
            "b": [round(float(v), 5) for v in h["b"]], "acc": h["acc"], "acc3": h["acc3"], "n": h["n"], "at": datetime.now(timezone.utc).isoformat(), "by": "server"}


# ---------------- main ----------------
def main():
    if SEED: random.seed(int(SEED)); np.random.seed(int(SEED))
    for k, v in {"SUPABASE_URL": SB_URL, "SUPABASE_SERVICE_ROLE_KEY": SB_KEY, "MAPILLARY_TOKEN": MLY_TOKEN}.items():
        if not v: raise SystemExit(f"Missing secret {k}. Add it under GitHub → Settings → Secrets and variables → Actions.")
    meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta.json")))
    log("downloading model…"); r = http.get(MODEL_URL, timeout=300)
    if not r.ok: raise SystemExit(f"Couldn't download the model ({r.status_code}). Is wanderpin-vision.onnx in Supabase Storage → wp-models?")
    model = Model(r.content, meta)
    log("downloading cities and borders…")
    places = decode_places(http.get(PLACES_URL, timeout=300).content)
    country_at, random_point = load_borders(http.get(BORDERS_URL, timeout=300).json())
    store = Store(); store.load()

    deadline = T0 + MINUTES * 60
    stats = {cc: {"photos": 0, "plates": 0} for cc in COVERAGE}
    pool = ThreadPoolExecutor(8)
    visits = 0
    while time.time() < deadline:
        # visit the country with the fewest examples (ties broken randomly)
        need = sorted(COVERAGE, key=lambda c: (store.count("pole", c) + store.count("sign", c) + store.count("scene", c), random.random()))
        cc = need[0] if random.random() < 0.7 else random.choice(need[:15])
        visits += 1
        try:
            # 1) street photos
            got, tries, ims = 0, 0, []
            while len(ims) < PER_VISIT * 2 and tries < 10:
                tries += 1
                towns = places.get(cc, [])
                if towns and random.random() < 0.6:
                    lat, lng, _ = random.choice(towns) if random.random() < 0.5 else random.choices(towns, weights=[math.sqrt(t[2] + 1) for t in towns])[0]
                else:
                    pt = random_point(cc)
                    if not pt: continue
                    lat, lng = pt
                spot = f"{cc}{round(lat, 1)}_{round(lng, 1)}"
                for im in mly_search(lat, lng):
                    im["_spot"] = spot
                    if im["id"] in store.data["scene"]["src"]: continue
                    g = (im.get("computed_geometry") or im.get("geometry") or {}).get("coordinates")
                    if g and country_at(g[1], g[0]) != cc: continue
                    ims.append(im)
            random.shuffle(ims)
            for res, im in zip(pool.map(mly_photo, ims[:PER_VISIT]), ims[:PER_VISIT]):
                if not res: continue
                img, crops = res
                embs, boll = model.run([(img, b) for _, b in crops])
                g = im.get("_spot", "")
                for (tag, _), e, pb in zip(crops, embs, boll):
                    if tag == "bollard?":
                        if pb >= 0.6: store.add("bollard", cc, e, group=g); store.add("det_bollard", "yes", e, group=g)
                    elif tag == "neg":
                        for dc in DET_CATS: store.add(dc, "no", e, group=g)
                    elif tag == "scene": store.add("scene", cc, e, src=im["id"], group=g)
                    else: store.add(tag, cc, e, group=g); store.add("det_" + tag, "yes", e, group=g)
                got += 1
            stats[cc]["photos"] += got
            # 2) plates, until each country has a decent set
            if store.count("plate", cc) < 120:
                files = commons_files(cc, 10)
                for img in pool.map(fetch_image, files):
                    if img is None: continue
                    W, H = img.size; s = min(W, H)
                    blurred = img.filter(ImageFilter.GaussianBlur(max(3, W / 90)))   # Street View blurs plates, so learn the blurred look too
                    items = [(im2, b) for im2 in (img, blurred) for b in ((0, 0, W, H), ((W - s) / 2, (H - s) / 2, s, s))]
                    for e in model.run(items)[0]: store.add("plate", cc, e, group=f"p{hash(img.tobytes()[:4096]) % 10**9}")
                    stats[cc]["plates"] += 1
            if visits % 10 == 0: log(f"{visits} visits; latest {cc}: {got} photos; collection: " + ", ".join(f"{k}={len(v['y'])}" for k, v in store.data.items()))
        except SystemExit: raise
        except Exception as e:
            log(f"{cc}: skipped ({e})"); traceback.print_exc(limit=1)
    pool.shutdown(wait=False)
    try: store.save()
    except Exception as e: gh_error(f"Saving the collection failed: {e}")

    # learn
    summary = []
    for cat in STYLE_CATS + DET_CATS:
        d = store.data[cat]
        if len(d["y"]) < 12: summary.append((cat, "not enough examples yet", len(d["y"]))); continue
        try: h = train_head(np.stack(d["e"]).astype(np.float32), np.array(d["y"]), d["g"])
        except Exception as e: summary.append((cat, f"couldn't learn yet ({e})", len(d["y"]))); log(cat, "failed:", e); continue
        if not h: summary.append((cat, "needs at least 2 groups", len(d["y"]))); continue
        try: save_head(cat, pack_head(h))
        except Exception as e: gh_error(f"Saving {cat} failed: {e}"); summary.append((cat, f"save failed: {e}", h["n"])); continue
        summary.append((cat, f"{len(h['classes'])} groups, {round(h['acc'] * 100)}% right, {round(h['acc3'] * 100)}% top-3", h["n"]))
        log("saved", cat, summary[-1][1])

    run = {"at": datetime.now(timezone.utc).isoformat(), "minutes": round((time.time() - T0) / 60, 1), "visits": visits,
           "photos": sum(v["photos"] for v in stats.values()), "plates": sum(v["plates"] for v in stats.values()),
           "heads": {c: s for c, s, _ in summary}, "examples": {c: n for c, _, n in summary}}
    with open("stats.json", "w") as f: json.dump(run, f, indent=1)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"## Wanderpin training run\n{run['photos']} street photos and {run['plates']} plate photos in {run['minutes']} min.\n\n| What | Examples | Result |\n|---|---|---|\n")
            for c, s, n in summary: f.write(f"| {c} | {n} | {s} |\n")
    log("done:", json.dumps(run))
    bad = [c for c, s, _ in summary if s.startswith("save failed")]
    if bad and len(bad) == len([1 for _, s, _ in summary if not s.startswith(("not enough", "needs", "couldn"))]): raise SystemExit("Nothing could be saved to Supabase (see errors above).")

if __name__ == "__main__":
    try: main()
    except SystemExit as e:
        if e.code not in (None, 0): gh_error(e.code)
        raise
    except Exception as e:
        gh_error(f"{type(e).__name__}: {e} @ " + " < ".join(f"{f.name}:{f.lineno}" for f in traceback.extract_tb(e.__traceback__)[-3:]))
        raise
