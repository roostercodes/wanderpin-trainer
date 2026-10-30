"""Which image model should the scanner use? Tests bigger models on our own collection.

For a balanced sample of scenes, poles and signs (cut-outs kept in the collection), each candidate
model turns every picture into numbers, and a straight-line classifier is scored on spots it never
saw: exactly how the current model is scored. Also notes size and speed, since players download
and run the model in their browser.
"""
import io, math, os, subprocess, sys, time, zlib
import numpy as np
from PIL import Image

# (name shown, open_clip architecture, weights, approx. browser download in MB as int8)
CANDIDATES = [
    ("CLIP ViT-B/32 (OpenAI)", "ViT-B-32", "openai", 88),
    ("CLIP ViT-B/16 (OpenAI)", "ViT-B-16", "openai", 87),
    ("CLIP ViT-B/16 (DataComp)", "ViT-B-16", "datacomp_xl_s13b_b90k", 87),
    ("SigLIP ViT-B/16", "ViT-B-16-SigLIP", "webli", 93),
]
STREETCLIP = ("StreetCLIP ViT-L/14-336 (made for Street View; too big for a browser, for comparison)", "geolocal/StreetCLIP", 305)

def ensure(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

def probe(E, y, g):
    from sklearn.linear_model import LogisticRegression
    te = np.array([zlib.crc32(str(x).encode()) % 5 == 0 for x in g])
    if te.sum() < 20 or (~te).sum() < 50: return None
    X = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(X[~te], y[~te])
    P = clf.predict_proba(X[te]); o = np.argsort(-P, 1); cl = clf.classes_
    return {"top1": round(float(np.mean(cl[o[:, 0]] == y[te])), 3), "top3": round(float(np.mean([y[te][i] in cl[o[i, :3]] for i in range(int(te.sum()))])), 3), "test": int(te.sum())}

def run(store, log, report, minutes=280, per=45):
    t_end = time.time() + minutes * 60
    ensure(["torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
    ensure(["open_clip_torch==2.29.0", "transformers==4.46.3"])
    import torch, open_clip
    torch.set_num_threads(os.cpu_count() or 2)
    cats = ["scene", "pole", "sign"]
    rows = {}
    for cat in cats:
        rs = []
        for (c, label), n in list(store.cnt.items()):
            if c != cat or n < 10: continue
            rs += store.db.execute("select id, y, g, img, e from ex where cat=? and y=? and img is not null order by random() limit ?", (cat, label, per)).fetchall()
        rows[cat] = rs
        log(f"bench: {cat}: {len(rs)} cut-outs, {len(set(r[1] for r in rs))} countries")
    res = {"per_country": per, "models": []}
    # the current model, from the numbers already in the collection
    cur = {"name": "TinyCLIP 40M/32 (now)", "mb": 45}
    for cat in cats:
        rs = rows[cat]
        E = np.frombuffer(b"".join(r[4] for r in rs), np.int8).reshape(len(rs), -1).astype(np.float32)
        cur[cat] = probe(E, np.array([r[1] for r in rs]), np.array([r[2] or f"r{r[0]}" for r in rs]))
    res["models"].append(cur); log("bench:", cur)

    def embed(fn, prep, imgs, bs=32):
        out = []
        for i in range(0, len(imgs), bs):
            x = torch.stack([prep(Image.open(io.BytesIO(b)).convert("RGB")) for b in imgs[i:i + bs]])
            with torch.no_grad(): out.append(fn(x).float().numpy())
        return np.concatenate(out)

    for name, arch, pre, mb in CANDIDATES:
        if time.time() > t_end - 20 * 60: log("bench: out of time before", name); break
        report("running", f"testing bigger models: {name}")
        try:
            model, _, prep = open_clip.create_model_and_transforms(arch, pretrained=pre); model.eval()
            r = {"name": name, "mb": mb}
            t0 = time.time(); n = 0
            for cat in cats:
                rs = rows[cat]
                E = embed(model.encode_image, prep, [x[3] for x in rs]); n += len(rs)
                r[cat] = probe(E, np.array([x[1] for x in rs]), np.array([x[2] or f"r{x[0]}" for x in rs]))
            r["ms_per_image_server"] = round((time.time() - t0) / max(1, n) * 1000)
            res["models"].append(r); log("bench:", r)
            del model
        except Exception as e:
            log("bench:", name, "failed:", e); res["models"].append({"name": name, "error": str(e)[:200]})
    # the big Street View model, scenes only, as a yardstick
    if time.time() < t_end - 30 * 60:
        name, hub, mb = STREETCLIP
        report("running", "testing bigger models: StreetCLIP (for comparison)")
        try:
            from transformers import CLIPModel, CLIPProcessor
            m = CLIPModel.from_pretrained(hub).eval(); proc = CLIPProcessor.from_pretrained(hub)
            prep = lambda im: proc(images=im, return_tensors="pt")["pixel_values"][0]
            seen = {}; rs = []
            for x in rows["scene"]:
                if seen.get(x[1], 0) < 20: seen[x[1]] = seen.get(x[1], 0) + 1; rs.append(x)
            Ec = np.frombuffer(b"".join(x[4] for x in rs), np.int8).reshape(len(rs), -1).astype(np.float32)
            same = probe(Ec, np.array([x[1] for x in rs]), np.array([x[2] or f"r{x[0]}" for x in rs]))   # the current model on the same smaller sample
            t0 = time.time()
            E = embed(lambda x: m.get_image_features(pixel_values=x), prep, [x[3] for x in rs], bs=8)
            r = {"name": name, "mb": mb, "scene": probe(E, np.array([x[1] for x in rs]), np.array([x[2] or f"r{x[0]}" for x in rs])),
                 "ms_per_image_server": round((time.time() - t0) / max(1, len(rs)) * 1000), "note": "scenes only, 20 per country", "current_model_same_sample": same}
            res["models"].append(r); log("bench:", r)
        except Exception as e:
            log("bench: StreetCLIP failed:", e); res["models"].append({"name": name, "error": str(e)[:200]})
    return res
