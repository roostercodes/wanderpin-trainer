"""The bigger image model for the country and area guesses: CLIP ViT-B/16 with DataComp weights.

The test of bigger models (bench.py, Oct 30) put it clearly ahead of TinyCLIP on our own photos
(overall look 27% -> 37%, poles 24% -> 33%, signs 13% -> 15%, same sample and same simple classifier).
It is about 100 MB (8-bit weights), so players download it once (in parts, Supabase allows 50 MB per file)
and their browser runs it only on the side views and the finds. TinyCLIP stays for spotting things.

Made here on GitHub (the weights come from Hugging Face), checked against the original, uploaded,
and kept in the cache between runs.
"""
import os, subprocess, sys, time
import numpy as np

ARCH, WEIGHTS = "ViT-B-16", "datacomp_xl_s13b_b90k"
TAG = "big1"                                  # heads trained on it say src: "big1"
NAME = "wanderpin-big-1.onnx"
PART = 45 * 1024 * 1024                       # under Supabase's 50 MB per file
GOOD = float(os.environ.get("BIG_GOOD") or 0.99)                               # how closely the shrunk copy must match the original


def _ensure():
    try:
        import torch, open_clip, onnx  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch==2.29.0", "onnx==1.17.0"])


def export(out_path, log=print, pretrained=WEIGHTS, check_images=None):
    """Writes the shrunk ONNX file (input pixels N×3×224×224, output emb N×512, unit length).
    Returns the average cosine between the int8 copy and the original on check_images."""
    _ensure()
    import torch, open_clip, inspect
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    torch.set_num_threads(os.cpu_count() or 2)
    model = open_clip.create_model(ARCH, pretrained=pretrained)
    vis = model.visual.eval()

    class Enc(torch.nn.Module):
        def __init__(s, v): super().__init__(); s.v = v
        def forward(s, x):
            e = s.v(x); return e / e.norm(dim=-1, keepdim=True)

    enc = Enc(vis).eval()
    fp, pre = out_path + ".fp32", out_path + ".pre"
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    # the fused attention shortcut can't be written to ONNX: turn it off (or, on old versions, export in
    # training mode, which skips it too; this model has no dropout, so the maths is the same)
    fast = getattr(getattr(torch.backends, "mha", None), "set_fastpath_enabled", None)
    if fast: fast(False)
    else: enc.train()
    with torch.no_grad():
        torch.onnx.export(enc, torch.randn(2, 3, 224, 224), fp, input_names=["pixels"], output_names=["emb"],
                          dynamic_axes={"pixels": {0: "n"}, "emb": {0: "n"}}, opset_version=17, **kw)
    enc.eval()
    X = np.stack(check_images).astype(np.float32) if check_images else np.random.RandomState(0).randn(8, 3, 224, 224).astype(np.float32)
    with torch.no_grad(): ref = enc(torch.from_numpy(X)).numpy()
    def agree(path):
        got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"pixels": X})[0]
        return float(np.mean(np.sum(ref * got, 1) / (np.linalg.norm(got, axis=1) + 1e-8)))
    # Ways to shrink it, fastest first. Plain int8 also squeezes the numbers flowing between layers, and CLIP
    # has a few huge ones there, so on the real weights it drifted (0.91-0.94). Weight-only 8-bit keeps
    # those exact (about 100 MB; browsers run it with onnxruntime-web 1.30).
    tried = []
    for kind in ("int8", "w8", "w4"):
        tmp = f"{out_path}.{kind}"
        try:
            if kind == "int8":
                try: quant_pre_process(fp, pre, skip_symbolic_shape=True); src = pre
                except Exception: src = fp
                quantize_dynamic(src, tmp, op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt8, per_channel=True)
            else:
                import onnx
                from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
                q = MatMulNBitsQuantizer(gemm_to_matmul(onnx.load(fp)), bits=8 if kind == "w8" else 4, block_size=32, is_symmetric=True)
                q.process(); q.model.save_model_to_file(tmp, use_external_data_format=False)
            cos = agree(tmp); mb = os.path.getsize(tmp) / 1e6
            log(f"big model: {kind} copy {round(mb, 1)} MB, agrees with the original: {round(cos, 4)} (cosine)")
            tried.append((cos, kind, tmp))
            if cos >= GOOD and mb <= 115: break
        except Exception as e: log(f"big model: {kind} failed: {e}")
    ok = [t for t in tried if t[0] >= GOOD]
    cos, kind, tmp = ok[0] if ok else max(tried) if tried else (0.0, "", "")
    if tmp: os.replace(tmp, out_path)
    for p in [fp, pre, fp + ".data"] + [t[2] for t in tried]:
        if p != out_path and os.path.exists(p): os.remove(p)
    del model, vis, enc
    export.kind = kind
    return cos


def gemm_to_matmul(m):
    """Gemm(A, B, C) with a constant B → MatMul(A, B') + C, so weight-only quantization covers those layers too."""
    from onnx import numpy_helper, helper
    g = m.graph; init = {i.name: i for i in g.initializer}; new = []
    for n in g.node:
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        if n.op_type != "Gemm" or n.input[1] not in init or a.get("transA", 0) or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0:
            new.append(n); continue
        W = numpy_helper.to_array(init[n.input[1]])
        if a.get("transB", 0): W = W.T
        wn = n.input[1] + "_mm"; g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(W), wn))
        mid = n.output[0] + "_mm"
        new.append(helper.make_node("MatMul", [n.input[0], wn], [mid if len(n.input) > 2 else n.output[0]], name=(n.name or mid) + "_mm"))
        if len(n.input) > 2: new.append(helper.make_node("Add", [mid, n.input[2]], [n.output[0]], name=(n.name or mid) + "_add"))
    del g.node[:]; g.node.extend(new)
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.initializer if i.name in used]; del g.initializer[:]; g.initializer.extend(keep)
    return m


def parts_of(blob):
    return [blob[i:i + PART] for i in range(0, len(blob), PART)]


class Big:
    """Runs the int8 file with onnxruntime (the same file the players get)."""
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def run(self, arrays, bs=16):
        out = []
        for i in range(0, len(arrays), bs):
            out.append(self.s.run(None, {self.inp: np.stack(arrays[i:i + bs]).astype(np.float32)})[0])
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


if __name__ == "__main__":   # local check of the export with random weights (no download)
    path = sys.argv[1] if len(sys.argv) > 1 else "big_test.onnx"
    t = time.time(); export(path, pretrained=None); print("export took", round(time.time() - t), "s")
    b = Big(path); X = [np.random.randn(3, 224, 224).astype(np.float32) for _ in range(32)]
    b.run(X[:4]); t = time.time(); b.run(X); print("ms per image (all cores):", round((time.time() - t) / 32 * 1000))
