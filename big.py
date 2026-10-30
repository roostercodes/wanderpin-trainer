"""The bigger image model for the country and area guesses: CLIP ViT-B/16 with DataComp weights.

The test of bigger models (bench.py, Oct 30) put it clearly ahead of TinyCLIP on our own photos
(overall look 27% -> 37%, poles 24% -> 33%, signs 13% -> 15%, same sample and same simple classifier).
It is ~87 MB as int8, so players download it once (in two parts, Supabase allows 50 MB per file)
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


def _ensure():
    try:
        import torch, open_clip, onnx  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch==2.29.0", "onnx==1.17.0"])


def export(out_path, log=print, pretrained=WEIGHTS, check_images=None):
    """Writes the int8 ONNX file (input pixels N×3×224×224, output emb N×512, unit length).
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
    try: quant_pre_process(fp, pre, skip_symbolic_shape=True)
    except Exception as e: log("big model: pre-processing skipped:", e); pre = fp
    quantize_dynamic(pre, out_path, op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt8)
    for p in (fp, pre, fp + ".data"):
        if p != out_path and os.path.exists(p): os.remove(p)
    X = np.stack(check_images).astype(np.float32) if check_images else np.random.RandomState(0).randn(8, 3, 224, 224).astype(np.float32)
    with torch.no_grad(): ref = enc(torch.from_numpy(X)).numpy()
    got = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"]).run(None, {"pixels": X})[0]
    cos = float(np.mean(np.sum(ref * got, 1) / (np.linalg.norm(got, axis=1) + 1e-8)))
    log(f"big model: int8 copy {round(os.path.getsize(out_path) / 1e6, 1)} MB, agrees with the original: {round(cos, 4)} (cosine)")
    del model, vis, enc
    return cos


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
