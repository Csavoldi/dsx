"""q_eval.py TAG L [VARIANT...]: quantize layer L's dumped experts (every 8th, with tokens in both dumps) with the
exllamav3 build on PYTHONPATH (stock = EXL3 mul1, patched = T16; one build per process), score each variant on the
calib held-out and Lucebox eval tokens, write /root/t16/rows_TAG_L.json. Variants: kG,kU,kD bit rates as "2,2,2"
or "2,2,3"; "ud" = Unsloth UD-IQ1_M's actual experts (from ud_fetch.py). TAG sanity: also the Gaussian check.
Scores as eval2.py: per expert err = sum (y_hat - y)^2, ye = sum y^2, n tokens, shrink = <y_hat, y> / <y, y>."""
import glob, json, os, sys
import numpy as np, torch
sys.path[:0] = ["/root/t16job", "/work/ds41/quant2"]
import quant_variants as QV
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

tag, L, variants = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
dev = torch.device("cuda:0")
os.makedirs("/root/t16", exist_ok=True)


def exl3(W, H, K, key):
    """W [out, in] -> quantized [out, in]; exllamav3 takes (in, out) and the input Hessian."""
    H_data = {"H": H.to(dev, torch.float32).clone(), "first_key": key, "count": 1, "finalized": False, "num_total": 0,
              "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
    qa = {"K": K, "seed": 1, "apply_out_scales": None, "devices": [dev], "mul1": True}
    wq, _, _ = quantize_exl3(W.t().contiguous().float(), H_data, qa, return_weight_q=True)
    return wq.t().contiguous().float().to(dev)


if L == 0:  # sanity: Gaussian 4096 x 4096, H = I, K = 2: weight MSE
    torch.manual_seed(0)
    W = torch.randn(4096, 4096, device=dev)
    for K in (2,):
        Q = exl3(W, torch.eye(4096, device=dev), K, "gauss")
        mse = float((Q - W).square().mean())
        print(f"{tag} gaussian K{K} mse {mse:.5f}", flush=True)
        json.dump({"mse": mse}, open(f"/root/t16/gauss_{tag}.json", "w"))
    sys.exit(0)

ud = None
if "ud" in variants:
    from gguf import quants
    import gguf
    meta = json.load(open("/root/ud/meta.json"))
    ud = {}
    for r in ("gate", "up", "down"):
        name = f"blk.{L}.ffn_{r}_exps.weight"
        ty, dims = meta[name]
        t = gguf.GGMLQuantizationType[ty]
        bs, ts = gguf.GGML_QUANT_SIZES[t]
        cols, rows, ne = dims
        raw = np.fromfile(f"/root/ud/{name}.bin", dtype=np.uint8).reshape(ne, rows, cols // bs * ts)
        ud[r] = (raw, t)

rows_out = []
for f in sorted(glob.glob(f"/work/udump/L{L:02d}_e*.pt")):
    d = torch.load(f); e = d["ex"]
    fe = f"/work/edump/L{L:02d}_e{e:03d}.pt"
    if not os.path.exists(fe):
        continue
    de = torch.load(fe)
    if not len(d["xv"]) or not len(de["xv"]):
        continue
    W1, W3, W2 = (d[k].float().to(dev) for k in ("W1", "W3", "W2"))
    H1, H2, lim = d["H1"].to(dev), d["H2"].to(dev), d["lim"]
    sets = {"calib": (d["xv"].to(dev), d["gv"].to(dev)), "eval": (de["xv"].to(dev), de["gv"].to(dev))}
    Y = {k: QV.expert_f(x, W1, W3, W2, g, lim) for k, (x, g) in sets.items()}
    cache = {}
    def q(W, H, K, key):
        if (key, K) not in cache:
            cache[(key, K)] = exl3(W, H, K, key)
        return cache[(key, K)]
    for v in variants:
        if v == "ud":
            from gguf import quants
            a, b, c = (torch.from_numpy(np.ascontiguousarray(quants.dequantize(ud[r][0][e], ud[r][1]))).float().to(dev) for r in ("gate", "up", "down"))
        else:
            kg, ku, kd = (int(x) for x in v.split(","))
            a, b, c = q(W1, H1, kg, "gate"), q(W3, H1, ku, "up"), q(W2, H2, kd, "down")
        r = {"ex": e, "variant": f"{tag}:{v}"}
        for k, (x, g) in sets.items():
            Yq = QV.expert_f(x, a, b, c, g, lim); y = Y[k]; E = Yq - y
            err, ye = float(E.square().sum()), float(y.square().sum())
            r[k] = {"n": len(x), "err": err, "ye": ye, "shrink": float((Yq * y).sum()) / max(ye, 1e-30)}
        rows_out.append(r)
    print(f"{tag} L{L} e{e} done ({len(d['xv'])} calib / {len(de['xv'])} eval tokens)", flush=True)
json.dump(rows_out, open(f"/root/t16/rows_{tag}_{L}.json", "w"))
