"""figs.py: the figures of the T16 paper (PDF, into figs/). Every number is a measurement recorded in the dsx repo
(docs/t16.md, FINDINGS.md, cards/dsx-t16) or reported by the DSX threads on 2026-10-21/22; sources are noted inline."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
T16, OTHER, REF, ACC = "#1f5fa8", "#9aa3ad", "#c0392b", "#e08e2b"


def save(fig, name):
    fig.savefig(f"figs/{name}.pdf")
    plt.close(fig)


# 1. Codebook accuracy on Gaussian data at 2 bits (docs/t16.md "Accuracy of the codebook": bench/cbq.c, cbsearch.c,
#    Viterbi-quantized i.i.d. Gaussian tiles at the best scale; E8P from QTIP Table 1)
cb = [("R(D) bound", 0.0625, REF), ("EXL3 mul1", 0.0640, OTHER), ("random Gaussian table", 0.0641, OTHER),
      ("16-bit mult., 4-byte sum", 0.0654, OTHER), ("T16 (0x3A2F, 2-byte sum)", 0.0665, T16),
      ("nibble tables (shuffle only)", 0.0775, OTHER), ("E8P (QuIP#), 8-D lattice", 0.089, OTHER),
      ("Lloyd-Max scalar", 0.1175, OTHER)]
fig, ax = plt.subplots(figsize=(3.3, 2.1))
ys = range(len(cb))[::-1]
ax.barh(list(ys), [v for _, v, _ in cb], color=[c for _, _, c in cb], height=0.65)
ax.set_yticks(list(ys)); ax.set_yticklabels([n for n, _, _ in cb])
for y, (_, v, _) in zip(ys, cb):
    ax.text(v + 0.0015, y, f"{v:.4f}", va="center", fontsize=7)
ax.axvline(0.0625, color=REF, lw=0.6, ls=":")
ax.set_xlim(0.05, 0.128); ax.set_xlabel("MSE / variance, i.i.d. Gaussian, 2 bits per weight")
save(fig, "codebook_mse")

# 2. Decode cost: GPU GEMV per 6-expert call (docs/t16.md GPU table, nsys medians at 285 W, two rounds) and CPU
#    (T16 v6 vs UD's IQ mix, 14 threads, interleaved)
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.2), gridspec_kw={"width_ratios": [2.1, 1]})
names = ["IQ2_XXS\n2.06", "IQ1_S\n1.56", "IQ2_XS\n2.31", "IQ2_S\n2.56", "T16\n1.5", "T16\n2", "T16\n2.5"]
gate = [42.6, 41.0, 50.3, 51.8, 29.3, 29.5, 34.7]
down = [39.2, 39.5, 42.7, 49.3, 27.9, 27.9, 32.1]
x = range(len(names))
w = 0.38
cols = [OTHER] * 4 + [T16] * 3
a.bar([i - w / 2 for i in x], gate, w, color=cols, label="gate/up x6")
a.bar([i + w / 2 for i in x], down, w, color=cols, alpha=0.55, label="down x6")
a.set_xticks(list(x)); a.set_xticklabels(names)
a.set_ylabel("GEMV time per call (us)"); a.set_title("RTX 3090, one token, 6 experts")
a.text(0.01, 0.97, "dark: gate/up x6   light: down x6\nlabels: format, bits per weight", transform=a.transAxes, va="top", fontsize=7)
a.set_ylim(0, 60)
cpu_n = ["UD-IQ1_M\nmix", "T16\n1.5", "T16\n2", "T16\n2.5"]
cpu_v = [1.73, 1.55, 1.57, 1.78]
b.bar(range(4), cpu_v, 0.6, color=[OTHER, T16, T16, T16])
for i, v in enumerate(cpu_v):
    b.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
b.set_xticks(range(4)); b.set_xticklabels(cpu_n); b.set_ylim(0, 2.1)
b.set_ylabel("ms per 6-expert call"); b.set_title("i7-12700K, 14 threads")
save(fig, "decode_cost")

# 3. The dense fractional-K Viterbi kernel (bench/qt_frac_bench.cu, quant/t16/qtf_check.py; RTX 3090 at 250 W)
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.0))
lab = ["K 1.5", "K 2.5", "K 2\n(integer)"]
stock, dense = [5.01, 2.44, 1.39], [0.98, 1.15, 1.40]
x = range(3)
a.bar([i - 0.19 for i in x], stock, 0.38, color=OTHER, label="exllamav3 generic")
a.bar([i + 0.19 for i in x], dense, 0.38, color=T16, label="dense kernel (ours)")
for i in x:
    if stock[i] / dense[i] > 1.2:
        a.text(i + 0.19, dense[i] + 0.1, f"{stock[i] / dense[i]:.1f}x", ha="center", fontsize=7)
a.set_xticks(list(x)); a.set_xticklabels(lab); a.set_ylabel("s per 2304 x 5120 matrix")
a.set_title("Viterbi over 46,080 Gaussian tiles"); a.legend(frameon=False, loc="upper right")
lab2 = ["layer 5\nK 1.5/1.5/2", "layer 20\nK 2/2/2.5"]
b.bar([i - 0.19 for i in range(2)], [16.3, 7.5], 0.38, color=OTHER, label="before")
b.bar([i + 0.19 for i in range(2)], [5.0, 5.9], 0.38, color=T16, label="after")
b.set_xticks(range(2)); b.set_xticklabels(lab2); b.set_ylabel("s per expert (3 matrices)")
b.set_title("quantize_exl3 on real experts"); b.legend(frameon=False)
save(fig, "viterbi")

# 4. Output gain (DSX Dev 1: half-T16 model, gate/up at K 1.5; full v1 model, gate/up at K 2; Lucebox KL vs FP8)
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.0))
g1, k1 = [1.00, 1.10, 1.13, 1.16], [0.3145, 0.2564, 0.2503, 0.2551]
a.plot(g1, k1, "o-", color=T16)
a.set_xlabel("output gain on 1.5-bit gate / up"); a.set_ylabel("KL vs FP8"); a.set_title("T16 layers 0-19 + UD layers 20-39")
g2, k2 = [1.00, 1.04, 1.07, 1.10], [0.1962, 0.1943, 0.1971, 0.2047]
b.plot(g2, k2, "o-", color=T16, label="dsx-t16 v1")
b.axhline(0.2715, color=OTHER, ls="--", lw=0.8); b.text(1.0, 0.2665, "dsx-ud (UD-IQ1_M experts) 0.2715", fontsize=7, color="#555")
b.set_ylim(0.185, 0.28)
b.set_xlabel("output gain on 2-bit gate / up"); b.set_title("full model (1.5-bit gain x1.13 baked)")
save(fig, "gain")

# 5. Quality per byte (Lucebox 1017 eval, KL vs FP8 top-256; size without the Engram table; cards and FINDINGS.md)
pts = [("dsx-2bit v4b (scalar GSQ tiers)", 148, 0.373, OTHER), ("Unsloth UD-IQ1_S", 122, 0.361, OTHER),
       ("Unsloth UD-IQ1_M (llama.cpp)", 140, 0.275, OTHER), ("dsx-ud (UD-IQ1_M experts)", 142, 0.272, OTHER),
       ("Maya-S (llama.cpp)", 178, 0.195, ACC), ("Unsloth UD-Q2_K_XL", 224, 0.114, OTHER),
       ("dsx-t16 v1 (this work)", 141, 0.196, T16)]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for n, s, k, c in pts:
    ax.scatter(s, k, color=c, s=28 if c == T16 else 18, zorder=3)
    dx, dy, ha = 3, 0.004, "left"
    if n.startswith("Unsloth UD-IQ1_M"):
        dx, dy = 4, -0.026
    if n.startswith("dsx-ud"):
        dx, dy = 4, 0.008
    if n.startswith("Unsloth UD-IQ1_S"):
        dx, dy = 2, -0.022
    if n.startswith("dsx-2bit"):
        dy = 0.008
    if n.startswith("dsx-t16"):
        dx, dy = 2, -0.03
    if n.startswith("Maya"):
        dx, dy = 3, 0.008
    ax.text(s + dx, k + dy, n, fontsize=6.6, ha=ha, color="#222" if c != T16 else T16)
ax.set_xlabel("experts + dense weights (GB)"); ax.set_ylabel("KL vs FP8 (lower is better)")
ax.set_xlim(110, 255); ax.set_ylim(0.08, 0.41)
save(fig, "pareto")

# 6. Transfer from calibration to eval text (docs/t16.md "Accuracy on real experts": relative output error, every 8th
#    expert of layers 5 and 20, full Hessian; quant/t16/real-experts-2026-10-21.txt)
fig, ax = plt.subplots(figsize=(3.3, 2.1))
var = ["EXL3 mul1\n2.0", "T16\n2.0", "UD-IQ1_M\n1.73 / 2.23"]
l5c, l5e = [0.049, 0.051, 0.153], [0.102, 0.106, 0.174]
l20c, l20e = [0.119, 0.123, 0.148], [0.185, 0.194, 0.151]
x = range(3)
w = 0.2
ax.bar([i - 1.5 * w for i in x], l5c, w, color="#7fa7d6", label="L5 calib")
ax.bar([i - 0.5 * w for i in x], l5e, w, color=T16, label="L5 eval")
ax.bar([i + 0.5 * w for i in x], l20c, w, color="#f0c08a", label="L20 calib")
ax.bar([i + 1.5 * w for i in x], l20e, w, color=ACC, label="L20 eval")
ax.set_xticks(list(x)); ax.set_xticklabels(var); ax.set_ylabel("relative output error")
ax.legend(frameon=False, ncol=2, fontsize=6.6, loc="upper left")
ax.set_ylim(0, 0.27)
save(fig, "transfer")
print("figures:", sorted(os.listdir("figs")))

# 7. Calibration coverage before / after the probe top-up (DSX Dev 4, requant_notes.md: fewest fit tokens of any
#    expert per layer over the 506 v5 rows, then with the 69 probe rows; 575 rows = 1,177,600 tokens)
before = [820, 250, 186, 353, 600, 384, 1049, 912, 922, 271, 945, 1831, 1000, 633, 911, 373, 1247, 1132, 847, 511,
          407, 100, 60, 371, 101, 204, 508, 143, 591, 432, 244, 214, 137, 134, 364, 184, 109, 184, 167, 241]
after = [1025, 267, 241, 550, 1015, 419, 1244, 974, 974, 460, 1088, 2145, 1283, 1004, 1097, 413, 1592, 1424, 1531, 678,
         848, 131, 91, 726, 142, 299, 614, 187, 827, 580, 439, 301, 256, 619, 588, 229, 139, 476, 338, 525]
fig, ax = plt.subplots(figsize=(6.8, 1.9))
L = range(40)
ax.vlines(list(L), before, after, color="#bbb", lw=1.2)
ax.scatter(list(L), before, s=10, color=OTHER, label="calib5 base (1.04M tokens)", zorder=3)
ax.scatter(list(L), after, s=10, color=T16, label="+ 69 probe rows (1.18M tokens)", zorder=3)
ax.axhline(256, color=REF, lw=0.7, ls=":"); ax.text(39.6, 262, "256", color=REF, fontsize=7, ha="right", va="bottom")
ax.set_yscale("log"); ax.set_xlim(-0.8, 39.8); ax.set_xticks(range(0, 40, 5))
ax.set_xlabel("layer"); ax.set_ylabel("fewest fit tokens\nof any expert")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "coverage")
print("coverage figure written")
