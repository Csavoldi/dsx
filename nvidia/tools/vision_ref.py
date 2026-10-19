"""vision_ref.py - the reference span rows for one image, for checking `dsx vision-check`.

Runs DeepSeek's inference/vision.py (ViT + Aligner) and image_processor.py from the official checkpoint, in bf16 on
a GPU, and writes the image span rows (delimiters + aligner rows, as merge_image_embeddings) as raw f32.

    python vision_ref.py CHECKPOINT_DIR IMAGE OUT.f32 [--compare DSX.f32]
"""
import argparse, json, os, sys, types
import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("image"); ap.add_argument("out"); ap.add_argument("--compare")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.ckpt, "inference"))
import vision as V  # noqa: E402
import image_processor as IP  # noqa: E402

cfg = json.load(open(os.path.join(a.ckpt, "config.json")))["vision_config"]
args = types.SimpleNamespace(vision_dim=cfg["hidden_size"], vision_n_heads=cfg["num_attention_heads"], vision_inter_dim=cfg["intermediate_size"],
                             vision_patch_size=cfg["patch_size"], vision_rope_theta=cfg["rope_theta"], vision_n_layers=cfg["num_hidden_layers"],
                             vision_downsample_ratio=cfg["downsample_ratio"], vision_max_n_token=cfg["max_image_tokens"],
                             vision_min_pixels=cfg["min_pixels"], vision_max_wh_ratio=cfg["max_wh_ratio"], dim=5120)
torch.set_default_dtype(torch.bfloat16)
vit, al = V.ViT(args).cuda(), V.Aligner(args).cuda()
idx = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
sd = {}
for f in sorted({v for k, v in idx.items() if k.startswith(("vision.", "aligner.", "image_"))}):
    with safe_open(os.path.join(a.ckpt, f), "pt") as h:
        for k in h.keys():
            if k.startswith(("vision.", "aligner.", "image_")):
                sd[k] = h.get_tensor(k)
vit.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("vision.")})
al.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("aligner.")})
patches, nvh, nvw, nlh, nlw = IP.load_image({"url": a.image}, args)
with torch.inference_mode():
    rows = al(vit(patches.cuda(), nvh, nvw), nvh, nvw).float().cpu()
types_ = IP.image_token_types(nlh, nlw)
out = torch.zeros(len(types_), 5120)
out[types_ == IP.IMAGE_START] = sd["image_start"].float()
out[types_ == IP.IMAGE_END] = sd["image_end"].float()
out[types_ == IP.IMAGE_NEW_LINE] = sd["image_newline"].float()
out[types_ == IP.IMAGE] = rows.bfloat16().float()
out.numpy().astype(np.float32).tofile(a.out)
print(f"vit {nvh}x{nvw} llm {nlh}x{nlw} tokens {len(types_)}")
if a.compare:
    d = np.fromfile(a.compare, dtype=np.float32).reshape(out.shape)
    r = out.numpy()
    img = (types_ == IP.IMAGE).numpy()
    cos = (d[img] * r[img]).sum(1) / (np.linalg.norm(d[img], axis=1) * np.linalg.norm(r[img], axis=1) + 1e-12)
    rel = np.linalg.norm(d[img] - r[img]) / np.linalg.norm(r[img])
    print(f"IMAGE rows: cosine min {cos.min():.5f} mean {cos.mean():.5f}; relative L2 error {rel:.4f}; delimiters exact {np.array_equal(d[~img], r[~img])}")
