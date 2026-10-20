# Expert placement

Every routed expert (40 layers x 384) lives in exactly one place that computes it during decode. On 2x RTX 3090 with
128 GB of RAM:

| Where | Experts | Computed by |
|---|---|---|
| Main GPU | ~1,050 (10.5 GB), next to the dense weights | itself; an adaptive cache |
| Other GPU | ~2,190 (23 GB) | itself, on activations shipped to it |
| RAM arena (hugepages) | ~12,600 (110 GB) | 14 CPU threads |
| The GGUF file | the ~550 coldest | the CPU, through the page cache |

About 74% of decode's expert picks are served from VRAM there; the CPU's share sets decode speed.

## GPU roles

dsx uses every visible GPU, up to six: one main GPU and up to five expert GPUs (`DSX_GPUS=0,2` picks which). At
startup it measures each GPU's memory and host-to-device bandwidth. The best-linked one (within 15%, then the most
memory) becomes the main GPU: dense weights, KV cache, prefill buffers, the vision tower and its own expert cache.
Every other GPU only holds experts.

An expert GPU whose link is at least half as fast as the main GPU's swaps experts like the main GPU's cache. A slower
one (a PCIe x4 slot) holds a fixed set: copying experts over it competes with the activations it needs every layer
(forcing swaps on the x4 card measured 22-31 tok/s against 29).

## Which experts go where

`--rank` orders experts by how often the router picked them on calibration text (`expert-rank.json` in the quant repo;
without it, layer-major round robin). Fixed GPUs take the hottest first, spread evenly across them; adaptive GPUs take
the next; the arena takes the rest in the same order until `--arena-gb` is full; whatever is left stays in the file.

## Following the conversation

After each token, every adaptive GPU commits finished copies and starts up to `DSX_ADAPT` new ones (default 8): the
experts with the highest decayed usage (x0.97 per token, seeded from the ranking) replace clearly colder residents of
the same quant types, copied from the arena on a side stream. An expert is computed by exactly one side in every token.
Experts an adaptive GPU may evict keep a copy in the arena, so the arena is sized for them too.

## Prefill

Each chunk streams the main GPU's missing experts from the arena over PCIe while the previous expert computes; the
other GPUs run their own experts on the chunk's activations in parallel.

## Flags

| | |
|---|---|
| `--main-gb` | GB of experts on the main GPU; negative = its free VRAM minus that much (default -1.5) |
| `--aux-gb` | GB of experts on each other GPU; negative = free VRAM minus that much (default -1.0); 0 = main GPU only |
| `--arena-gb` | RAM for the experts the CPU computes and copies of the cached ones |
| `--rank` | expert hotness order |
| `--chunk` | prefill tokens per pass (default 4096) |
| `DSX_GPUS=0,2` | use only these GPUs |
| `DSX_ADAPT=N` | swaps per token per adaptive GPU (default 8; 0 = static placement) |
| `DSX_AUX_ADAPTIVE=1` / `0` | force the other GPUs adaptive / fixed regardless of their link |

`--gpu0-gb` and `--gpu1-gb` still work as aliases of `--main-gb` and `--aux-gb`.

## What takes the main GPU's memory

Besides ~8 GB of dense weights and 1.5 GB of headroom, these come out of the main GPU's expert cache (RTX 3090,
v4b):

| Setting | Experts on the main GPU |
|---|---:|
| `--chunk 4096`, `--ctx 131072` (served) | 1,048 |
| `--chunk 4096`, `--ctx 32768` | 1,099 |
| `--chunk 2048`, `--ctx 131072` | 1,196 |
| `--chunk 1024`, `--ctx 131072` | 1,267 |

Image input loads a vision tower onto the main GPU, which leaves room for ~50 fewer experts there (1,048 instead of
1,099). A text-only server can start with `--no-vision` to keep that room for experts.

## Machines

| Machine | Flags |
|---|---|
| 2x 24 GB GPU, 128 GB RAM | `--arena-gb 110` |
| 1x 32 GB GPU, 128 GB RAM | `--main-gb=-2.0 --arena-gb 80`, `DSX_ADAPT=0`: the cache's RAM copies cost space this machine does not have |

Measurements behind these choices (cache hit rates, swap and decay sweeps, CPU bandwidth): [FINDINGS.md](../FINDINGS.md),
sections 6 and 9.
