# Packaging T16: decision and format spec

**Decision.** Keep GGUF as the container and T16 as dsx's own GGUF tensor types, and make the next release (the
calib5 requant) a cleaned-up format v2: one tensor per role for the trellis and standard f16 tensors for its scales,
the T16 parameters recorded in the GGUF's own metadata, this spec, and a numpy reference decoder. Don't ship an EXL3
layout, and don't pursue a ggml type. dsx reads v1 and v2.

The goal (lxe) is a clean, friendly package for dsx, not compatibility with other runtimes. GGUF already gives what
lxe likes about it: one file set, metadata and the tokenizer inside, and Hugging Face shows it as GGUF. What is not
clean today is how T16 sits inside it, and that is a small, local fix.

## What dsx-t16 v1 ships

`lxe/DeepSeek-V4.1-Flash-1017-dsx-t16`: four GGUF shards (llama.cpp split naming), plus `config.json`,
`expert-rank.json`, `vision.safetensors`, `README.md` and `SHA256SUMS`.

- Routed experts: per layer, `blk.L.ffn_{gate,up,down}_exps.weight` with dims `[k, n, 384]` and dsx type 1010 / 1011 /
  1012 (T16 at 1.5 / 2 / 2.5 bits). Each expert's matrix is one byte blob, `trellis | suh | svh`, so a tensor's size is
  not `rows x row bytes`; dsx special-cases it (`core/src/gguf.rs`).
- Everything else uses standard ggml types: dense Q8_0, embeddings F16, the Engram table Q6_K (shard 4).
- Inside the GGUF: the architecture config, Engram layout, `dsx.tokenizer_json`. Not inside: what T16 *is* (codebook,
  tile order, Hadamard, the output gains baked into svh, calibration). That lives only in the `config.json` prose.

How tools see it (checked 2026-10-22):
- Hugging Face's GGUF parser (`@huggingface/gguf`, which the Hub uses) keeps unknown type ids as numbers and counts
  parameters from shapes, so the files parse and the counts are right.
- llama.cpp's loader and gguf-py both reject the whole file on an unknown type id (`invalid tensor type` /
  `GGMLQuantizationType(1010)`), so `gguf-dump` cannot even show the metadata. That is the one tooling gap; the
  reference reader below closes it.

## Options

| | GGUF + dsx types (v2) | EXL3 layout | safetensors dir (HF-native) | ggml type upstream |
|---|---|---|---|---|
| One file set, metadata + tokenizer inside | yes | config + tokenizer files beside the shards | same as EXL3 | yes |
| Runs in dsx | yes (today) | new loader | new loader | yes |
| Runs elsewhere | no | **no, and silently wrong** (below) | no | after weeks of work, maybe |
| Generic tools read it | HF yes; gguf-py no | yes | yes | yes |
| The rest of this model (Q8_0 dense, Q6_K Engram, MTP, placement) | native | not representable | opaque byte arrays | native |
| Effort | ~1 day | 2-4 days + an exllamav3 PR | days, plus converter / card churn | weeks; acceptance unlikely |

**EXL3 layout.** Our tensors already look like EXL3 (`trellis`, `suh`, `svh`, a `mul1` marker), which is exactly the
problem: exllamav3 picks its codebook by the *presence* of the `mul1` tensor, not its value, and reconstructs tiles in
tensor-core position order. T16 uses a different multiplier and normalization (0x3A2F, 2-byte sum) and column-major
positions, so exllamav3, or an engine that runs EXL3 quants, would load our files and produce wrong weights without an
error. Making it real interop needs a distinct marker and an exllamav3 change for the codebook and the order. Even then
EXL3 has no place for the rest of the model: exllamav3 quantizes dense layers itself and knows nothing of Engram, the
DSpark draft layers or expert placement. If interop is ever wanted, the path is an experts-only export with a new
marker (e.g. `mul16`) plus that exllamav3 PR, not a change to dsx's package.

**safetensors directory.** Clean for generic tools, but the Q8_0 / Q6_K blocks would be opaque `uint8` arrays with
format tags in a sidecar, the tokenizer and config become separate files, and dsx, the converters and the model cards
are all built around GGUF. It moves the problem rather than solving it.

**A ggml type.** Not needed for this goal; recorded for completeness:
- ggml types are row-local: `blck_size` weights in `type_size` bytes along a row, with `to_float` / `from_float` /
  `vec_dot` per row and CUDA dequantize / mmvq / mmq paths per block. T16's 16 x 16 tiles span 16 rows (outputs), and
  each matrix needs per-input and per-output scales plus Hadamards on both sides, which a block type cannot carry.
- A row-local variant (1 x 256 strips along a row) would cross LDLQ's 16-input bands, so the quantizer would need
  block-LDLQ at 256 or a diagonal-importance Viterbi, at some quality cost. A row-stream GEMV ran at ~34 us (gate x6,
  RTX 3090) in the first T16 experiments, so speed would be fine.
- llama-quantize runs on the CPU, and a 65,536-state Viterbi over every 256-weight strip is hours per matrix there.
  ik_llama.cpp's IQ*_KT trellis types avoid Viterbi (independent 8-weight seeds) and add per-row metadata; they exist
  only in that fork.
- Precedents on the graph side do exist: mainline llama.cpp gained Hadamard rotations for Prism Bonsai
  (PR #29600, `llama_mul_mat_hadamard`, `prism.hadamard.*` keys) and for the KV cache, and per-expert scales could be
  gathered by expert id. Still a multi-week, multi-backend change for one model.

## Format v2 (with the calib5 release)

Changes from v1; dsx keeps reading v1.

1. **Three tensors per role instead of a blob.** `blk.L.ffn_{role}_exps.weight` holds only trellises (T16 type, dims
   `[k, n, E]`); `blk.L.ffn_{role}_exps.suh` (F16 `[k, E]`) and `blk.L.ffn_{role}_exps.svh` (F16 `[n, E]`) hold the
   scales. A T16 tensor's size then follows a plain per-weight formula, `k n E K / 8` bytes, like any block type, and
   the scales are readable by any tool.
2. **The T16 description in the GGUF.** Keys:
   - `dsx.format.version = 2`
   - `dsx.type_names = ["1010:T16_K1_5", "1011:T16_K2", "1012:T16_K2_5"]`
   - `dsx.t16.codebook = "t16-v1"` (multiplier 0x3A2F, 2-byte sum, k_inv fp16 0x20e6, k_bias fp16 0xca1e)
   - `dsx.t16.tile_order = "column-major"`, `dsx.t16.hadamard = 128`
   - `dsx.t16.svh_gain = ["gate,up:1.5=1.13", ...]` (the gains already multiplied into svh)
   - `dsx.t16.quantizer` (exllamav3 commit + T16 patch, sigma_reg), `dsx.t16.calibration` (set name, tokens, R2 path).
3. **Placement inside.** `expert-rank.json` becomes `dsx.expert_rank` (an array), so the GGUF set is complete;
   `config.json` stays as a human-readable manifest.
4. **Tools.** `quant/t16/t16_ref.py`: `inspect` (metadata and tensors of any dsx GGUF, which gguf-py refuses), `dequant`
   (an expert's effective f32 matrix), and `selftest`. It decodes the golden files made by exllamav3 exactly
   (W_inner bit-identical; W_eff and y within fp16 rounding, 5e-4 to 9e-4 relative).
5. **Type ids stay 1010-1012.** ggml's ids end at 42 (`GGML_TYPE_COUNT` 43), so no collision for a long time.

Effort: converter ~2 h (split the blob, write the keys), dsx loader ~3 h (separate tensors, v1 fallback), the rest is
done (this file, `t16_ref.py`).

## T16 tensor format (byte-exact)

A T16 expert matrix maps inputs `x` (k) to outputs `y` (n):

    y = svh * had128( had128(x * suh) @ W_inner )

`had128` is the orthonormal Walsh-Hadamard transform over consecutive blocks of 128 (k and n are multiples of 128).
`W_inner` is `[k][n]` (row = input) and stored as a trellis:

- **Tiles.** 16 x 16 tiles; tile `(kb, nb)` covers inputs `16 kb..16 kb + 15` and outputs `16 nb..16 nb + 15`. Tiles are
  stored by tile column: tile `(kb, nb)` is the `(nb * k/16 + kb)`-th, each `32 K` bytes (K = 1.5 / 2 / 2.5 bits per
  weight: 48 / 64 / 80 bytes).
- **Bit stream.** Each tile is one circular stream of `256 K` bits; byte 0 holds bits 0-7, most significant bit first.
- **Positions.** Position `t` (0-255) of a tile is input `t % 16`, output `t // 16` (column-major). Integer K: position t
  takes K bits and its bits end at `(t + 1) K`. Half-integer K: even positions take `floor(K)` bits, odd ones one more;
  position t's bits end at `(t + 1) floor(K) + (t + 1) // 2`.
- **State.** Position t's state is the 16 bits ending at its end bit, read MSB first, wrapping around the stream.
- **Value.** `p = (state * 0x3A2F) mod 2^16`, `b = (p & 0xFF) + (p >> 8)`, value `= fp16(fma(fp16(1024 + b), fp16 0x20e6,
  fp16 0xca1e))`: the exact product and sum rounded once to fp16, i.e. `0.009567260742 b - 2.4375` within half an ulp.
  There are 511 possible values.
- **Scales.** `suh` f16 `[k]`, `svh` f16 `[n]`. v1 stores each expert as `trellis | suh | svh` (one blob, experts
  consecutive); v2 stores them as the three tensors above.

The trellis bits come from exllamav3's quantizer (v1.6.4 + `quant/t16/make_mul16_exllamav3.sh`), whose own layout is
`int16 [k/16][n/16][16 K]` read as little-endian u32 words, MSB first; dsx's stored layout is that, with tiles reordered
by column and each u32 byte-swapped (`from_exl3` in `core/src/t16.rs` and `t16_ref.py`).
