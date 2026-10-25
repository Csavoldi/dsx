---
license: apache-2.0
base_model: Nanbeige/Nanbeige4.3-33B
base_model_relation: quantized
language:
- en
- zh
pipeline_tag: text-generation
library_name: exllamav3
tags:
- nanbeige
- llama
- dense
- reasoning
- quantized
- exl3
- trellis
- 2.5bpw
- rtx-5090
---

# Nanbeige4.3-33B EXL3 2.5 bpw

This is [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B), a dense reasoning model, with all 364 decoder projections quantized to **2.5 bits per weight** in the standard ExLlamaV3 trellis format. The token embedding and normalization weights remain BF16; the output head uses FP16 storage. The pack uses ExLlamaV3's `mul1` codebook and tensor layout.

The quant was made from the original BF16 checkpoint. It is intended for GPU inference with ExLlamaV3 and for experiments with the DSX Rust and CUDA runner. The model remains a dense network: every decoder matrix participates in every generated token, so expert offloading strategies do not apply.

## At a glance

| Property | Value |
|---|---|
| Source | `Nanbeige/Nanbeige4.3-33B` at `c1093c501cc6460427c0ac634db7223d0450c0ee` |
| Decoder | Uniform 2.5 bpw, 52 layers, 7 projections per layer |
| Format | EXL3, `mul1` codebook (`0x83DCD12D`), upstream tensor layout |
| Embedding and head | BF16 embedding; FP16 head; 16 bits per weight |
| Download | 14.40 GB (13.41 GiB) in two weight shards, plus tokenizer and metadata |
| Effective whole-model bitrate | 3.479 bpw, including BF16 tables, scales, and norms |
| Calibration | 32 rows × 512 tokens = 16,384 tokens |
| Original maximum position setting | 524,288 tokens; usable context depends on the runtime and KV memory |
| License | Apache 2.0, following the source checkpoint |

The 2.5 bpw label describes decoder trellis data. The two large 16-bit vocabulary tables increase the total download and effective whole-model bitrate. Weight size alone does not determine VRAM use: KV cache, activations, temporary buffers, and the output-head implementation also consume memory.

## Quality measurements

The reference comparison ran on one 96 GB RTX PRO 6000 Blackwell using ExLlamaV3 1.6.4 and PyTorch SDPA for both checkpoints. It forwarded the same 4,095 input tokens without a KV cache and scored the final 128 next-token targets from a 4,096-token WikiText-2 raw validation prefix.

| Configuration | Perplexity | Correct next-token predictions / 128 |
|---|---:|---:|
| Source checkpoint, same reference runtime | 10.2676 | 75 |
| This EXL3 pack, same reference runtime | 11.8915 | 70 |
| This pack in DSX, INT8 prefill, 3-bit KV, 4-bit runtime head | 12.7581 | 68 |

The matched reference comparison measured **0.17443 mean KL** from source to quant and **85.9375% next-token agreement**. Perplexity increased by 15.8% on this 128-token sample. The local DSX row includes additional runtime precision choices and a batched prefix followed by tokenwise teacher forcing, so it does not isolate weight quantization alone.

The reference receipt is [benchmarks/reference-quality.json](benchmarks/reference-quality.json). This is a short precision check; it is too small to characterize general model quality.

These measurements describe this quant and the named runtime configuration. They do not establish retention of the original model's coding, reasoning, or agent benchmark scores. Task-completion evaluations remain pending.

## Performance

Measured locally on one RTX 5090 with 32 GiB VRAM, an i7-13700, and 128 GiB DDR5-5200, using the DSX Nanbeige branch at `316b510`. Every row allocated 262,144-token capacity, used 3-bit KV, INT8 prefill, CPU embedding lookup, and a 4-bit runtime output head. The shipped head remains FP16. The desktop remained active.

| Workload | Actual input tokens | Generated tokens | Decode rate |
|---|---:|---:|---:|
| Python prime-function request, serial | 30 | 256 | 63.13 tok/s |
| Same request, three-token prompt lookup | 30 | 256 | 65.11 tok/s |
| Repeated assignment copying, serial | 628 | 512 | 62.65 tok/s |
| Same copying request, three-token prompt lookup | 628 | 512 | 151.97 tok/s |

The decode rows used a 2,048-token prefill batch limit. Copying accepted 364 of 410 proposed tokens; coding accepted 13 of 39. Repeated-copy speed does not establish general coding throughput. Verification changes execution shapes, so free-running output can differ across serial and lookup modes.

With the batch limit raised to 4,096, a cold 3,968-token WikiText prefix took 1.8862 seconds, or **2103.75 prefill tok/s**. A cold 16,384-token buried-key prompt took **11.8008 seconds**, or **1388.38 tok/s**, and returned the correct key. The corresponding 128-token perplexity check was **12.7036**. Larger batches use canonical 3-bit KV for any keys that have left the BF16 recent ring.

The longest locally populated prompt tested with this EXL3 pack was 16,384 tokens. A populated 256K retrieval test with this pack remains pending. The 100 tok/s general-generation target has not been reached. Prefix-cache hits are reported separately and do not inflate prefill rates.

Raw single-run receipts are under [benchmarks/](benchmarks/). Decode timing excludes loading and prefill. The reference-quality rows above describe the initial batch-512 local configuration; the later batch-4,096 result includes different prefill rounding and KV precision choices. Task-completion evaluations remain pending.

## Quantization

The converter was ExLlamaV3 1.6.4 at [`da68d57de61781e5b74ad68698ac4caf36408d7a`](https://github.com/turboderp-org/exllamav3/tree/da68d57de61781e5b74ad68698ac4caf36408d7a). Conversion ran on one RTX PRO 6000 Blackwell Server GPU with 96 GB of VRAM.

The conversion uses Hadamard rotations, input Hessians, LDLQ error feedback, and Viterbi trellis search. Decoder bitrate is fixed at 2.5 bpw, output scales use the converter's `auto` setting, and the embedding and head use 16-bit storage. Calibration uses the converter's bundled mixture of C4, code, multilingual text, technical text, Wikipedia, the bundled `tiny.utf8` story corpus, and random token rows. The exact source files and conversion settings are recorded in [quantization provenance](quantization-provenance.json).

Two conversion patches are included under `conversion/`: a faster fractional-rate Viterbi implementation taken from the DSX quantization work, and a PyTorch SDPA fallback for the rental image's attention compatibility. These patches run during conversion. The resulting pack retains the standard EXL3 codebook and tensor format; it does not use DSX T16's custom codebook.

```bash
python convert.py \
  -i Nanbeige4.3-33B \
  -w quant-work \
  -o Nanbeige4.3-33B-EXL3-2.5bpw \
  -b 2.5 -hb 16 -eb 16 -cr 32 -cc 512 -cb mul1
```

## Download and run

```bash
pip install -U huggingface_hub
hf download lxe/Nanbeige4.3-33B-EXL3-2.5bpw \
  --local-dir Nanbeige4.3-33B-EXL3-2.5bpw
```

Use an EXL3-capable loader such as [ExLlamaV3](https://github.com/turboderp-org/exllamav3). Load the downloaded directory as the model path and use the included tokenizer and chat template. Ordinary Transformers and llama.cpp loaders do not interpret EXL3 trellis weights.

The measured Rust/CUDA implementation is the [DSX Nanbeige branch](https://github.com/lxe/dsx/tree/codex/nanbeige-nvfp4), measured at commit `316b510`. See its [Nanbeige instructions](https://github.com/lxe/dsx/blob/codex/nanbeige-nvfp4/README.nanbeige.md). This experimental path currently targets SM120.

```bash
git clone --branch codex/nanbeige-nvfp4 https://github.com/lxe/dsx.git
cd dsx
CUDA_ARCHS=120 cargo build -p dsx --features cuda --release --bin dsx-nanbeige-server
DSX_NB_BATCH=4096 DSX_NB_ATTN8=2 DSX_NB_PART_SIZE=16384 DSX_NB_EXL3_NARROW=1 \
  ./target/release/dsx-nanbeige-server --model ../Nanbeige4.3-33B-EXL3-2.5bpw \
  --ctx 262144 --kv-bits 3 --host-embed --head-bits 4 --lookup 3 \
  --host 127.0.0.1 --port 8095
```

The DSX runtime quantizes the head independently at load time. Its 3-bit KV codec, INT8 prefill, and prompt lookup are runtime choices; the EXL3 weight files retain the standard format.

## Validation and files

[validation.json](validation.json) records tensor counts, matrix shapes, exact trellis byte counts, codebook markers, scale checks, and shard hashes. [SHA256SUMS](SHA256SUMS) contains the SHA256 digest of every weight shard. The files retain the converter's default 8 GiB shard limit and index layout.

```bash
cd Nanbeige4.3-33B-EXL3-2.5bpw
sha256sum -c SHA256SUMS
```

The published model configuration and tokenizer come from the pinned source checkpoint. Quantization metadata is included in `config.json` and `quantization_config.json`. The model config records the fractional bitrate as `bits_per_weight: 2.5`; its integer-only `bits` field is omitted for Hugging Face parser compatibility. Tensor-level quantization metadata retains the exact rates.

## Credits and license

The original model and tokenizer are by [Nanbeige](https://huggingface.co/Nanbeige). ExLlamaV3 and its quantization implementation are by [turboderp and contributors](https://github.com/turboderp-org/exllamav3), distributed under the MIT license. The model weights follow the source checkpoint's Apache 2.0 license. Included conversion code retains its applicable license and attribution.
