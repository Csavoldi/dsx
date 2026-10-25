# T16 prefill throughput

Target: at least 1,000 newly processed prompt tokens per second on the dual RTX 3090 host. Keep T16 model bytes fixed, rank-3 decode deferral and the 256K context allocation. Report prompt length, chunk size, cold/warm state, cache reuse and resident expert membership with each result.

The baseline uses the pre-existing archived prefill-residency oracle binary, the saved 3K/8K prompts, no adaptation and no auxiliary streaming. The server is idle before taking ownership; each controller restores the original 256K launcher and checks executable/model identity and health. Raw output and controller/thermal logs are retained on the host and mirrored locally.

## Batch size

Compare the 8K prompt at 4K and 8K chunks. An 8K chunk should remove one encoder traversal's repeated weight transfers/expansion. Larger buffers can evict resident experts, so record the resulting placement and decode throughput. A faster 8K prompt does not establish a faster 3K prompt or short tool turn.

## Kernel changes

`DSX_T16_WARP_HAD=1`: one warp owns 128 values, using shuffle instructions and registers in the reference butterfly order. Removes block barriers and shared-memory round trips. Expected modest whole-prefill improvement; these transforms are a minority of total time. Require bit-identical pre/post outputs including in-place down inputs.

`DSX_T16_SPECIALIZED_DQ=1`: compile the T16 weight decoder for the three fixed bit widths and two layouts, preserving all decoded fp16 values. Removes runtime division/remainder and bit-width branches. Existing optimized decoder already achieves substantial bandwidth, so a large whole-prefill gain is uncertain. Require exact decoded weights for all six combinations and both projection dimensions.

Screen each flag separately and combined between matching controls. Compare final full-vocabulary logits and fixed decode continuations; run a real task before adopting a candidate as an agent-workflow improvement. Do not infer a throughput gain from kernel timing alone.

Previous rejected mechanisms: large GPU1 streaming share, decode int8 substitution for few-token prefill experts, and speculative next-layer expert prefetch that consumes main-GPU SM/DRAM bandwidth. See the October 24 decisions ledger.

## Buffer reuse

`DSX_PREFILL_REUSE_BUFFERS=1` shares the largest Q buffer with attention output and Engram/router scratch. Each attention block loads its own query slice before writing output; small split-K attention writes output in a later merge kernel. Q projection moves after compression/indexer scratch use; indexer queries continue to use the unchanged low-rank `qr` input. This saves approximately 1.91 GB at an 8K allocation. Tiny Engram GEMVs receive a separate source in the unused weight-expansion buffer because they read F32 input directly; the batched GEMM path first packs its source into separate F16 storage.

Require exact attention results for tensor-core, serial and split-K implementations, and paired full-model checks at fixed expert residency for full prompts and 8/1-token increments. Auto-placement performance runs separately record the extra resident experts and decode performance. All new flags remain off by default during evaluation.
