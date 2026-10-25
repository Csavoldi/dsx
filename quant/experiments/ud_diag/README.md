# UD-IQ1_M versus early scalar quants (October 20, 2026)

This is a historical study of the scalar quants before T16 v1.
The scripts support FINDINGS.md section 7’s codebook measurements. They ran on a rented H100 pod with absolute paths:

- `/work/udN`: dumps from a forward-only `v41_calib` chain over the `calib/` corpus, every 8th expert of layer N
  (`V41_SCREEN_TIERS="" V41_SCREEN_IQ2=0 V41_DUMP=... V41_DUMP_LAYER=5,20,32,39`)
- `/work/edump`: the same experts' routed inputs from a second chain over the Lucebox eval tokens
  (`v41_calib.py prep-ids heldout.i32 --rows 8`)
- `/work/wdump`: a chain over the corpus's 20 web sequences only (web-only Hessians)
- `/work/um/UD-IQ1_M`: Unsloth's shards 1-4; `/work/um/imatrix_unsloth.gguf_file`: their imatrix
- `/work/ggml`: llama.cpp (PR #30817, b72050d) built with `BUILD_SHARED_LIBS=ON` for `libggml-base.so`

| Script | Measures |
|---|---|
| `unsloth_score.py` | UD's dequantized experts on the calib dumps' held-out tokens |
| `eval2.py` | our tiers (rotated + refit), IQ2_XXS, our v5 / v7 allocations and UD on calib vs Lucebox tokens; E_L, shrink, bias share |
| `rowstats.py` | how output and error energy spread over (token, expert) rows |
| `eval3.py` | RTN, diag-H, GPTQ, GPTQ damp 10, production, on both token sets |
| `eval4.py` | Hessians from all domains vs the web sequences only |
| `repro.py` | ggml's own quantizers (ctypes) with UD's layout and either imatrix, vs UD's experts |
| `cmpbytes.py` | UD's raw IQ2_XXS blocks vs stock ggml's on the same weights |
| `wspace.py` | weight-space error of UD's experts vs stock ggml's |
| `merge.py` | an iq2xxs-only screening run's stats merged into an earlier run's (v7) |
