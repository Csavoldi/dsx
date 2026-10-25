# Conversion source

These files reproduce the conversion patches and validation used for this model. `patch_qtf_opt.py` and `qt_frac_opt.cuh` come from the DSX T16 quantization work; the patch was applied to an unchanged upstream `mul1` codebook. `patch_sdpa.py` selects PyTorch SDPA for calibration compatibility with the rental image.

Clone ExLlamaV3 at the revision in `../quantization-provenance.json`, then run both patch scripts with the checkout directory as their argument. Rebuild ExLlamaV3 before running the conversion command in the model card. The paths in the validation scripts describe the release workspace; adjust them when reproducing the checks.

ExLlamaV3 code follows its MIT license. DSX conversion and validation additions follow DSX's MIT license. Model weights follow the Apache 2.0 license in the repository root.
