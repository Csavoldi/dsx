"""Use PyTorch SDPA during calibration on the pinned rental software image."""
from pathlib import Path
import sys
p=Path(sys.argv[1])/'exllamav3/modules/attention_fn/dispatch.py'
s=p.read_text()
marker='# Nanbeige quantization-only SDPA compatibility'
if marker not in s:
 s+='\n'+marker+' for the rental image Triton version.\nattn_fns = [fn_torch_sdpa_fallback_nocache, fn_torch_sdpa_fallback_cache]\n'
 p.write_text(s)
