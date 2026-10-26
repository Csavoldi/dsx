from huggingface_hub import snapshot_download,HfApi
from pathlib import Path
import json,time
repo='Luni/Nanbeige-4.3-33B-EXL3';revision='bc301055ecdfd5d992f2afaf5f6a405a7f235572';dest=Path('/home/lxe/nanbeige-runner/models/Luni-Nanbeige-4.3-33B-EXL3-4.00bpw')
start=time.time();snapshot_download(repo_id=repo,revision=revision,local_dir=dest,allow_patterns=['*.json','*.safetensors','README.md'],max_workers=4)
info=HfApi().model_info(repo,revision=revision,files_metadata=True);files={x.rfilename:{'size':x.size,'sha256':x.lfs.sha256 if x.lfs else None} for x in info.siblings if x.rfilename.endswith('.safetensors')}
assert len(files)==5
for name,entry in files.items():assert (dest/name).stat().st_size==entry['size']
q=json.loads((dest/'quantization_config.json').read_text());assert q['codebook']=='mul1';assert all(q['tensor_storage'][f'model.layers.{i}.{p}']['bits_per_weight']==4 for i in range(52) for p in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj'])
r={'repo':repo,'revision':revision,'local_directory':str(dest),'download_seconds':time.time()-start,'files':files,'purpose':'higher-precision decoder candidate for same DSX CUDA runner; no runtime switch until validated','status':'downloaded; untested for speed/quality/task completion','adapter_plan':'DSX_NB_EXL3 decoder override with original embedding/head from our source pack; candidate quantized head is not used'}
Path('/home/lxe/gauntlet-nanbeige-doom/20261026-config-retry/receipts/quality-candidate-download.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
