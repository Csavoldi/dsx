#!/usr/bin/env python3
"""Read-only serving metadata. Run over SSH on the model host."""
import hashlib,json,pathlib,subprocess,time
pids=subprocess.check_output(['pgrep','-x','dsx-server'],text=True).split()
assert len(pids)==1,pids
pid=int(pids[0]);proc=pathlib.Path('/proc')/str(pid)
base=pathlib.Path('/home/lxe/ai/ds41');model=base/'hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf'
stat=model.stat()
env=[x.decode(errors='replace') for x in (proc/'environ').read_bytes().split(b'\0') if x.startswith(b'DSX_')]
print(json.dumps({'time':time.time(),'pid':pid,'command':(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(),'binary_sha256':hashlib.sha256((proc/'exe').read_bytes()).hexdigest(),'dsx_env':env,'model':{'path':str(model),'size':stat.st_size,'inode':stat.st_ino,'mtime_ns':stat.st_mtime_ns},'rank_sha256':hashlib.sha256((base/'calib/rank.json').read_bytes()).hexdigest(),'rapl_long_uw':pathlib.Path('/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw').read_text().strip(),'gpu_state':subprocess.check_output(['nvidia-smi','--query-gpu=index,power.limit,temperature.gpu,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),'gpu_lock':(base/'GPU_LOCK').read_text() if (base/'GPU_LOCK').exists() else None},indent=2))
