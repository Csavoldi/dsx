#!/home/lxe/nanbeige-runner/eval-venv/bin/python
"""Prepare by default. --execute requires an exclusive idle GPU window."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = Path('/home/lxe/dsx-nanbeige')
SOURCE_BINARY = REPO / 'target/release/dsx-nanbeige'
MODEL = Path('/home/lxe/nanbeige-runner/quant-exl3/result')
TOKENS = Path('/home/lxe/nanbeige-runner/research-256k')
SERVICE = 'dsx-nanbeige.service'
FIXED = {
    'DSX_NB_BATCH': '4096', 'DSX_NB_ATTN8': '2', 'DSX_NB_PART_SIZE': '16384',
    'DSX_NB_EXL3_NARROW': '1', 'DSX_NB_EXL3_EXPERIMENT': 'pf4',
}

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def run_child(cmd, env, log):
    with log.open('w') as f:
        child = subprocess.Popen(cmd, env=env, stdout=f, stderr=f)
        try:
            status = child.wait()
            if status:
                raise subprocess.CalledProcessError(status, cmd)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()

def health():
    with urllib.request.urlopen('http://127.0.0.1:8095/metrics/progress', timeout=3) as r:
        return json.load(r)

def quality_comparison(out):
    import numpy as np
    a = json.loads((out/'ppl-default.json').read_text())
    b = json.loads((out/'ppl-i8pv.json').read_text())
    pa = np.memmap(out/'ppl-default.f32', dtype='<f4', mode='r').reshape(128, 166912)
    pb = np.memmap(out/'ppl-i8pv.f32', dtype='<f4', mode='r').reshape(128, 166912)
    kls, agreements = [], []
    for aa, bb in zip(pa, pb):
        la = np.asarray(aa, dtype=np.float64)
        lb = np.asarray(bb, dtype=np.float64)
        assert np.isfinite(la).all() and np.isfinite(lb).all()
        agreements.append(int(la.argmax() == lb.argmax()))
        la -= la.max(); lb -= lb.max()
        la -= np.log(np.exp(la).sum()); lb -= np.log(np.exp(lb).sum())
        kls.append(float((np.exp(la) * (la-lb)).sum()))
    return {'targets': 128, 'default_ppl': a['ppl'], 'i8pv_ppl': b['ppl'],
            'ppl_relative_change': b['ppl']/a['ppl']-1,
            'mean_kl_default_to_i8pv': float(np.mean(kls)),
            'max_kl_default_to_i8pv': max(kls), 'top1_agreement': sum(agreements)/128}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path, default=HERE/'fullmodel')
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(exist_ok=True, parents=True)
    wiki4 = json.loads((TOKENS/'wiki-4096.tokens.json').read_text())
    wiki32 = json.loads((TOKENS/'wiki-32768.tokens.json').read_text())
    assert len(wiki4) == 4096 and len(wiki32) == 32768
    # The 64K workload is intentionally synthetic repeated WikiText, not a task completion eval.
    write(out/'wiki-repeated-65536.tokens.json', wiki32 + wiki32)
    inputs = {4096: TOKENS/'wiki-4096.tokens.json',
              32768: TOKENS/'wiki-32768.tokens.json',
              65536: out/'wiki-repeated-65536.tokens.json'}
    manifest = {'prepared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'binary_source': str(SOURCE_BINARY), 'binary_sha256': sha(SOURCE_BINARY),
                'model': str(MODEL), 'capacity': 262144, 'kv_bits': 3, 'head_bits': 4,
                'host_embedding': True, 'lookup': 0, 'sampling': 'greedy, max128, EOS may stop early',
                'fixed_environment': FIXED, 'ppl_input_tokens': 4096, 'ppl_targets': 128,
                'ppl_prefill_tokens': 3968,
                'inputs': {str(n): {'path': str(p), 'sha256': sha(p)} for n,p in inputs.items()},
                'limitations': ['64K input repeats WikiText32K twice and is synthetic',
                               'WikiText continuation decode is not representative task completion',
                               'Output IDs may diverge because INT8 attention is approximate',
                               'Clocks are not locked; sequential pairs are not statistical confidence intervals',
                               'No source server deployment; baseline service restored in finally']}
    write(out/'manifest.json', manifest)
    if not args.execute:
        print(json.dumps({'status':'prepared only; no GPU use or service changes',
                          'manifest':str(out/'manifest.json'),
                          'estimated_minutes': '4-7, mostly 32K/64K cold prefill'}, indent=2))
        return
    progress = health()
    assert not progress['active'] and progress['pending_requests']==0 and progress['phase']=='idle', progress
    write(out/'idle-before-window.json', progress)
    pinned = out/'dsx-nanbeige-pinned'
    shutil.copy2(SOURCE_BINARY, pinned)
    assert sha(pinned) == manifest['binary_sha256']
    env = os.environ.copy()
    # Isolate the experiment from inherited format, graph, kernel and profiling overrides.
    for key in list(env):
        if key.startswith('DSX_NB_'):
            env.pop(key)
    env.update(FIXED)
    base = [str(pinned), '--model', str(MODEL), '--ctx', '262144', '--kv-bits', '3',
            '--host-embed', '--head-bits', '4', '--lookup', '0']
    plan = []
    for flag, label in [('0','default'), ('1','i8pv')]:
        plan.append((flag, f'ppl-{label}', ['--tokens',str(inputs[4096]), '--ppl-tail','128',
                                           '--ppl-logits',str(out/f'ppl-{label}.f32')]))
    for n in [4096,32768,65536]:
        for flag,label in [('0','default'), ('1','i8pv')]:
            plan.append((flag,f'decode-{n}-{label}', ['--tokens',str(inputs[n]),'--max-tokens','128']))
    runs=[]
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    stopped=False
    try:
        subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
        stopped=True
        for flag,name,extra in plan:
            e={**env,'DSX_NB_GQA_I8PV':flag}
            assert sha(pinned)==manifest['binary_sha256']
            cmd=base+extra+['--receipt',str(out/f'{name}.json')]
            start=time.monotonic()
            run_child(cmd,e,out/f'{name}.log')
            receipt=json.loads((out/f'{name}.json').read_text())
            assert receipt['ctx']==262144 and receipt['kv_bits']==3 and receipt['head_bits']==4
            row={'name':name,'flag':flag,'wall_seconds':time.monotonic()-start,
                 'binary_sha256':manifest['binary_sha256']}
            if name.startswith('decode-'):
                assert receipt['input_tokens']==int(name.split('-')[1])
                row.update(prompt_tokens=receipt['input_tokens'],output_tokens=receipt['output_tokens'],
                           nominal_final_context=receipt['input_tokens']+receipt['output_tokens'],
                           decode_tps=receipt['decode_tps'],prefill_seconds=receipt['prefill_seconds'],
                           prefill_tps=receipt['input_tokens']/receipt['prefill_seconds'])
            else:
                row.update(ppl=receipt['ppl'],evaluated_tokens=receipt['evaluated_tokens'])
            runs.append(row);write(out/'runs.json',runs);print(json.dumps(row),flush=True)
        summary={'quality':quality_comparison(out),'decode':[],'limitations':manifest['limitations']}
        for n in [4096,32768,65536]:
            a=json.loads((out/f'decode-{n}-default.json').read_text())
            b=json.loads((out/f'decode-{n}-i8pv.json').read_text())
            aa,bb=a['output_ids'],b['output_ids'];shared=min(len(aa),len(bb))
            mismatch=next((i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),None)
            summary['decode'].append({'input_tokens':n,'default_tps':a['decode_tps'],
                'i8pv_tps':b['decode_tps'],'speedup':b['decode_tps']/a['decode_tps'],
                'default_output_tokens':len(aa),'i8pv_output_tokens':len(bb),
                'default_final_context':n+len(aa),'i8pv_final_context':n+len(bb),
                'output_ids_identical':aa==bb,'first_mismatch_index':mismatch,
                'matching_positions_fraction':sum(x==y for x,y in zip(aa,bb))/shared if shared else None,
                'default_prefill_tps':n/a['prefill_seconds'],'i8pv_prefill_tps':n/b['prefill_seconds']})
        write(out/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        if stopped:
            subprocess.run(['systemctl','--user','start',SERVICE],check=True)
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                try:
                    restored=health()
                    write(out/'baseline-restored.json',restored)
                    print('Unchanged baseline service restored and responsive.',flush=True)
                    break
                except Exception:
                    time.sleep(.5)
            else:
                raise RuntimeError('baseline service did not become responsive within60s')

if __name__=='__main__':
    main()
