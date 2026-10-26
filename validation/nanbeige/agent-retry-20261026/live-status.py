#!/usr/bin/python3
import json,urllib.request
try:
 with urllib.request.urlopen('http://127.0.0.1:8095/metrics/progress',timeout=.5) as r:d=json.load(r)
 phase=d['phase'];avg=d.get('average_output_tps');rate=d.get('output_tps');n=d['generated_tokens'];seq=d['sequence_tokens'];ctx=d['context_capacity']
 if phase=='queued_or_prefill':print('Prefill / queue')
 elif d['active']:print(f"{phase} | {rate:.1f} tok/s | avg {avg:.1f} | out {n:,} | seq {seq:,}/{ctx:,}" if rate is not None and avg is not None else f'{phase} | warming up')
 else:print(f'Idle | last {avg:.1f} tok/s | {n:,} output tokens' if avg is not None else 'Ready | EXL3 2.5 bpw')
except Exception:print('Server starting / telemetry unavailable')
