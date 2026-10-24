# .102 measurements, night of 2026-10-22/23

Raw logs and the scripts that produced them (window scripts `win*.sh` ran on .102 with the dsx server stopped;
each `win*.log` is its output). The numbers are summarized in FINDINGS.md sections 8d and 9 and in
docs/budget-102.md. Windows 30–35 record the later integration, summarized in FINDINGS section 9b and
[docs/current-state.md](../../../docs/current-state.md). These scripts stop the server and run hardware jobs; inspect
them and coordinate a new window before use. They are historical reproductions, not a queue to resume.

Large artifacts stay on .102 (not in git): nsys traces /tmp/pf3000*, /tmp/pf60*, /tmp/pf300*,
/tmp/dec64*, /tmp/srvturn*, /tmp/lp3000* (.nsys-rep / .sqlite) and /tmp/pregate.dump.

Note on limits: from 02:10:25 on 2026-10-23 both GPUs are held at 230 W (gpu-cool service) and the CPU has a RAPL
cap of 125/180 W in the recorded window; earlier runs were at GPU0 250 W / GPU1 280 W and no CPU cap. These
settings are not persistent guarantees. An October 24 read-only inspection found GPU limits still at 230 W but the
CPU long-term limit at ~4095.9 W and short-term at 180 W. Re-read the limits before load; do not compare across caps.
