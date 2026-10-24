# HC premix reuse on MI325X

When `hc_mix_t` supplies `xo.mixd`, `hc_step` already has the completed premix weights. Skip the second reduction and finish on wave 1. The path without `mixd` is unchanged.

Four interleaved B4 decode pairs measured median 16.05 to 15.90 ms/step, about 0.9% faster. The HC kernel component fell from about 0.99 to 0.80 ms/step. Lucebox teacher-top-256 KL over 8,184 positions changed from 0.075210 to 0.075337; the existing floating atomic reductions are not bit-reproducible across whole-model runs. See the JSON receipt for placement, hashes and scope.

The standalone reference test compares the production kernel against a frozen pre-change kernel. It checks all output arrays bit-for-bit across 72 batch/mix/null/graph/TPB cases. On a ROCm host, compile with `hipcc -O3 -std=c++17 --offload-arch=gfx942 -mllvm -amdgpu-early-inline-all=true amd/bench/hc_premix_bench.hip -Wl,-rpath,/opt/rocm/lib -o hc_premix_bench` and run `./hc_premix_bench --check`. Omitting `--check` includes isolated event timing.
