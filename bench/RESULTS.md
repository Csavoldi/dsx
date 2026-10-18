# Phase 0: measured on .102 (2026-10-17, Strata stopped, a ~63 MB/s download running)

| What | Result | Meaning |
|---|---:|---|
| RAM read, 2-20 threads | **39.5 GB/s** peak (2 threads already 38.5) | the decode bottleneck; matches the 40 GB/s in roofline.py |
| CPU 2-bit matvec (q2, 2.5 bpw), 16 threads on P-cores + HT | **38.9 GB/s** of weights | the CPU keeps up with RAM; not compute-bound |
| CPU 2-bit symmetric (q2s, 2.31 bpw) | 38.6 GB/s | same |
| CPU ternary (tq1, 1.70 bpw), 16 threads | 35.6 GB/s | nearly RAM-bound; 8 threads only 19.9 |
| Adding the 4 E-cores (20 threads) | slower (33-35 GB/s) | use P-core threads only |
| Expert-tier time per token, 180 experts from RAM | ~38-48 ms | the roofline's 38 ms holds |
| H2D GPU0 (Gen4 x16), cudaMallocHost | **13 GB/s** | IOMMU on (DMA-FQ), 4K pages |
| H2D GPU0 from hugepage memory + cudaHostRegister | **24.7 GB/s** | the arena must be THP/hugetlb backed |
| H2D GPU1 (chipset Gen3 x4) | 3.3 GB/s | GPU1 can't stream experts; static set only |
| Kernel launch: back-to-back / in a CUDA graph / launch+sync | 1.17 / 0.58 / 3.8-5.2 us | ~40 layers x ~10 kernels = ~0.25 ms/token in a graph |

| GPU -> CPU -> GPU flag handshake (mapped host memory, cuStreamWaitValue32) | **5.3 us** GPU0, 8.5 us GPU1 | 40 layers x ~6 us = 0.25 ms per token: per-layer CPU expert handoff is cheap |

Files: membw.c, qdot.c (synthetic weights, real kernels), pcie.cu. Built with gcc 13 -march=alderlake and CUDA 13 nvcc -arch=sm_86.
Not measured yet: NVMe random 256 B reads (the Engram rows); wait until the tables are on disk.

## CPU expert kernels on .102 (engine/src/bin/cpubench.rs, 2026-10-17)

| format | T=1, 16 thr | T=6, 16 thr |
|---|---:|---:|
| Q2_0 | 42.4 GB/s | 39.6 GB/s |
| PTQ1_0 | 39.8 GB/s | 25.3 GB/s (decode-bound) |
| Q3_K | 39.5 GB/s | 36.2 GB/s |
| Q8_0 | 40.1 GB/s | 39.3 GB/s |

One layer, 6 mixed-format experts from RAM: **1.58 ms at T=1**, 1.88 ms at T=6. Pool overhead 1-5 us per call while
spinning; 474 us if the workers parked, so they must keep spinning between layers during decode.
