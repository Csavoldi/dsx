# Decode attention partition experiment

The GQA decode kernel now accepts its partition length as an argument. Production callers continue to pass 512, preserving the deployed geometry. `dsx-nanbeige --attention-bench` and `attention-partitions.py` sweep 128, 256, 512, 1024, and 2048 token partitions at populated positions from 4,096 through 262,135, with one, four, and eight query rows.

The Python harness executed the compiled SM120 CUDA kernel and split merge on the rented RTX PRO 6000 Blackwell while its reference API was idle. This avoids interference with the local RTX 5090 Doom run. These are synthetic attention measurements, not full-model output TPS or RTX 5090 measurements. The sweep uses exact ceil-divided split counts; production additionally pads split counts to a power of two, capped by capacity. That launch difference must be included in a production validation.

For one row at position 4,096, the 512-token partition took 0.2072 ms and 128 took 0.0632 ms. At 32,768, 512 took 0.4180 ms and 256 took 0.3275 ms. At 262,135, 512 remained fastest at 3.2413 ms. Larger partitions did not consistently improve long-context attention. Changing the live setting is not justified by this sweep.

All sampled outputs were finite and stayed within 0.01 absolute difference of the 512-token partition on random packed KV. This is a partition equivalence check, not an independent model-quality evaluation. The existing independent f64 attention audit remains separate. Kernel and Rust benchmark compilation passed; the standalone Rust benchmark has not yet run on an idle RTX 5090.

The next experiment should measure production-padded split counts and smaller partitions at short populated contexts on RTX 5090, then check representative end-to-end decode, graph replay across partition transitions, and quality before deployment. Preserve the live comparison while it runs. No throughput gain is booked from this experiment.

## Final local checkpoint

After the local Pi request was cancelled and the GPU became idle, the standalone Rust benchmark ran successfully on RTX 5090. `attention-partitions-5090.json` contains those measurements. The local sweep used exact ceil-divided split counts and one warmup call, so production-padded launches and warmed end-to-end validation remain outstanding. The user stopped further work before that validation. No new attention geometry was deployed.
