# Engram verification history copy

Each speculative verification needs only the last n-1 mapped history tokens before appending its first new token. Copy that suffix instead of the entire context. Every generated n-gram tail is preserved.

`amd/bench/history_tail.rs` compares the actual full-history and suffix-copy calculations over 30,753 cases (history lengths 0–200, n-grams 0–16, draft counts 0–8). All match. Compile with `rustc -O amd/bench/history_tail.rs -o history_tail` and run it for equivalence plus isolated CPU timings.

The attached timing screen ran on the local workstation, not the rented MI325X host. It indicates the context-length scaling of the removed copy, and does not establish a whole-model throughput gain. Four slots with 4K histories cost around 3 microseconds in the old calculation; at 256K it is around 300 microseconds. The suffix calculation is about 0.4 microseconds in this screen.
