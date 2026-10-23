#!/bin/bash
# pf102.sh (on .102, inside a window with the dsx server stopped; takes no GPU_LOCK): DSX_PREFETCH (next-layer expert
# prefetch into GPU0) A/B on branch pf-prefetch. Builds dsx, then:
#   1. decode: DSX_PREFETCH 0 / 1 / 2, each without and with DSX_DEFER=3, interleaved, ROUNDS rounds (default 1;
#      one cold `dsx gen` process a run: ~80 s load + prefill + N tokens)
#   2. quality: `dsx eval --decode` (teacher-forced decode), DSX_PREFETCH 0 vs 2: the logits must agree closely
#      (routing is unchanged; a prefetched expert rounds like GPU0's kernels instead of the CPU's)
# 25 s pauses with the package temperature between runs (.102's watchdog stops ds41 scopes at >= 95 C).
# GEN_ARGS: your usual `dsx gen` arguments for the 3K agent prompt and ~400 decode tokens (model, engram, --ids, -n,
# --rank, --gpu*-gb, --arena-gb ...), EVAL_ARGS: `dsx eval` arguments for the Lucebox eval (~/ai/ds41/eval), e.g.
#   GEN_ARGS="--model ... --ids ~/ai/ds41/prompt3k.ids -n 400 ..." EVAL_ARGS="--model ... --eval ~/ai/ds41/eval" \
#     bash bench/pf102.sh > pf102.out 2>&1
# Lines to read: "prefetch:" (setup), "decode N tokens ... tok/s ... % prefetched to GPU0", and eval's KL / top-1.
cd "$(dirname "$0")/.." || exit 1
[ -n "$GEN_ARGS" ] || { echo "set GEN_ARGS (dsx gen arguments) and EVAL_ARGS (dsx eval arguments)"; exit 1; }
echo "serve running: $(pgrep -fa 'dsx.*serve' | grep -v pgrep | wc -l); $(uptime)"
s=$(date +%s); (cd nvidia && ~/.cargo/bin/cargo build --release --features cuda --bin dsx 2>&1 | tail -1); echo "build $(( $(date +%s) - s )) s"
B=target/release/dsx
cool() { sleep 25; echo "  $(sensors 2> /dev/null | grep -m1 Package) $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' ')"; }
for r in $(seq 1 "${ROUNDS:-1}"); do
  for defer in none 3; do
    for pf in 0 1 2; do
      echo "== round $r, DSX_PREFETCH=$pf, DSX_DEFER=$defer"
      if [ $defer = none ]; then E="DSX_PREFETCH=$pf"; else E="DSX_PREFETCH=$pf DSX_DEFER=$defer"; fi
      env $E $B gen $GEN_ARGS 2>&1 | grep -E "^prefetch|^decode|rror|panic"
      cool
    done
  done
done
if [ -n "$EVAL_ARGS" ]; then
  for pf in 0 2; do
    echo "== eval --decode --seqs 2, DSX_PREFETCH=$pf"
    env DSX_PREFETCH=$pf $B eval $EVAL_ARGS --decode --seqs 2 2>&1 | grep -E "^prefetch|KL|kl|top|ppl|perplex|rror|panic" | tail -8
    cool
  done
fi
echo "done $(date)"
