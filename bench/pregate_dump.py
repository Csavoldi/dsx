#!/usr/bin/env python3
"""pregate_dump.py DUMP [DUMP...]: what prefetching layer L+1's predicted CPU experts during its attention would catch.

DUMP is DSX_PREGATE_DUMP's output (DSX_PREGATE=1, decode): per token and layer "pos layer | predicted top-8 in score
order | actual 6 | where each of those 14 is" (c = CPU worker, 0 = GPU0, 1 = an aux GPU, ? = none). For a copy budget
of m experts per layer, the copies are the m highest-scored predictions that only the CPU holds. Reported: the share of
the CPU's actual picks the copies catch (f, the work that would move to GPU0), copies per token, the share of copies
that were wasted (not picked), and f per layer."""
import sys
from collections import defaultdict


def parse(paths):
    for p in paths:
        for line in open(p):
            parts = line.split("|")
            if len(parts) != 4:
                continue
            pos, layer = map(int, parts[0].split())
            pred = list(map(int, parts[1].split()))
            act = list(map(int, parts[2].split()))
            at = parts[3].strip()
            yield pos, layer, pred, act, at[:8], at[8:]


def main():
    rows = list(parse(sys.argv[1:]))
    if not rows:
        sys.exit("no rows")
    tokens = len({(r[0]) for r in rows})
    cpu_picks = sum(a.count("c") for *_, a in rows)
    all_picks = sum(len(r[3]) for r in rows)
    print(f"{tokens} tokens, {len(rows)} (token, layer) rows; CPU picks {cpu_picks} = {cpu_picks / all_picks:.1%} of picks, "
          f"{cpu_picks / tokens:.1f} per token, {cpu_picks / len(rows):.2f} per layer")
    in8 = sum(1 for _, _, pred, act, _, aa in rows for e, w in zip(act, aa) if w == "c" and e in pred)
    print(f"CPU picks in the predicted top-8: {in8 / cpu_picks:.1%}")
    for m in (1, 2, 3, 4):
        caught = copies = 0
        per_layer = defaultdict(lambda: [0, 0])
        for _, layer, pred, act, pa, aa in rows:
            chosen = [e for e, w in zip(pred, pa) if w == "c"][:m]
            copies += len(chosen)
            cpu_act = [e for e, w in zip(act, aa) if w == "c"]
            hit = sum(1 for e in cpu_act if e in chosen)
            caught += hit
            per_layer[layer][0] += hit
            per_layer[layer][1] += len(cpu_act)
        pl = " ".join(f"{l}:{h / max(n, 1):.2f}" for l, (h, n) in sorted(per_layer.items()))
        print(f"m={m}: f = {caught / cpu_picks:.1%} of CPU picks caught, {copies / tokens:.1f} copies per token, "
              f"{1 - caught / max(copies, 1):.1%} of copies wasted\n  f per layer: {pl}")


if __name__ == "__main__":
    main()
