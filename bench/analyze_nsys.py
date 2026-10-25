#!/usr/bin/env python3
"""Summarize an Nsight SQLite export of a bounded, initialization-free capture."""
import argparse
import collections
import json
import pathlib
import sqlite3


def union_ns(intervals):
    total = 0
    end = None
    for lo, hi in sorted(intervals):
        if end is None or lo >= end:
            total += hi - lo
        elif hi > end:
            total += hi - end
        end = hi if end is None else max(end, hi)
    return total


def summarize(path, tokens, token_kind="output"):
    con = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    kernels = [dict(r) for r in con.execute('SELECT start, "end", deviceId, shortName, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL')]
    copies = [dict(r) for r in con.execute('SELECT start, "end", deviceId, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY')] if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
    kinds = {r[0]: r[1] for r in con.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")} if "ENUM_CUDA_MEMCPY_OPER" in tables else {}
    intervals = [(r["start"], r["end"]) for r in kernels + copies]
    if not intervals:
        raise ValueError("capture contains no CUDA activity")
    span = (max(hi for _, hi in intervals) - min(lo for lo, _ in intervals)) / 1e9
    devices = {}
    for dev in sorted({r["deviceId"] for r in kernels + copies}):
        ks = [r for r in kernels if r["deviceId"] == dev]
        ms = [r for r in copies if r["deviceId"] == dev]
        by_kernel = collections.defaultdict(lambda: {"calls": 0, "sum_ns": 0})
        for r in ks:
            entry = by_kernel[names.get(r["shortName"], str(r["shortName"]))]
            entry["calls"] += 1
            entry["sum_ns"] += r["end"] - r["start"]
        by_copy = collections.defaultdict(lambda: {"calls": 0, "bytes": 0, "sum_ns": 0, "max_copy_bytes": 0})
        for r in ms:
            entry = by_copy[kinds.get(r["copyKind"], str(r["copyKind"]))]
            entry["calls"] += 1
            entry["bytes"] += r["bytes"]
            entry["sum_ns"] += r["end"] - r["start"]
            entry["max_copy_bytes"] = max(entry["max_copy_bytes"], r["bytes"])
        for entry in by_copy.values():
            entry[f"bytes_per_{token_kind}_token"] = entry["bytes"] / tokens
            entry[f"sum_ms_per_{token_kind}_token"] = entry["sum_ns"] / tokens / 1e6
        busy = union_ns([(r["start"], r["end"]) for r in ks])
        devices[str(dev)] = {
            "kernel_calls": len(ks),
            f"copy_busy_union_ms_per_{token_kind}_token": union_ns([(r["start"],r["end"]) for r in ms])/tokens/1e6,
            "kernel_and_copy_busy_fraction_of_capture_span": union_ns([(r["start"],r["end"]) for r in ks+ms])/1e9/span,
            f"kernel_calls_per_{token_kind}_token": len(ks) / tokens,
            f"kernel_busy_union_ms_per_{token_kind}_token": busy / tokens / 1e6,
            "kernel_busy_fraction_of_capture_span": busy / 1e9 / span,
            f"kernel_duration_sum_ms_per_{token_kind}_token": sum(r["end"] - r["start"] for r in ks) / tokens / 1e6,
            "top_kernels": [{"name": name, **entry, f"sum_ms_per_{token_kind}_token": entry["sum_ns"] / tokens / 1e6} for name, entry in sorted(by_kernel.items(), key=lambda item: item[1]["sum_ns"], reverse=True)[:20]],
            "memcpy": dict(by_copy),
        }
    return {
        "sqlite": str(path), f"{token_kind}_tokens": tokens, "cuda_activity_span_s": span,
        "devices": devices,
        "limits": "Kernel busy is an interval union, not SM occupancy or bandwidth. Summed kernel times can overlap. Copy bytes exclude mapped-memory accesses made inside kernels. Profiling overhead is not a performance gain or an untraced timing measurement.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=pathlib.Path)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--token-kind", choices=["input","output"], default="output")
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    result = summarize(args.sqlite, args.tokens, args.token_kind)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")
