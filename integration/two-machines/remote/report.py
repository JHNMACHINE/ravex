"""Turn the timing JSONL into the three answers GPU-81 asks for.

Medians, not means: one transfer that met a stall says more as an outlier than
folded into an average, and the outliers are printed separately underneath.
"""

import json
import statistics
import os
import sys
from collections import defaultdict


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


_CHECKPOINT_EVERY = int(os.environ.get("EVERY", "2"))

records = defaultdict(list)
for path in sys.argv[1:]:
    # timing.control.rank0.jsonl and single.control.rank1.jsonl both name the
    # arm in the component before the rank. Keyed off the prefix instead, the
    # single-box files all collapsed into one bucket and the two arms were
    # averaged together — a report that reads fine and says nothing.
    parts = os.path.basename(path).split(".")[:-1]
    if parts and parts[-1].startswith("rank"):
        parts = parts[:-1]
    arm = parts[-1] if parts else "unknown"
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records[arm].append(json.loads(line))

if not records:
    print("  no timing records - was RAVEX_TIMING_OUT set for the run?")
    raise SystemExit(0)

steps = {}
for arm, rows in sorted(records.items()):
    kinds = defaultdict(list)
    for row in rows:
        kinds[row["kind"]].append(row)

    print(f"\n  [{arm}]")

    step_seconds = [r["seconds"] for r in kinds.get("step", [])]
    if step_seconds:
        steps[arm] = statistics.median(step_seconds)
        print(f"    step               {steps[arm] * 1000:8.0f} ms  median of "
              f"{len(step_seconds)}")

    merges = [r["seconds"] for r in kinds.get("merge_now", [])]
    if merges:
        print(f"    merge_now          {statistics.median(merges) * 1000:8.0f} ms  "
              f"median of {len(merges)}, max {max(merges) * 1000:.0f} ms"
              "   <- measure 2, and it is on the training thread")

    exchanges = kinds.get("exchange", [])
    if exchanges:
        seconds = [r["seconds"] for r in exchanges]
        sent = statistics.median([r["sent_bytes"] for r in exchanges])
        median = statistics.median(seconds)
        rate = sent / median / 1e6 if median else 0
        failed = [r for r in exchanges if not r["ok"]]
        print(f"    exchange_stores    {median * 1000:8.0f} ms  median of "
              f"{len(seconds)}, max {max(seconds) * 1000:.0f} ms"
              "   <- measure 1")
        print(f"    store sent         {human(sent):>11s}  -> {rate:.0f} MB/s "
              "effective, chunked at 4 MiB")
        if failed:
            print(f"    ** {len(failed)} exchange(s) reported an incomplete arrival **")

        # The question the number exists to answer: does the copy fit inside the
        # gap between copies. Below 1 it fits; above 1 the replication is
        # backpressure on the training loop, which is GPU-55 again.
        # The control arm's step time is the honest denominator: the replicated
        # arm's already has the transfer inside it, so using it would compare a
        # transfer against an interval the transfer had stretched.
        step = steps.get("control") or steps.get(arm)
        if step:
            for every in (1, 2, 5, 10):
                interval = every * step * _CHECKPOINT_EVERY
                verdict = "fits" if median < interval else "DOES NOT FIT"
                print(f"      replicate_every={every:<3d} interval "
                      f"{interval:6.1f} s vs {median:5.1f} s transfer   {verdict}")

if "control" in steps and "replicated" in steps:
    delta = steps["replicated"] - steps["control"]
    print(f"\n    replication costs {delta * 1000:+.0f} ms per step "
          f"({delta / steps['control']:+.0%} on the loop), the two arms run "
          "back to back")
