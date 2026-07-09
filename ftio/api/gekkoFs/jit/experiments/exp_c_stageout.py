"""
Experiment C — stage-out variance decomposition (FTIO decision vs Lustre copy).

Consumes the flush log that GLASS already writes during a run
(`posix_control._write_flush_log`): one line per staged-out item with who
triggered it (FTIO predictor vs post-app drain), the Lustre copy time, and the
delete time. From that we decompose where stage-out time — and its variance —
goes: the FTIO-triggered path (overlapped with compute) vs the post-app path,
and copy (Lustre) vs delete.

Log line format (from posix_control._write_flush_log):
    <ts> | FTIO-trigger | <item> -> <dst> | copy: <s> s | delete: <s> s
    <ts> | post-app     | <item> -> <dst> | copy: <s> s | delete: <s> s

Run:
    python -m ftio.api.gekkoFs.jit.experiments.exp_c_stageout <flush_log>

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: July 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import argparse
import re
import statistics as stats

_LINE = re.compile(
    r"\|\s*(FTIO-trigger|post-app)\s*\|.*\|\s*copy:\s*([0-9.]+)\s*s"
    r"\s*\|\s*delete:\s*([0-9.]+)\s*s"
)


def parse_flush_log(path: str) -> list[dict]:
    """Parse a flush log into a list of {triggered_by, copy, delete} records."""
    records = []
    with open(path) as fh:
        for line in fh:
            m = _LINE.search(line)
            if not m:
                continue
            records.append(
                {
                    "triggered_by": (
                        "ftio" if m.group(1) == "FTIO-trigger" else "post_app"
                    ),
                    "copy": float(m.group(2)),
                    "delete": float(m.group(3)),
                }
            )
    return records


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "total": 0.0}
    return {
        "n": len(values),
        "mean": stats.fmean(values),
        "std": stats.pstdev(values) if len(values) > 1 else 0.0,
        "total": sum(values),
    }


def decompose(records: list[dict]) -> dict:
    """Decompose stage-out time and variance by trigger source and stage."""
    out = {}
    for group in ("ftio", "post_app", "all"):
        sub = (
            records
            if group == "all"
            else [r for r in records if r["triggered_by"] == group]
        )
        out[group] = {
            "copy": _summary([r["copy"] for r in sub]),
            "delete": _summary([r["delete"] for r in sub]),
        }
    return out


def report(records: list[dict]) -> dict:
    d = decompose(records)
    print(
        f"items: {len(records)} "
        f"(ftio={d['ftio']['copy']['n']}, post_app={d['post_app']['copy']['n']})"
    )
    for group in ("ftio", "post_app", "all"):
        c, x = d[group]["copy"], d[group]["delete"]
        print(
            f"  {group:8s}  copy mean={c['mean']:.3f}s std={c['std']:.3f}s "
            f"total={c['total']:.1f}s | delete mean={x['mean']:.3f}s std={x['std']:.3f}s"
        )
    # variance attribution: copy (Lustre) is the dominant, high-variance stage
    all_copy_var = d["all"]["copy"]["std"] ** 2
    all_del_var = d["all"]["delete"]["std"] ** 2
    denom = all_copy_var + all_del_var
    if denom > 0:
        print(
            f"  variance share -> copy(Lustre): {all_copy_var/denom*100:.1f}%  "
            f"delete: {all_del_var/denom*100:.1f}%"
        )
    return d


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage-out variance decomposition")
    parser.add_argument("flush_log", help="path to the GLASS flush log")
    args = parser.parse_args(argv)
    report(parse_flush_log(args.flush_log))


if __name__ == "__main__":
    main()
