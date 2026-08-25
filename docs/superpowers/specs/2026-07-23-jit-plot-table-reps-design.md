# jit_plot -t: show repetitions on separate lines

## Problem

`jit_plot -t` prints a per-node table (glass | gekko | pfs) from a `result.json`.
`totals_by_node()` collapses every entry for a given `(nodes, mode)` down to
whichever has the latest timestamp. When a node count has more than one rep
(e.g. a disk run then a mem (`-m`) run of the same mode, or a plain 3x variance
run), every rep except the most recent is silently discarded from the table —
there is no way to see them without opening `result.json` by hand.

This surfaced directly from the 2026-07-23 IOR mem-vs-disk validation
(`ior/nodes_9`, `ior/nodes_19`), where each of glass and gekko has two reps
(disk, then mem) at the same node count, and the table showed only the mem run.

## Design

Add a new flag, `-r`/`--reps`, to `jit_plot -t`. Without it, behavior is
unchanged (latest-per-mode wins, as today). With it, every rep is shown.

### `totals_by_node()`

Gains a `show_reps: bool = False` parameter.

- `show_reps=False` (default): unchanged — latest entry per `(nodes, label)`.
- `show_reps=True`: returns every entry per `(nodes, label)`, sorted by
  timestamp ascending, instead of collapsing to one.

### Rep labeling

For a `(nodes, label)` group with more than one entry:

- If `settings["node local"]` differs across the entries, label them by that
  axis: `True` → `disk`, `False` → `mem` (matches the exact case that
  motivated this).
- Otherwise (e.g. a plain repeated-for-variance run), fall back to a plain
  ordinal: `rep 1`, `rep 2`, ...

A group with exactly one entry gets no label (same look as today).

### Table layout

One row per `(nodes, rep-index)` instead of one row per `nodes`. Rows are
aligned by rep-index across glass/gekko/pfs (rep-index 1 = each mode's first
entry by timestamp, rep-index 2 = each mode's second entry, etc.). A new `run`
column follows `nodes`, holding the label described above (blank if the group
at that node count has only one rep for every mode present).

A mode with fewer reps than another at the same node count (e.g. `pfs`, which
never takes `-m` and so never has a second rep) shows `-` for its row past its
own last rep, not `FAIL` — `FAIL` stays reserved for a run that started
(has a log dir) but never produced a result.

The existing green/yellow smallest/largest-per-row highlighting and the
GLASS-goal-hit bold-green `nodes` cell continue to apply per row, now scoped
to that row's rep instead of the whole node count.

### Out of scope

- No change to `totals_by_node()`'s default (no-flag) behavior or to any other
  `jit_plot` mode (plotting, `-n`/`--no_diff`).
- No change to how `compose_log_dir()` assigns `rep_N` on disk — this is a
  read-only, display-side change.
- No attempt to label reps by anything other than `node local`; if a future
  need arises to distinguish reps by some other setting, that is a separate
  design.

## Testing

Unit tests in `test/` (per repo convention, pure-logic, no cluster/BSC
dependency) covering:

- `totals_by_node(show_reps=True)` returns all entries per `(nodes, label)`,
  ordered by timestamp, for a synthetic multi-rep `result.json` fixture.
- Rep labeling: disk/mem label applied when `node local` differs; falls back
  to `rep N` when it doesn't; no label when there is only one entry.
- `print_totals_table(..., show_reps=True)` row count and `run` column content
  for a small fixture (glass with 2 reps, gekko with 2 reps, pfs with 1).
- `-r`/`--reps` argparse wiring: absent → old behavior; present → new behavior.
