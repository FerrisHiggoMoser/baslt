# Performance

Requirement checks are vectorized with numpy: every requirement is a handful of whole-array operations on its
clock, never a loop over samples. `benchmarks/bench_check.py` measures the targets below; `--quick` runs a tenth of
each size.

```sh
python benchmarks/bench_check.py            # full size, about a minute
python benchmarks/bench_check.py --quick    # smoke run
```

## Targets and measurements

Measured with `bench_check.py` on a 24-CPU Linux machine (Python 3.14, numpy 2.5, NVMe disk):

| Scenario | Target | Measured |
|---|---|---|
| Lint a 2,000-row requirements table | 1 s | 0.06 s |
| One run: 120 s at 1 kHz, 50 signals, 200 requirements (evaluation) | 1.5 s | 0.55 s |
| The same with the report, the workbook and the checked copy | 3 s | 0.67 s |
| A long run: 10 million samples × 20 signals, 50 requirements | 30 s | 7.3 s |
| `results.xlsx` for 1,000 runs × 200 requirements | 3 s | 0.85 s |
| `index.html` for 1,000 runs × 200 requirements | 10 s | 4.3 s |
| A batch of 1,000 simulated ascents (200 Hz HDF5), `--jobs auto` | 4 min | 2.8 s |
| The same batch again with `--resume` | 5 s | 0.95 s |

The requirement mix is limit checks with windows, tolerances and margins, `max` aggregates, durations of
comparisons, time-weighted means and asserts (`bench_check.requirement_rows`).

## Where the time goes

- **Reading.** A check loads only the signals its requirements, events and conditions read; the rest of the file is
  never touched. HDF5 and v7.3 MAT data that is stored contiguously is memory-mapped.
- **Clocks.** Signals that share a clock are used as they are. Others are resampled once per (signal, clock) pair and
  kept in a cache of at most 512 MiB, as are expression results per (expression, clock), so conditions shared by many
  requirements (`ascent`, `high_q`) are computed once.
- **Events.** Each event is detected once per run.
- **Pages.** Plot traces keep at most about 1,000 points per signal (the extremes of 512 bins) plus full resolution
  around failures, and a signal plotted by several requirements is stored once, so report pages stay near 100 KB for
  the rocket example and under 700 KB for 200 requirements. The dashboard keeps 256-bin envelopes per run and signal,
  8-bit traces for at most 32 runs per signal (more only for failures) and one factor instead of a percentage column
  when a requirement has one limit: about 0.8 MB for 1,000 runs × 50 requirements and 2.3 MB for 1,000 × 200
  (tested in `test_dashboard_size.py`).
- **Batches.** Each run is checked in its own process with one numeric thread and a lower priority, so a batch uses
  the free cores without slowing the machine's other work. Results are written as runs finish; `--resume` compares
  a key of the run file (path, size, modification time), the requirements, mapping and parameters, and skips runs
  whose key matches.

## Memory

A check holds the loaded signals, their resampled copies on the requirements' clocks and the expression cache. The
10-million-sample scenario (1.6 GB of signal data) peaked at 6.3 GB. For long runs, give requirements the clock of
the signal they check (the default `grid: first`) rather than `grid: union`, and keep `--jobs` below the number of
runs that fit in memory at once.
