# MutationDesigner benchmark report

_Generated: 2026-08-02 11:02:07_

## Configuration

- **Windows (mut_len):** 1,000, 10,000, 100,000
- **Replicates (nreps / mut_reps):** 20 (fixed)
- **mut_distance_min:** 50 (for `bins` and `star-and-bar`)
- **random design:** rate-based (mut_rate=0.01, num_seq=1000); does not use nreps
- **vanilla design:** single-mutation ISM; ignores nreps; skipped when dense tensor > 2 GB

## Run-time summary (seconds)

| Method | L=1,000 | L=10,000 | L=100,000 |
|---|---|---|---|
| vanilla | 0.00 | 0.05 | skip |
| bins | 0.65 | 2.64 | 40.69 |
| star-and-bar | 0.65 | 1.71 | 15.78 |
| random | 0.13 | 0.38 | 3.29 |

## Peak memory summary (MB, process RSS)

| Method | L=1,000 | L=10,000 | L=100,000 |
|---|---|---|---|
| vanilla | 61 | 180 | skip |
| bins | 61 | 91 | 405 |
| star-and-bar | 62 | 127 | 835 |
| random | 53 | 63 | 151 |

## Design statistics per window

### Window L = 1,000

| Statistic | vanilla | bins | star-and-bar | random |
|---|---|---|---|---|
| # sequences | 3,000 | 6,000 | 4,935 | 1,000 |
| total mutations | 3,000 | 60,000 | 71,320 | 9,883 |
| mut/seq (mean) | 1.0 | 10.0 | 14.5 | 9.9 |
| mut/seq (min) | 1 | 10 | 14 | 2 |
| mut/seq (max) | 1 | 10 | 16 | 20 |
| min gap | - | 51 | 51 | 1 |
| mean gap | - | 100.0 | 71.0 | 89.7 |
| pos covered | 100.00% | 100.00% | 100.00% | 100.00% |
| pos cov (min) | 3 | 60 | 60 | 2 |
| reps/mut (mean) | 1.0 | 20.0 | 23.8 | 3.4 |
| reps/mut (min) | 1 | 20 | 20 | 1 |
| reps/mut (max) | 1 | 20 | 84 | 13 |
| time (s) | 0.00 | 0.65 | 0.65 | 0.13 |
| peak MB | 61 | 61 | 62 | 53 |

### Window L = 10,000

| Statistic | vanilla | bins | star-and-bar | random |
|---|---|---|---|---|
| # sequences | 30,000 | 6,000 | 5,299 | 1,000 |
| total mutations | 30,000 | 600,000 | 791,636 | 100,149 |
| mut/seq (mean) | 1.0 | 100.0 | 149.4 | 100.1 |
| mut/seq (min) | 1 | 100 | 149 | 72 |
| mut/seq (max) | 1 | 100 | 151 | 136 |
| min gap | - | 51 | 51 | 1 |
| mean gap | - | 100.0 | 67.1 | 98.9 |
| pos covered | 100.00% | 100.00% | 100.00% | 100.00% |
| pos cov (min) | 3 | 60 | 60 | 1 |
| reps/mut (mean) | 1.0 | 20.0 | 26.4 | 3.5 |
| reps/mut (min) | 1 | 20 | 20 | 1 |
| reps/mut (max) | 1 | 20 | 92 | 12 |
| time (s) | 0.05 | 2.64 | 1.71 | 0.38 |
| peak MB | 180 | 91 | 127 | 63 |

### Window L = 100,000

| Statistic | vanilla | bins | star-and-bar | random |
|---|---|---|---|---|
| # sequences | skipped | 6,000 | 5,696 | 1,000 |
| total mutations | skipped | 6,000,000 | 8,540,725 | 999,417 |
| mut/seq (mean) | skipped | 1000.0 | 1499.4 | 999.4 |
| mut/seq (min) | skipped | 1000 | 1499 | 879 |
| mut/seq (max) | skipped | 1000 | 1501 | 1101 |
| min gap | skipped | 51 | 51 | 1 |
| mean gap | skipped | 100.0 | 66.7 | 100.0 |
| pos covered | skipped | 100.00% | 100.00% | 100.00% |
| pos cov (min) | skipped | 60 | 60 | 0 |
| reps/mut (mean) | skipped | 20.0 | 28.5 | 3.5 |
| reps/mut (min) | skipped | 20 | 20 | 1 |
| reps/mut (max) | skipped | 20 | 106 | 14 |
| time (s) | skipped | 40.69 | 15.78 | 3.29 |
| peak MB | skipped | 405 | 835 | 151 |

_`vanilla`: skipped (dense tensor ~90.0 GB exceeds cap)_

## Notes

- **min gap** = smallest distance between two mutated positions within a single sequence (spacing guarantee).
- **reps/mut** = how many times each individual (position, nucleotide) mutation appears across all sequences.
- **pos covered** = fraction of the window's positions that receive at least one mutation.
- `vanilla` is single-mutation ISM: one mutation per sequence (min gap is undefined), coverage is exactly 1 per (pos, nt).
- Each configuration ran in an isolated subprocess; peak MB is that process's peak RSS.
