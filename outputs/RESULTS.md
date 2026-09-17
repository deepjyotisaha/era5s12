# Session 12 results — notebook v1

Model: 3,793,664 parameters (P = 7.24 MiB bf16). World size 32, 20 training steps. Simulator on cpu; GPU check: Tesla T4.

Acceptance: **24/24**

## Memory per rank, N=32 (counted)

| arrangement | persistent MiB | bytes/weight | modelled | peak MiB | peak at |
|---|---|---|---|---|---|
| DP | 57.887 | 16.0000 | 16.0000 | 60.840 | step 1: bwd head |
| ZeRO-1 | 15.828 | 4.3750 | 4.3750 | 18.781 | step 1: bwd head |
| ZeRO-2 | 8.819 | 2.4375 | 2.4375 | 11.772 | step 1: bwd head |
| ZeRO-3 | 1.809 | 0.5000 | 0.5000 | 7.620 | step 1: bwd head |

## Sweep: persistent MiB per rank / bytes sent per rank (×P)

| N | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| 1 | 57.887 / 0.0000 | 57.887 / 0.0000 | 57.887 / 0.0000 | 57.887 / 0.0000 |
| 2 | 57.887 / 1.0000 | 36.179 / 1.0000 | 32.561 / 1.0000 | 28.943 / 1.5000 |
| 4 | 57.887 / 1.5000 | 25.325 / 1.5000 | 19.899 / 1.5000 | 14.472 / 2.2500 |
| 8 | 57.887 / 1.7500 | 19.899 / 1.7500 | 13.567 / 1.7500 | 7.236 / 2.6250 |
| 16 | 57.887 / 1.8750 | 17.185 / 1.8750 | 10.402 / 1.8750 | 3.618 / 2.8125 |
| 32 | 57.887 / 1.9375 | 15.828 / 1.9375 | 8.819 / 1.9375 | 1.809 / 2.9062 |

## Computation per rank per step, N=32

| | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| unit forwards / rank | 6 | 6 | 6 | 6 |
| unit recomputes / rank | 6 | 6 | 6 | 6 |
| unit backwards / rank | 6 | 6 | 6 | 6 |
| Adam elements / rank | 3,793,664 | 118,552 | 118,552 | 118,552 |
| reduce-scatter calls | 1 | 1 | 6 | 6 |
| all-gather calls | 1 | 1 | 1 | 12 |
| bytes sent / rank (xP) | 1.9375 | 1.9375 | 1.9375 | 2.9062 |

## Projection to 30B (GiB per GPU; lesson page in brackets)

| | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|
| DP | 447.0 (447.0) | 447.0 (447.0) | 447.0 (447.0) | 447.0 (447.0) |
| ZeRO-1 | 153.7 (153.7) | 132.7 (132.7) | 122.2 (122.2) | 117.0 (117.0) |
| ZeRO-2 | 104.8 (104.8) | 80.3 (80.3) | 68.1 (68.1) | 62.0 (62.0) |
| ZeRO-3 | 55.9 (55.9) | 27.9 (27.9) | 14.0 (14.0) | 7.0 (7.0) |

## Acceptance

- [PASS] A. ring reduce-scatter + all-gather == all-reduce, bit-exact
- [PASS] A. reduce-scatter alone leaves each rank its exact slice
- [PASS] A. ring bytes == exact expression (even and uneven chunks)
- [PASS] B. DP gradient == big-batch gradient (fp64)
- [PASS] B'. adam_ on a shard == same slice of full-vector update, bit-exact
- [PASS] C. loss fell by at least the declared threshold
- [PASS] C. 32 DP replicas bit-identical
- [PASS] C. ZeRO-1 bit-identical to DP
- [PASS] C. ZeRO-2 bit-identical to DP
- [PASS] C. ZeRO-3 bit-identical to DP
- [PASS] C'. planted bug 'drop_contribution' caught
- [PASS] C'. planted bug 'double_average' caught
- [PASS] D. counted persistent == modelled, every rank (N=32)
- [PASS] D. no leak across steps, no transient survives a step
- [PASS] D. activation bytes identical across arrangements
- [PASS] E. ring bytes == exact expression, every rank, every N
- [PASS] E. ledger == modelled, every rank, every N
- [PASS] E. ZeRO-1 and ZeRO-2 send exactly DP's bytes
- [PASS] E. ZeRO-3 sends exactly 1.5x DP (even shards)
- [PASS] 13. model arithmetic identical across arrangements
- [PASS] 14. whole-tensor partition more imbalanced than flat
- [PASS] F. 30B projection matches the lesson page (16 cells)
- [PASS] F. ZeRO-1 floor exceeds the card; 20B boundary
- [PASS] 16. GPU-measured persistent within 1% of counted
