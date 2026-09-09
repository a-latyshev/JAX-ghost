# Results — allocation 5878288, 9 September 2026

Fresh measurements on iris-169: one physical CPU core per rank for every backend, plus one distinct V100 GPU per JAX rank. All three use the same per-trial 3D Poisson fixtures. Times are µs per matvec, from 100-call Python loops timed with MPI.Wtime; lower is better. No outer-loop JIT.

| DoFs | MPI ranks | DOLFINx (CPU) | mpi4jax (GPU) | Sharding (GPU) |
| ---: | ---: | ---: | ---: | ---: |
| 103,823 | 1 | 2016.39 | 101.28 | 137.12 |
| 103,823 | 2 | 1022.57 | 707.42 | 237.42 |
| 103,823 | 3 | 731.93 | 824.21 | 209.89 |
| 103,823 | 4 | 579.51 | 831.46 | 207.47 |
| 1,000,000 | 1 | 28961.97 | 347.79 | 381.38 |
| 1,000,000 | 2 | 14387.31 | 740.05 | 268.88 |
| 1,000,000 | 3 | 10295.93 | 850.86 | 228.31 |
| 1,000,000 | 4 | 7629.50 | compilation failed | 226.78 |

mpi4jax is faster than sharding on one GPU for both sizes. At matching multi-GPU counts with successful measurements, sharding is 2.75–4.01× faster than mpi4jax. DOLFINx beats mpi4jax at three and four ranks for the 103,823-DoF matrix.

For the smaller matrix, the fastest measured configuration is one-GPU mpi4jax (101.28 µs). For one million DoFs, sharding on three or four GPUs is fastest (228.31 and 226.78 µs); the four-GPU improvement over three is small.

Compilation is not negligible: median compilation phases range from 113 to 522 ms, with tracing/lowering adding 21–76 ms. Sharding has a separate first-execution cost of roughly 0.57–1.38 s at multiple GPUs. These startup costs are excluded from the warmed timings above and are reported independently; they matter when comparing a single cold batch.

All 12 smoke launches and 69 full launches passed numerical validation. Maximum recorded full-run global L2 error was 5.37e-13. All three million-DoF/four-rank mpi4jax trials failed in XLA GPU fusion cost analysis (`shape.IsArray() || shape.IsTuple()`), matching the documented historical failure; no timing is substituted.

The required MPI regression runner reproduced the existing two-rank float32 tolerance failure and stopped there. Five new pytest audit tests passed. Worker source hashes remained unchanged during the final campaign; compilation logs show exactly one matvec compilation per rank in every JAX smoke launch.

These results compare the installed stacks with matched CPU/GPU placement. Their different MPI/runtime environments remain part of the comparison.

[Full report](results/20260909T075352Z/REPORT.md) · [CSV](results/20260909T075352Z/summary.csv) · [Validation](VALIDATION.md)

![Runtime comparison](results/20260909T075352Z/matvec.png)

![Startup phases](results/20260909T075352Z/startup.png)
