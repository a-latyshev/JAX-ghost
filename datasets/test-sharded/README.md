# CPU scaling: sharded forward updates versus DOLFINx

This directory measures forward INSERT only, at fixed global mesh sizes on
1–4 MPI ranks / JAX CPU devices. It does not benchmark sparse matvec or solvers.

From the repository root in the compatible DOLFINx/JAX environment:

```bash
FI_PROVIDER=tcp FI_TCP_IFACE=en0 python datasets/test-sharded/run.py \
  --local-cpu --output datasets/test-sharded/results/cpu-strong-scaling
python datasets/test-sharded/plot.py datasets/test-sharded/results/cpu-strong-scaling
```

`--local-cpu` is the explicit single-host macOS workaround, using the same
version-specific CPU backend hooks as the sharding test runner. Omit it on a
properly configured distributed environment. The runner forces CPU execution,
one JAX CPU device per rank, and single-thread settings for common compute
libraries; JAX runtime/helper threads remain. Use `--mpirun /path/to/mpirun`
when needed. Do not run concurrent benchmarks. Jobs have a 300-second timeout.

Defaults: square subdivisions 256 and 1024, scalar P1, float64, 3 independent
launches per point, 7 batches of 100 updates after 10 warmups. Compile and first
execute before timing. DOLFINx and JAX use the same concrete partition within
each job. Both are checked against global-ID values; input preservation is
checked for JAX. Change sizes with `--sizes`; keep each size fixed across ranks.
Reusing an output folder overwrites matching raw files; use a new folder for a
new study (omitting `--output` creates a timestamped directory).

[Results and methodology](results/cpu-strong-scaling/REPORT.md) ·
[Scaling plot](results/cpu-strong-scaling/scaling.png) ·
[Vector PDF](results/cpu-strong-scaling/scaling.pdf) ·
[Summary CSV](results/cpu-strong-scaling/summary.csv)

Raw JSON stores per-rank measurements and environment details. The figure shows
median launch medians and their range. One rank is a no-communication control:
ordinary T(1)-based speedup and parallel efficiency are misleading for this
operation. T(2)/T(p) is shown separately for communicating ranks. On this Mac,
these are device-count measurements without fixed CPU-core affinity.
