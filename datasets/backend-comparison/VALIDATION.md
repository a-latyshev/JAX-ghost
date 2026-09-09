# Validation record — 2026-09-09

The benchmark is being exercised in allocation 5878288 on iris-169. Production
source code is unchanged outside this benchmark folder.

## Checks completed

- Python syntax compilation and shell syntax checks passed.
- Five pytest audit tests passed in mpc-v10 and its JAX venv. They cover maximum
  rank time before median aggregation and rejection of different fixtures,
  duplicate GPUs, different CPU assignments and incorrect stored medians.
- The required `scripts/run_mpi_tests.py` runner reproduced the previously
  documented two-rank float32 `test_matrix_dolfinx` failure: 27 passed, 1 failed
  per rank; maximum violating absolute difference 9.059906e-06. The compiled
  MPI smoke check and one-rank tests passed. The runner stops at that failure.
  This matches the failure already recorded in
  `../benchmark/JAX-analysis/MATVEC-LOOP.md`; public library code is unchanged.

## Implementation diagnostics retained

- `results/20260909T074745Z`: preflight stopped because the compute node has
  system Python 3.6. The orchestrator now uses compatible subprocess arguments.
- `results/20260909T074802Z` and `results/20260909T074919Z`: initial activation
  checks exposed overwritten Spack PYTHONPATH. The clean launcher now preserves
  paths created by its own Spack activation and uses explicit venv interpreters.
- `results/20260909T075057Z`: direct AOT executable calls passed their first
  execution but failed on repeated multi-rank mpi4jax calls with
  `Execution supplied 3 arguments but compiled program expected 4`. Both JAX
  backends now execute through the cached `jax.jit` entrypoint after explicit
  lowering and compilation. No outer-loop JIT is used. Failed launches are kept
  and excluded from the production comparison.

The main campaign is `results/20260909T075352Z`. Its manifest records the exact
worker/launcher source hashes at launch, physical CPU/GPU assignments, and source
revision. The audit test was converted to pytest during earlier diagnostics;
production worker source is frozen for this campaign.
