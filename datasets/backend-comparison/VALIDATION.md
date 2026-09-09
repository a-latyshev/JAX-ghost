# Validation record — 2026-09-09

The benchmark was executed in allocation 5878288 on iris-169. Production
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

## Final campaign outcome

`results/20260909T075352Z` completed all 72 planned full launches, plus all 12
small validation launches. All 12 small cases and 69 full cases passed. All three
n99/four-rank mpi4jax attempts aborted in XLA GPU fusion cost analysis with
`shape_util.cc:757: shape.IsArray() || shape.IsTuple()`. This is the same
failure recorded in `../benchmark/JAX-analysis/README.md`, not a successful
measurement. The campaign returned status 1 because it retains these failures
and the known regression failure; all available reports were generated.

The final audit checked all 28 size/rank/trial fixture groups, recomputed every
successful timing median, verified CPU/GPU assignment and matching JAX versions,
and confirmed unchanged worker/launcher source hashes. All eight JAX smoke logs
show exactly one matvec compilation per rank; see `compilation-audit.json`.
The maximum full-run global L2 error was 5.373506661414258e-13.

CSV, Markdown, and five PNG/PDF figure pairs were generated. Runtime and startup
figures were visually inspected. Results are summarized in [RESULTS.md](RESULTS.md).
