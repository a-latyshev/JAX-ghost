# Operator roadmap

Implemented foundation: IndexMap-derived forward INSERT and reverse ADD,
fixed scalar/vector/tensor blocks, JIT execution, and CPU correctness tests.

1. [x] Scalar distributed CSR `y += Ax`, using the matrix column IndexMap;
   validate against DOLFINx on 1–4 ranks. GPU validation remains pending.
2. [ ] CPU/GPU benchmarks: measure setup, compilation, transfers, local kernels,
   and full communication-inclusive applications separately. Compare DOLFINx
   CPU, JAX CPU, and JAX GPU at matching precision and partition. Is acceleration
   retained as work per rank decreases? No speedup is assumed.
3. [ ] Transpose matvec with reverse ADD; verify the owned-entry adjoint identity,
   then gradients, including asymmetric exchanges and overwritten ghost inputs.
4. [ ] Blocked CSR with independent row/column block sizes.
5. [ ] Matrix-free mass and diffusion operators; start with affine elements and
   fixed coefficients, then test higher order and varying coefficients. Compare
   memory and application costs against assembled DOLFINx matrices explicitly.
6. [ ] Nonlinear residuals and Jacobian-vector products; distinguish local-kernel
   differentiation from differentiation through distributed communication.
7. [ ] Optional distributed-vector wrapper and sharded-backend experiments;
   evaluate irregular layouts, padding, integration and communication costs.

Communication/computation overlap is a later optimization: first benchmark the
simple forward-scatter-then-matvec baseline. Keep the assembly demonstration in
`experiments/`; matrix assembly and linear solvers are outside the core API.
