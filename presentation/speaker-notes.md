# From distributed FEM to JAX GPU ghost exchange

Eight slides, numbered 0–7. Target duration: 11½ minutes.

## Slide 0 — Finite elements → distributed algebra

Timing: 0:40.

Finite elements split the domain into cells and express the approximate solution
as a weighted sum of local basis functions. Those weights are the degrees of
freedom. For the scalar continuous P1 example used next, each vertex has one
DoF; that vertex picture does not apply to all finite elements. A variational
form and assembly produce a sparse algebraic system because basis functions
have local support. Boundary conditions and the specific PDE are omitted here.

FEniCSx is the finite-element computing platform; DOLFINx is its computational
core and problem-solving interface, handling meshes, function spaces and
distributed assembly. We retain its ownership conventions when moving numerical
operations into JAX. Transition: how are those DoFs distributed across ranks?

Reference: https://docs.fenicsproject.org/dolfinx/main/python/

## Slide 1 — One mesh, two MPI ranks

Timing: 1:35.

Read the left diagram as a 2D triangular mesh, with two cell columns assigned
to rank 0 and one to rank 1. The global vertex labels run from 0 to 11. The
shared interface contains DoFs 2, 6 and 10. We deliberately assign 2 and 10 to
rank 0 and 6 to rank 1 so that forward copies travel in both directions. This
is a valid illustrative ownership convention, not a captured DOLFINx partition.
Real ownership and local ordering must come from the DoF IndexMap.

The separated diagrams show cells owned by each rank, without ghost cells.
A filled node is authoritative on this rank; an outlined node is a cached
coefficient owned remotely. Rank 0 stores eight owned entries and ghost 6;
rank 1 stores four owned entries and ghosts 2 and 10. Local storage is flat,
even though the geometric mesh is two-dimensional. Outlines retain the owner's
color, so the arrows can be followed directly from source to destination.

A forward INSERT copies current owner values into ghost slots, preserving
owned entries. Mesh ghost cells are distinct: additional cells stored locally
for algorithms requiring neighboring cell geometry/topology. DoF ghosts can be
needed by owned cells even when extra ghost cells are not shown. Reverse ADD
is a different operation that accumulates contributions into owners.

Repository: ../src/jaxghost/_metadata.py; ../src/jaxghost/jaxghost.py.
Reference: https://docs.fenicsproject.org/dolfinx/main/python/demos.html#mesh-partitioning-and-parallel-communication-analysis

## Slide 2 — How do we preserve this on GPUs with JAX?

Timing: 0:45.

These are exactly the local arrays from the previous slide, now held on GPUs.
We keep one MPI process per local JAX device, matching the current library's
supported layout. Rank 0 has nine local scalar entries and rank 1 has six.
Moving the vector to a GPU does not establish any ghost synchronization.

We need the same owner-to-ghost copy operation as in DOLFINx, callable within
a JIT-compiled numerical computation. DOLFINx supplies the layout at setup;
the repeated numerical values are JAX arrays. The plan must survive tracing
and account for irregular neighbor lists and unequal sizes without changing
the semantic meaning of owned and ghost entries. We will first illustrate
why matvec needs this exchange, then compare the two implemented mechanisms.

Repository: ../README.md (Current scope); ../src/jaxghost/sharded.py.

## Slide 3 — A local matrix row needs a remote vector value

Timing: 1:45.

This is a deliberately smaller algebraic toy, not the stiffness matrix of the
twelve-vertex mesh on the preceding slides. Use zero-based indices throughout.
Rank 0 owns rows and vector entries 0 and 1; rank 1 owns 2 and 3. The matrix has
2 on its diagonal and −1 on its first sub- and super-diagonals. With x equal to
[1, 2, 4, 8], the full result is [0, −1, −2, 12].

Focus on row 1: its coefficient in column 2 multiplies x₂, which lives on rank
1. Rank 0 therefore receives a copy of 4 and stores [1, 2 | 4]. Row 2 has the
symmetric need for x₁, so rank 1 stores [4, 8 | 2]. The outlined vector boxes
retain the remote owner's color. The matrix's nonlocal-column coefficients
are already stored with their owned row; it is x that needs refreshing.
An implementation remaps global columns to positions in its local extended x.

After the forward update, each rank computes its owned output rows. Setting
initial y to zero makes our owned y += Ax API look like y = Ax in this example.
The returned array preserves output ghost entries; an additional forward
exchange is needed if a later operation requires updated y ghosts. The input
x is not mutated even though an internally refreshed version feeds matvec.
Assembly's reverse accumulation is distinct from the forward exchange shown.

Repository: ../src/jaxghost/matrix.py; ../src/jaxghost/sharded_matrix.py.

## Slide 4 — mpi4jax: pack → sendrecv → unpack

Timing: 1:25.

JAXGhost builds an immutable communication layout from the DOLFINx IndexMap.
The one-time metadata phase uses mpi4py. At runtime, JAX gathers selected owned
entries into a packed buffer, loops over its sorted peers, and invokes
mpi4jax.sendrecv with the matching source, destination and buffer sizes. The
received parts are concatenated and assigned to precomputed ghost positions.
The picture follows rank 0 in the algebraic toy: send 2, receive 4.

mpi4jax lowers MPI operations into the compiled JAX computation through its
native bridge; the Python call describes an operation to execute at runtime.
Our pinned 0.9.1.post1 API returns the received array and uses ordered effects,
which also retain send-only exchanges. All ranks must participate in compatible
order with matching types and counts. Runtime communication uses a duplicated
communicator separate from host metadata traffic.

With host staging, mpi4jax copies between GPU and CPU buffers around MPI.
CUDA-aware mode instead passes device pointers to a compatible MPI stack.
That does not prove the MPI transport avoids host copies. In the recorded
IRIS investigation, both a working TCP route and a later working UCX small-
message route staged payloads inside MPI/UCX. That is a configuration-specific
observation, not a universal claim about UCX. We make no GPUDirect assertion.

Repository: ../src/jaxghost/jaxghost.py; ../HPC-GPU/transport-investigation.md.
References:
https://mpi4jax.readthedocs.io/en/latest/installation.html
https://mpi4jax.readthedocs.io/en/latest/sharp-bits.html

## Slide 5 — The challenge: make the whole HPC stack agree

Timing: 1:25.

The connecting lines indicate compatibility or linkage requirements, not the
order of data movement. DOLFINx and PETSc share a native MPI stack with mpi4py.
mpi4jax must use the intended MPI installation and a supported JAX/jaxlib
combination. GPU execution adds CUDA discovery, the installed GPU backend and
driver compatibility. MPI then selects a transport such as UCX at runtime.
On HPC, module order, Spack prefixes, virtual environments and launcher settings
all affect the libraries actually loaded.

Two concrete issues from our work make this less abstract. First, mpi4jax
0.9.1.post1 searched for NVIDIA wheels beside mpi4py. mpi4py was supplied by
Spack while CUDA wheels were in the virtual environment; the build missed CUDA
and needed a version-specific discovery patch. This is a historical project
example, not a claim that every current release has that bug.

Second, the selected UCX 1.15.0 installation exposed no CUDA memory domain or
CUDA modules on the GPU node. The observed send path tried a CPU access to a
device buffer and crashed. A working TCP configuration passed numerical checks
but staged internally. A subsequently tested CUDA-enabled UCX configuration
also passed; the traced small messages still used host staging. Thus the lesson
is to verify the selected stack and real exchange, not just package imports
or the top-level Open MPI CUDA-support flag. Neither issue required changing
the owner/ghost semantics.

Repository: ../HPC-GPU/notes.md (CUDA-discovery fix);
../HPC-GPU/transport-investigation.md (initial and subsequent configurations).
Reference: https://mpi4jax.readthedocs.io/en/latest/installation.html

## Slide 6 — Explicit ghost exchange with JAX sharding

Timing: 2:25. Elapsed speaking time through this slide: 10:00.

Reuse the twelve-vertex mesh's exact local order. Rank 0 has eight owned values
and one ghost: nine entries. Rank 1 has four owned values and two ghosts: six
entries, padded to nine. The conceptual global array has shape [2, 9], with its
first axis sharded across the two devices. It is not a dense representation of
the original global FEM vector: the rows contain local storage including ghost
duplicates and padding. Each process holds its own addressable row, not both.

The setup phase uses DOLFINx IndexMap ownership and mpi4py metadata exchanges.
It computes globally consistent padded lengths and fixed-width peer packets,
then places send indices, receive positions and validity masks into sharded
JAX arrays. Initialize distributed JAX before querying devices and align mesh
positions with MPI ranks. DOLFINx is used at preparation, and is not needed for
replay of already exported fixtures; mpi4py remains used for setup/bootstrap.

At runtime, shard_map runs the same device body on each shard, with rank-specific
tables as operands. Gather the required owner values and mask unused entries.
The native lax.all_to_all collective routes fixed-width packets by destination.
Rank 0 sends u₂ and u₁₀; rank 1 sends u₆ plus one padded item. Self and unused
peer slots also exist in the all-to-all layout. Receive-position tables route
valid data into the original ghost slots. Invalid positions use an out-of-bounds
sentinel with scatter mode='drop'. Owned values and array padding are preserved.

ShardedJAXMatrixCSR.mult then computes owned y += Ax using local scalar CSR
storage. It preserves the caller's arrays and output ghosts, and does not
overlap communication with multiplication. There are two distinct paddings:
the vector padding drawn in gray, and fixed-width peer packet padding. This
implementation uses all_to_all, not ragged_all_to_all or an automatic halo
inference mechanism. The padded representation and all-rank collective cost
are the tradeoff for fixed shapes and native JAX communication.

The repeated numerical exchange no longer uses the mpi4jax bridge or a runtime
MPI communicator in our operator. This does not remove all MPI dependencies
from setup or establish a physical GPU transport. Forward INSERT and scalar
CSR matvec are implemented; reverse ADD is provided by the MPI backend but not
this sharded backend. Differentiation is outside the supported interface.

Repository: ../src/jaxghost/sharded.py; ../src/jaxghost/sharded_matrix.py;
../datasets/test-sharded-gpu/results/matrix-timing-20260909T052640Z/REPORT.md.
Reference: https://docs.jax.dev/en/latest/notebooks/shard_map.html

## Slide 7 — Newer JAX APIs: optimization opportunities

Timing: 1:30. Total planned speaking time with the extra slide: 11:30.

This is a roadmap, not a report of optimizations already implemented. These
are capabilities in current JAX documentation; they were not all introduced
in the same release. The supplied audit reports small CPU probes on JAX 0.10.2,
not complete GPU or transformation validation. We did not rerun those probes
or change the numerical backend while preparing this slide. Check the actual
JAX/jaxlib and GPU stack before each experiment.

LESS PADDING. Our existing all-to-all allocates one maximum-width numerical
packet per peer. The mini-diagram illustrates the same three useful values
inside eight packet slots versus a packed representation; it is not a measured
size reduction for the earlier two-rank mesh. ragged_all_to_all exchanges slices
using counts and offsets, matching the DOLFINx-derived plan. Zero counts can
represent non-neighbors. With our uniform shards, capacities remain fixed:
use the maximum total outgoing count rather than rank count times the largest
peer message. Metadata and vector/CSR padding remain. output_offsets specifies
where each outgoing slice lands on its receiver, so receiver offsets must be
exchanged during setup before unpacking into original ghost order. Validate
GPU lowering and compare latency, not just payload size; the supplied audit's
CPU probe could not compile this primitive.

REUSE STORAGE. Donation is the smallest API experiment: permit reuse of the
forward input x or matvec accumulator y when the caller relinquishes it. The
donated input must not be reused afterward; memory reuse is an opportunity
for the compiler, not a promised speedup. Keep the existing preserving API as
the baseline. Mutable Refs offer a separate API with indexed writes to ghost
or owned-output slots. Pass Refs explicitly into shard_map, rather than closing
over them, and perform the indexed updates directly. Wrapping the existing
full-vector functional update in a Ref will not automatically remove its
temporaries. Current Ref documentation notes slower Python dispatch to impure
JIT functions taking Ref inputs; benchmark against donated functional calls.

OVERLAP. Split the local operator into owned-column and ghost-column parts.
The timeline is a target: initiate the exchange, compute the owned-column part,
then use received values for the ghost-column part. psend/precv expose separate
send/receive operations; their GPU semantics map to NCCL communication. They
are not interchangeable with MPI nonblocking begin/end. Matching permutations,
fixed operand shapes, ordering and progress need testing. The supplied audit
found no differentiation rule for psend; do not infer AD support from JIT
support. Also compare ordinary collectives with independent computation and
XLA latency-hiding / profile-guided scheduling. A GPU trace must show overlap.

FOLLOW-UP. Profile local CSR separately: the current JAX sparse module is
experimental and does not promise performance-critical suitability. Consider
BCSR for batched vectors or gather-plus-segment-sum with cached row indices,
keeping existing CSR as the baseline. Pallas/Mosaic GPU offers later kernel-
level communication and computation integration. Its dense collective matmul
example does not establish performance or hardware suitability for our
irregular float64 CSR workload, including the GPUs used in this project.

Suggested order: global no-communication fast path and donation; persistent
Refs; GPU ragged exchange; profile local kernels; test overlap; specialized
kernels only if warranted. The no-communication decision must be global:
a rank with no local ghosts may still need to send to another rank. Retain the
DOLFINx ownership and ghost ordering, and check correctness, memory and total
matvec time for each candidate. Backend execution, JIT, batching and AD are
separate properties to verify.

References (official JAX documentation, checked for this slide):
https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_all_to_all.html
https://docs.jax.dev/en/latest/buffer_donation.html
https://docs.jax.dev/en/latest/array_refs.html
https://docs.jax.dev/en/latest/_autosummary/jax.lax.psend.html
https://docs.jax.dev/en/latest/_autosummary/jax.lax.precv.html
https://docs.jax.dev/en/latest/gpu_performance_tips.html
https://docs.jax.dev/en/latest/jax.experimental.sparse.html
https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html

Repository baseline: ../src/jaxghost/sharded.py;
../src/jaxghost/sharded_matrix.py. The user-supplied API audit informed the
priorities; its probe outcomes are attributed above rather than revalidated.
