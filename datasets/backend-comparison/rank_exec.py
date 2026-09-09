"""Pin before importing numerical libraries, identically for both MPI stacks."""
import os
import sys

rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
cpus = os.environ['BENCH_CPU_IDS'].split(',')
os.sched_setaffinity(0, {int(cpus[rank])})
if sys.argv[1] == 'dolfinx':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['BENCH_GPU_UUIDS'].split(',')[rank]
os.execv(sys.executable, [sys.executable, *sys.argv[2:]])
