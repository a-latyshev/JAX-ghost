#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
if [[ ${1:-} == --inside ]]; then
    shift
    exec /usr/bin/python3 "$here/run.py" "$@"
fi
job=${BENCH_JOB_ID:-5878288}
# One enclosing step exposes the allocated cores/GPUs; rank_exec.py binds workers.
exec srun --jobid "$job" --overlap -N 1 -n 1 -c 28 --cpu-bind=none \
    bash --noprofile --norc "$here/run.sh" --inside "$@"
