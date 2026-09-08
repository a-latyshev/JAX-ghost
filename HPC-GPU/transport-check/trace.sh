#!/usr/bin/env bash
set -eu
here=$(cd -- "$(dirname -- "$0")" && pwd)
python_bin=${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
python_bin=${python_bin:-python}
[[ -r "$here/trace.gdb.in" ]] || { echo 'Missing trace.gdb.in; update/copy the complete transport-check directory.' >&2; exit 2; }
"$python_bin" - <<'PYCHECK'
import importlib.util
import sys
missing = [name for name in ('numpy', 'mpi4py', 'jax', 'mpi4jax')
           if importlib.util.find_spec(name) is None]
if missing:
    sys.exit(f"{sys.executable}: missing {', '.join(missing)}. "
             "Activate the GPU venv and set PYTHON to its bin/python.")
PYCHECK
command -v gdb >/dev/null || { echo 'GDB unavailable: transport result UNKNOWN' >&2; exit 2; }
trace_log=$(mktemp)
trap 'rm -f "$trace_log"' EXIT
set +e
gdb -batch -return-child-result -x "$here/trace.gdb.in" --args "$python_bin" -u "$here/check.py" "$@" 2>&1 | tee "$trace_log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
status=${pipeline_status[0]}
if (( status == 0 && pipeline_status[1] != 0 )); then
    status=${pipeline_status[1]}
fi
"$python_bin" "$here/summarize.py" "$trace_log" "$status"
