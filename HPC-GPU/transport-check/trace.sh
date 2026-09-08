#!/usr/bin/env bash
set -eu
here=$(cd -- "$(dirname -- "$0")" && pwd)
python_bin=${PYTHON:-python}
command -v gdb >/dev/null || { echo 'GDB unavailable: transport result UNKNOWN' >&2; exit 2; }
exec gdb -batch -return-child-result -x "$here/trace.gdb" --args "$python_bin" -u "$here/check.py" "$@"
