#!/usr/bin/env bash
#
# Run the whole suite twice: double-precision rte3d against the double-precision
# Fortran reference, then single against single. Fails if either run fails.
#
#   ./tests/run_all.sh [pytest arguments]
#
# The build directories default to build/ and build_cpu_sp/; override them with
# RTE3D_BUILD_DP and RTE3D_BUILD_SP. The references are the ones
# tests/build_reference.sh writes, dp and sp; a missing one makes its comparisons
# skip, as they do for a bare pytest run.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
build_dp="${RTE3D_BUILD_DP:-$root/build}"
build_sp="${RTE3D_BUILD_SP:-$root/build_cpu_sp}"

ext=so
[[ "$(uname)" == "Darwin" ]] && ext=dylib

run() {
    local name="$1" build="$2" ref="$3"
    echo "=== $name: $build, reference $ref"

    local env=(RTE3D_PYTHON_PATH="$build/main_python")
    [[ -f "$ref/librte_kernels.$ext" ]] && env+=(RTE3D_FORTRAN_REF="$ref/librte_kernels.$ext")
    [[ -f "$ref/librte3d_shim.$ext" ]] && env+=(RTE3D_FORTRAN_SHIM="$ref/librte3d_shim.$ext")

    env -u RTE3D_FORTRAN_REF -u RTE3D_FORTRAN_SHIM "${env[@]}" \
        python -m pytest "$root/tests" "${@:4}"
}

status=0
run "double precision" "$build_dp" "$root/build/reference" "$@" || status=1
run "single precision" "$build_sp" "$root/build/reference_sp" "$@" || status=1

[[ $status == 0 ]] && echo "=== both precisions passed" || echo "=== FAILED"
exit $status
