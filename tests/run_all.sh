#!/usr/bin/env bash
#
# Run the whole suite once per build: each against the Fortran reference of its own
# precision. Fails if any run fails.
#
#   ./tests/run_all.sh [pytest arguments]
#
# The builds are build_cpu_dp/, build_cpu_sp/, build_gpu_dp/ and build_gpu_sp/; one
# that does not exist is skipped, so a machine without a GPU runs the two CPU builds.
# Override the list with RTE3D_BUILDS, space-separated directories, each ending in _dp
# or _sp so its reference can be chosen. The references are the ones
# tests/build_reference.sh writes, build_ref_dp/ and build_ref_sp/; a missing one makes
# its comparisons skip, as they do for a bare pytest run.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
read -ra builds <<< "${RTE3D_BUILDS:-$root/build_cpu_dp $root/build_cpu_sp $root/build_gpu_dp $root/build_gpu_sp}"

ext=so
[[ "$(uname)" == "Darwin" ]] && ext=dylib

run() {
    local build="$1" ref="$2"
    echo "=== $build, reference $ref"

    local env=(RTE3D_PYTHON_PATH="$build/main_python")
    [[ -f "$ref/librte_kernels.$ext" ]] && env+=(RTE3D_FORTRAN_REF="$ref/librte_kernels.$ext")
    [[ -f "$ref/librte3d_shim.$ext" ]] && env+=(RTE3D_FORTRAN_SHIM="$ref/librte3d_shim.$ext")

    env -u RTE3D_FORTRAN_REF -u RTE3D_FORTRAN_SHIM "${env[@]}" \
        python -m pytest "$root/tests" "${@:3}"
}

status=0
failed=()
nrun=0
for build in "${builds[@]}"; do
    case "$build" in
        *_dp) ref="$root/build_ref_dp" ;;
        *_sp) ref="$root/build_ref_sp" ;;
        *)    echo "=== $build: name must end in _dp or _sp" >&2; exit 1 ;;
    esac
    if [[ ! -d "$build/main_python" ]]; then
        echo "=== $build: not built, skipped"
        continue
    fi
    nrun=$((nrun + 1))
    run "$build" "$ref" "$@" || { status=1; failed+=("$(basename "$build")"); }
done

if [[ $nrun == 0 ]]; then
    echo "=== no build found"
    exit 1
fi
[[ $status == 0 ]] && echo "=== all $nrun builds passed" || echo "=== FAILED: ${failed[*]}"
exit $status
