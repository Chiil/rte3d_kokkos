#!/usr/bin/env bash
#
# Build the deprecated Fortran reference as a shared library, for use as a test
# oracle only. rte3d itself never links this and needs no Fortran compiler.
#
# The RTE and RRTMGP kernels depend on nothing but mo_rte_kind and mo_rte_util_array,
# so this is six files and no data.
#
#   ./tests/build_reference.sh [rte-rrtmgp-dir] [output-dir]
#   export RTE3D_FORTRAN_REF=<output-dir>/librte_kernels.<ext>
#
set -euo pipefail

src="${1:-$(dirname "$0")/../rte-rrtmgp}"
out="${2:-$(dirname "$0")/../build/reference}"
FC="${FC:-gfortran}"

ext=so
[[ "$(uname)" == "Darwin" ]] && ext=dylib

mkdir -p "$out"

# Resolve both to absolute paths before compiling: the compile runs with the output
# directory as its working directory, so that .mod files land there.
src="$(cd "$src" && pwd)"
out="$(cd "$out" && pwd)"

# Compiled in dependency order so the .mod files exist when they are needed.
# Defaults match rte3d: double precision, and the default 4-byte logical kind that
# rte_kernels.h describes as C int. Add -DRTE_USE_CBOOL here and set
# RTE3D_FORTRAN_REF_CBOOL for the tests if you want 1-byte logicals instead.
( cd "$out" && "$FC" -fPIC -O2 -shared -o "librte_kernels.$ext" \
    "$src/rte-frontend/mo_rte_kind.F90" \
    "$src/rte-kernels/mo_rte_util_array.F90" \
    "$src/rte-kernels/mo_rte_solver_kernels.F90" \
    "$src/rte-kernels/mo_optical_props_kernels.F90" \
    "$src/rte-kernels/mo_fluxes_broadband_kernels.F90" \
    "$src/rrtmgp-kernels/mo_gas_optics_rrtmgp_kernels.F90" )

echo "Built $out/librte_kernels.$ext"
echo "export RTE3D_FORTRAN_REF=$out/librte_kernels.$ext"
