#!/usr/bin/env bash
#
# Build the deprecated Fortran reference as a shared library, for use as a test
# oracle only. rte3d itself never links this and needs no Fortran compiler.
#
# Two libraries are built:
#
#   librte_kernels  - the RTE and RRTMGP kernels, which depend on nothing but
#                     mo_rte_kind and mo_rte_util_array. These carry the bind(C) API
#                     the kernel tests call directly.
#   librte3d_shim   - the whole Fortran frontend plus tests/shim/rte3d_shim.F90, which
#                     exposes the k-distribution reduction inside
#                     ty_gas_optics_rrtmgp%load. That reduction has no bind(C) entry
#                     point of its own, so without this shim it has no oracle.
#
# Neither needs NetCDF: Python reads the coefficient file and passes arrays in.
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
shim="$(cd "$(dirname "$0")/shim" && pwd)"

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

# The reduced arrays are PRIVATE components of ty_gas_optics_rrtmgp, so the shim cannot
# read them from outside the module. Compile a copy with the type-body `private`
# relaxed -- the second bare `private` in the file; the first is the module's own and
# must stay. The reference source itself is never modified.
awk 'BEGIN{n=0}
     /^[[:space:]]*private[[:space:]]*$/{
       n++
       if (n == 2) { print "    ! private relaxed by tests/build_reference.sh"; next }
     }
     {print}' \
    "$src/rrtmgp-frontend/mo_gas_optics_rrtmgp.F90" > "$out/mo_gas_optics_rrtmgp_open.F90"

( cd "$out" && "$FC" -fPIC -O2 -shared -o "librte3d_shim.$ext" \
    "$src/rte-frontend/mo_rte_kind.F90" \
    "$src/rte-frontend/mo_rte_config.F90" \
    "$src/rte-kernels/mo_rte_util_array.F90" \
    "$src/rte-frontend/mo_rte_util_array_validation.F90" \
    "$src/rte-kernels/mo_optical_props_kernels.F90" \
    "$src/rte-kernels/mo_rte_solver_kernels.F90" \
    "$src/rte-kernels/mo_fluxes_broadband_kernels.F90" \
    "$src/rte-frontend/mo_optical_props.F90" \
    "$src/rte-frontend/mo_source_functions.F90" \
    "$src/rte-frontend/mo_fluxes.F90" \
    "$src/gas-optics/mo_gas_optics_constants.F90" \
    "$src/gas-optics/mo_gas_optics_util_string.F90" \
    "$src/gas-optics/mo_gas_concentrations.F90" \
    "$src/gas-optics/mo_gas_optics.F90" \
    "$src/rrtmgp-kernels/mo_gas_optics_rrtmgp_kernels.F90" \
    "$out/mo_gas_optics_rrtmgp_open.F90" \
    "$shim/rte3d_shim.F90" )

echo "Built $out/librte_kernels.$ext"
echo "Built $out/librte3d_shim.$ext"
echo "export RTE3D_FORTRAN_REF=$out/librte_kernels.$ext"
echo "export RTE3D_FORTRAN_SHIM=$out/librte3d_shim.$ext"
