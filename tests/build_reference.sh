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
# The reference is rte-rrtmgp v1.9. Its source layout is what the file lists below
# assume, and its conventions are rte3d's: the longwave quadrature weights scale by pi
# rather than 2 pi (from v1.8), and logicals are C bool, one byte. An older checkout
# builds, but its no-scattering longwave fluxes come out twice rte3d's.
#
# Build one per precision, to test the matching rte3d build against:
#
#   ./tests/build_reference.sh [dp|sp] [rte-rrtmgp-dir] [output-dir]
#   export RTE3D_FORTRAN_REF=<output-dir>/librte_kernels.<ext>
#
# dp, the default, goes to build/reference; sp to build/reference_sp.
set -euo pipefail

precision="${1:-dp}"
case "$precision" in
    dp) fflags=();              default_out=reference ;;
    sp) fflags=(-DRTE_USE_SP);  default_out=reference_sp ;;
    *)  echo "precision must be dp or sp, not '$precision'" >&2; exit 1 ;;
esac

src="${2:-$(dirname "$0")/../rte-rrtmgp}"
out="${3:-$(dirname "$0")/../build/$default_out}"
FC="${FC:-gfortran}"

# -march=native as rte3d's own configs build: it lets the compiler contract a*b + c into
# a fused multiply-add, as rte3d's does. Without it the two round differently, which in
# single precision is the whole of the difference in the shortwave two-stream fluxes;
# with it they come out bit-identical. Override with FFLAGS for another configuration.
read -ra fflags_opt <<< "${FFLAGS:--O2 -march=native}"
fflags+=("${fflags_opt[@]}")

ext=so
[[ "$(uname)" == "Darwin" ]] && ext=dylib

mkdir -p "$out"

# Resolve both to absolute paths before compiling: the compile runs with the output
# directory as its working directory, so that .mod files land there.
src="$(cd "$src" && pwd)"
out="$(cd "$out" && pwd)"
shim="$(cd "$(dirname "$0")/shim" && pwd)"

# Compiled in dependency order so the .mod files exist when they are needed.
( cd "$out" && "$FC" -fPIC -shared "${fflags[@]}" -o "librte_kernels.$ext" \
    "$src/rte-kernels/mo_rte_kind.F90" \
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

( cd "$out" && "$FC" -fPIC -shared "${fflags[@]}" -o "librte3d_shim.$ext" \
    "$src/rte-kernels/mo_rte_kind.F90" \
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
    "$src/rrtmgp-kernels/mo_cloud_optics_rrtmgp_kernels.F90" \
    "$src/rrtmgp-frontend/mo_cloud_optics_rrtmgp.F90" \
    "$shim/rte3d_shim.F90" )

echo "Built $out/librte_kernels.$ext"
echo "Built $out/librte3d_shim.$ext"
echo "export RTE3D_FORTRAN_REF=$out/librte_kernels.$ext"
echo "export RTE3D_FORTRAN_SHIM=$out/librte3d_shim.$ext"
