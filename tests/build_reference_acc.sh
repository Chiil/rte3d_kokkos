#!/usr/bin/env bash
#
# Build the Fortran reference's all-sky example with OpenACC, using NVHPC, as the GPU
# baseline rte3d's performance is compared against. Not a test oracle -- that is
# tests/build_reference.sh -- and rte3d never links it.
#
#   ./tests/build_reference_acc.sh [dp|sp] [rte-rrtmgp-dir] [output-dir]
#
# dp, the default, goes to build_acc_dp/; sp to build_acc_sp/. The result is
# <output-dir>/examples/all-sky/rrtmgp_allsky, which cases/allsky_rte/bench_allsky_acc.py
# runs. The kernels are rte-rrtmgp's accel ones, the OpenACC versions.
#
# Only that one target is built: the reference's default target also fetches its data
# over the network, which rte3d has in extern/rrtmgp-data already.
#
# NetCDF-Fortran must be built with the same compiler; NETCDF_DIR points at it.
# Override the GPU architecture or flags with ACC_FLAGS. -Mstack_arrays puts the
# frontend's automatic arrays on the stack, so the example needs `ulimit -s unlimited`;
# the benchmark script raises the limit itself.
#
# The build is of a copy of the reference, in <output-dir>/src; the reference itself is
# never modified. Up to rte-rrtmgp #433 the copy was patched for a GPU bug in col_dry
# (the device copy of col_dry_arr was never written, so every flux below the boundary
# came out NaN); #433 fixed it upstream, so the copy is now built unchanged.
set -euo pipefail

precision="${1:-dp}"
case "$precision" in
    dp) sp=OFF; default_out=build_acc_dp ;;
    sp) sp=ON;  default_out=build_acc_sp ;;
    *)  echo "precision must be dp or sp, not '$precision'" >&2; exit 1 ;;
esac

src="${2:-$(dirname "$0")/../rte-rrtmgp}"
out="${3:-$(dirname "$0")/../$default_out}"
netcdf="${NETCDF_DIR:-$HOME/devel/netcdf-nvidia}"
# The flags of rte-rrtmgp's own OpenACC CI (.gitlab/dkrz.yml, which are ICON's), with the
# architecture of an RTX A4500.
flags="${ACC_FLAGS:--g -O2 -Mrecursive -Mallocatable=03 -Mstack_arrays -Minfo=accel -acc=gpu,verystrict -gpu=cc86}"

src="$(cd "$src" && pwd)"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

# The tracked files only, as they stand in the working tree: not the reference's own
# build products or data.
rm -rf "$out/src"
mkdir -p "$out/src"
git -C "$src" ls-files -z | rsync -a --from0 --files-from=- "$src/" "$out/src/"

cmake -S "$out/src" -B "$out" \
    -DCMAKE_Fortran_COMPILER=nvfortran \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_Fortran_FLAGS="$flags" \
    -DCMAKE_Fortran_FLAGS_RELEASE="" \
    -DCMAKE_PREFIX_PATH="$netcdf" \
    -DRTE_KERNEL_MODE=accel \
    -DRTE_ENABLE_SP="$sp" \
    -DRTE_BUILD_TESTING=ON

cmake --build "$out" --target rrtmgp_allsky -j

echo "Built $out/examples/all-sky/rrtmgp_allsky"
