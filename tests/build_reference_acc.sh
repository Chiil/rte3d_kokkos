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
# The build is of a patched copy of the reference, in <output-dir>/src; the reference
# itself is never modified. One patch, for a bug in the GPU path since rte-rrtmgp #419:
#
#   col_dry_arr = get_col_dry(ncol, nlay, vmr(:,:,idx_h2o), plev) * 1.e-4_wp
#
# in rrtmgp/frontend/mo_gas_optics_rrtmgp.F90. get_col_dry computes on the device and
# copies its result out; the scaling and the assignment then happen on the host, and
# the device copy of col_dry_arr, which every kernel after it reads, is never written.
# Every flux below the boundary comes out NaN on the GPU, right on the host. The patch
# adds the missing update device. It is applied only where the line is followed by no
# update already, so a reference with the fix upstream builds unchanged.
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

frontend="$out/src/rrtmgp/frontend/mo_gas_optics_rrtmgp.F90"
awk '{
       print
       if ($0 ~ /^[[:space:]]*col_dry_arr = get_col_dry\(.*\* *1\.e-4_wp/) {
         getline next_line
         if (next_line !~ /update/) {
           print "        !$acc        update device(col_dry_arr)  ! added by build_reference_acc.sh"
           print "        !$omp target update to(col_dry_arr)      ! added by build_reference_acc.sh"
         }
         print next_line
       }
     }' "$frontend" > "$frontend.patched"
mv "$frontend.patched" "$frontend"
grep -q "added by build_reference_acc.sh" "$frontend" \
    && echo "Patched $frontend: update device(col_dry_arr)" \
    || echo "No col_dry patch needed in $frontend"

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
