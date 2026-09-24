import os
import sys
from pathlib import Path

import pytest

# Kokkos recommends this for unit testing and it silences its startup warning.
os.environ.setdefault('OMP_PROC_BIND', 'false')

_root = Path(__file__).resolve().parents[1]

# The rte3d package lives in main_python/, next to the compiled module.
sys.path.insert(0, str(_root / 'main_python'))


@pytest.fixture(scope='session')
def rte3d():
    """The compiled extension module.

    Requires RTE3D_PYTHON_PATH (or RTE3D_LIB_DIR) to point at a build directory.
    """
    import rte3d as _rte3d
    return _rte3d


def _check_precision(oracle, rte3d, variable):
    """The oracle must be built in the precision rte3d was, or the comparison is
    measuring round-off rather than the port. A mismatch is a setup error, so it fails
    rather than skips."""
    precision = rte3d.runtime().precision
    if oracle.precision != precision:
        pytest.fail(
            f'{variable} is a {oracle.precision}-precision build but rte3d is {precision}; '
            f'build the matching one with tests/build_reference.sh '
            f'{"sp" if precision == "single" else "dp"}.')
    return oracle


@pytest.fixture(scope='session')
def fortran_ref(rte3d):
    """The deprecated Fortran reference, loaded as a correctness oracle only.

    rte3d itself never links this: it is not a build dependency, needs no Fortran
    compiler, and every kernel it contains is reimplemented in Kokkos. Tests that use
    this fixture skip when RTE3D_FORTRAN_REF is unset, so the suite stays runnable
    once the reference is dropped.

    Build it with tests/build_reference.sh, in the precision rte3d was built in, then
    point RTE3D_FORTRAN_REF at the resulting .so/.dylib. The C prototypes are documented
    in rte-rrtmgp/rte-kernels/api/rte_kernels.h.
    """
    path = os.environ.get('RTE3D_FORTRAN_REF')
    if path is None:
        pytest.skip('RTE3D_FORTRAN_REF not set; skipping comparison against the Fortran reference.')

    import ctypes

    from reference import Reference
    return _check_precision(Reference(ctypes.CDLL(path)), rte3d, 'RTE3D_FORTRAN_REF')


@pytest.fixture(scope='session')
def fortran_shim(rte3d):
    """The reduction oracle. See tests/shim/rte3d_shim.F90.

    Built by tests/build_reference.sh alongside the kernel library; skips when
    RTE3D_FORTRAN_SHIM is unset.
    """
    path = os.environ.get('RTE3D_FORTRAN_SHIM')
    if path is None:
        pytest.skip('RTE3D_FORTRAN_SHIM not set; skipping the k-distribution reduction oracle.')

    import ctypes

    from shim import Shim
    return _check_precision(Shim(ctypes.CDLL(path)), rte3d, 'RTE3D_FORTRAN_SHIM')
