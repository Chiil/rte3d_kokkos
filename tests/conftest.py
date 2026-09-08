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


@pytest.fixture(scope='session')
def fortran_ref():
    """The deprecated Fortran reference, loaded as a correctness oracle only.

    rte3d itself never links this: it is not a build dependency, needs no Fortran
    compiler, and every kernel it contains is reimplemented in Kokkos. Tests that use
    this fixture skip when RTE3D_FORTRAN_REF is unset, so the suite stays runnable
    once the reference is dropped.

    Build it out of tree with -fPIC as a shared library from rte-rrtmgp/, then point
    RTE3D_FORTRAN_REF at the resulting .so/.dylib. The C prototypes are documented in
    rte-rrtmgp/rte-kernels/api/rte_kernels.h.
    """
    path = os.environ.get('RTE3D_FORTRAN_REF')
    if path is None:
        pytest.skip('RTE3D_FORTRAN_REF not set; skipping comparison against the Fortran reference.')

    import ctypes

    from reference import Reference
    return Reference(ctypes.CDLL(path))


@pytest.fixture(scope='session')
def fortran_shim():
    """The reduction oracle. See tests/shim/rte3d_shim.F90.

    Built by tests/build_reference.sh alongside the kernel library; skips when
    RTE3D_FORTRAN_SHIM is unset.
    """
    path = os.environ.get('RTE3D_FORTRAN_SHIM')
    if path is None:
        pytest.skip('RTE3D_FORTRAN_SHIM not set; skipping the k-distribution reduction oracle.')

    import ctypes

    from shim import Shim
    return Shim(ctypes.CDLL(path))
