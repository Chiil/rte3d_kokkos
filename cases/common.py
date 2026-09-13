"""Shared helpers for the case scripts.

Every case script writes its fluxes to NetCDF and, given --plot, a figure. Plotting is
optional: matplotlib is not a dependency of rte3d and the scripts run without it.
"""
import os
import sys
import time

import numpy as np

# This file lives in cases/, so rte3d is one level up. Every path here is built from
# this file and not from the caller's, so a script in cases/<name>/ reaches it too.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'main_python'))

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'extern', 'rrtmgp-data')


def have_matplotlib():
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def require_plotting():
    if not have_matplotlib():
        raise SystemExit(
            'Plotting needs matplotlib, which is not installed.\n'
            '    pip install matplotlib\n'
            'The fluxes are written to NetCDF either way.')


def write_fluxes(path, fluxes, plev, extra_attrs=None):
    """Write broadband fluxes as (level, column), the layout rte3d computes in."""
    import xarray as xr

    ds = xr.Dataset(
        {name: (('lev', 'col'), value) for name, value in fluxes.items()},
        coords={'p_lev': (('lev', 'col'), plev)},
    )
    ds.attrs.update(extra_attrs or {})
    ds.to_netcdf(path)

    return path


class Timer:
    """Wall-clock timing with a warm-up, since the first call pays for allocation."""

    def __init__(self, label):
        self.label = label
        self.times = []

    def run(self, fn, repeats=3, warmup=1):
        for _ in range(warmup):
            result = fn()
        for _ in range(repeats):
            start = time.perf_counter()
            result = fn()
            self.times.append(time.perf_counter() - start)

        return result

    @property
    def best(self):
        return min(self.times) if self.times else float('nan')

    def report(self, ncol=None, nlay=None):
        line = f'{self.label:28s} {self.best*1e3:9.1f} ms'
        if ncol is not None:
            per_col = self.best/ncol*1e6
            line += f'   {per_col:8.2f} us/column'
        return line
