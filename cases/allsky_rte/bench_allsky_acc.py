#!/usr/bin/env python
"""Time rte3d against the Fortran reference built with OpenACC, on the all-sky case.

    python cases/allsky_rte/bench_allsky_acc.py --acc-build build_acc_sp [--ncol 1024,4096]

The reference is rte-rrtmgp's own all-sky example, built by
tests/build_reference_acc.sh. For each band and column count it generates the
atmosphere, times its loop and writes inputs and fluxes to NetCDF; rte3d then reads
that file and solves the same atmosphere, so the fluxes are compared as well. Use a
reference of the same precision as the rte3d build on RTE3D_PYTHON_PATH.

Both sides time the same span: cloud optics, gas optics and the solver, with inputs
and fluxes on the device. The example prints its times only to the millisecond, so
its time per loop is taken as (T(1 + n loops) - T(1 loop))/n, which also leaves out
the first loop's allocations, as rte3d's warm-up does.
"""
import argparse
import os
import resource
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA  # noqa: E402

import rte3d  # noqa: E402
from rte3d.kdist import read_cloud_optics, read_kdist  # noqa: E402
from rte3d.allsky import (ICERGH, make_gas_concs, read_allsky,  # noqa: E402
                          solve_lw, solve_sw)

BANDS = {
    'lw': ('rrtmgp-gas-lw-g256.nc', 'rrtmgp-clouds-lw-bnd.nc', solve_lw,
           ('lw_flux_up', 'lw_flux_dn')),
    'sw': ('rrtmgp-gas-sw-g224.nc', 'rrtmgp-clouds-sw-bnd.nc', solve_sw,
           ('sw_flux_up', 'sw_flux_dn', 'sw_flux_dir')),
}


class Recorder:
    """The rte3d module, with the solve_time of every solve_lw and solve_sw kept.

    rte3d.allsky's drivers return the fluxes only; this lets them be used unchanged.
    """

    def __init__(self, module):
        self._module = module
        self.solve_times = []

    def __getattr__(self, name):
        attr = getattr(self._module, name)
        if name not in ('solve_lw', 'solve_sw'):
            return attr

        def timed(*args, **kwargs):
            out = attr(*args, **kwargs)
            self.solve_times.append(out['solve_time'])
            return out

        return timed


def unlimited_stack():
    """The example is built with -Mstack_arrays, as ICON builds it, and needs the room."""
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))


def run_reference(exe, ncol, nlay, nloops, out, gas_file, cloud_file):
    """Total time of the example's loop over nloops, in seconds; None if it failed."""
    p = subprocess.run([exe, str(ncol), str(nlay), str(nloops), out, gas_file, cloud_file],
                       capture_output=True, text=True, preexec_fn=unlimited_stack)
    if p.returncode != 0:
        print(f'    reference failed at ncol={ncol}: '
              f'{(p.stdout + p.stderr).strip().splitlines()[-1:]}')
        return None

    # The line after the header: ncol nlay ngpt clouds aerosols per_col nloops total min.
    lines = p.stdout.strip().splitlines()
    header = next(i for i, line in enumerate(lines) if 'time_total_s' in line)
    return float(lines[header + 1].split()[7])


def read_reference(path):
    """read_allsky, for the file the example writes: its ice size is named dei."""
    import xarray as xr
    with xr.open_dataset(path) as d:
        if 'rei' in d or 'dei' not in d:
            return read_allsky(path)
        renamed = path.replace('.nc', '_rei.nc')
        d.rename({'dei': 'rei'}).to_netcdf(renamed)
    return read_allsky(renamed)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--acc-build', required=True,
                   help='build directory of tests/build_reference_acc.sh')
    p.add_argument('--ncol', default='1024,2048,4096,8192,16384,32768',
                   help='column counts, comma-separated (default: %(default)s)')
    p.add_argument('--nlay', type=int, default=72)
    p.add_argument('--nloops', type=int, default=20,
                   help='timed loops on each side (default: %(default)s)')
    p.add_argument('--bands', default='lw,sw')
    p.add_argument('--workdir', default=None,
                   help='where the example writes its NetCDF (default: a temporary dir)')
    args = p.parse_args()

    exe = os.path.join(os.path.abspath(args.acc_build), 'examples', 'all-sky', 'rrtmgp_allsky')
    workdir = args.workdir or tempfile.mkdtemp(prefix='bench_allsky_acc_')
    os.makedirs(workdir, exist_ok=True)
    columns = [int(n) for n in args.ncol.split(',')]

    print(f'reference: {exe}')
    print(f'rte3d:     {os.environ.get("RTE3D_PYTHON_PATH", "?")}')
    print(f'nlay={args.nlay}, mean of {args.nloops} loops each side\n')

    rows = []
    for band in args.bands.split(','):
        gas_name, cloud_name, solve, flux_names = BANDS[band]
        gas_file = os.path.join(DATA, gas_name)
        cloud_file = os.path.join(DATA, cloud_name)
        kdist_data = read_kdist(gas_file)
        cloud_optics = rte3d.load_cloud_optics(read_cloud_optics(cloud_file), ICERGH)

        for ncol in columns:
            out = os.path.join(workdir, f'{band}_{ncol}.nc')
            t1 = run_reference(exe, ncol, args.nlay, 1, out, gas_file, cloud_file)
            tn = t1 and run_reference(exe, ncol, args.nlay, 1 + args.nloops,
                                      out + '.n', gas_file, cloud_file)
            if tn is None:
                rows.append((band, ncol, None, None, None))
                continue
            t_ref = (tn - t1)/args.nloops

            atm = read_reference(out)
            gas_concs = make_gas_concs(rte3d, atm)
            kdist = rte3d.load_kdist(kdist_data, gas_concs)

            recorder = Recorder(rte3d)
            fluxes = solve(recorder, kdist, cloud_optics, gas_concs, atm)  # warm-up
            recorder.solve_times.clear()
            for _ in range(args.nloops):
                solve(recorder, kdist, cloud_optics, gas_concs, atm)
            t_rte3d = float(np.mean(recorder.solve_times))

            diff = max(np.abs(f - atm[name]).max() for f, name in zip(fluxes, flux_names))
            rows.append((band, ncol, t_ref, t_rte3d, diff))
            print(f'  {band} {ncol:6d}: reference {t_ref*1e3:8.2f} ms, '
                  f'rte3d {t_rte3d*1e3:8.2f} ms, max |diff| {diff:.2e} W/m2', flush=True)

            for f in (out, out + '.n', out.replace('.nc', '_rei.nc')):
                if os.path.exists(f):
                    os.remove(f)

    print(f'\n| band | ncol | Fortran OpenACC | rte3d | speed-up | max abs diff (W/m2) |')
    print(f'|---|---|---|---|---|---|')
    for band, ncol, t_ref, t_rte3d, diff in rows:
        if t_ref is None:
            print(f'| {band} | {ncol} | failed | | | |')
        else:
            print(f'| {band} | {ncol} | {t_ref*1e3:.2f} ms | {t_rte3d*1e3:.2f} ms | '
                  f'{t_ref/t_rte3d:.2f}x | {diff:.1e} |')


if __name__ == '__main__':
    raise SystemExit(main())
