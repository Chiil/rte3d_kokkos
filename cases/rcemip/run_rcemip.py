#!/usr/bin/env python
"""RCEMIP: a large-problem benchmark, optionally against the Fortran reference.

    python cases/rcemip/run_rcemip.py [--ncol 4096] [--band lw|sw|both]
    python cases/rcemip/run_rcemip.py --compare-fortran

The RCEMIP input from rte-rrtmgp-cpp is a single radiative-convective-equilibrium
profile of 256 layers replicated over a 64x64 domain, so --ncol tiles that profile to
whatever column count you want.

What is timed
-------------
Both sides run the same sequence and the same amount of work:

    column gas amounts -> interpolation -> absorption optical depth
      -> Planck sources (longwave) or Rayleigh (shortwave) -> transport

The column gas amounts come from rte3d in both cases -- get_col_dry has no bind(C)
entry point -- so that step is a small constant added to each and does not bias the
comparison. Everything else is rte3d's Kokkos kernels on one side and the reference's
Fortran kernels, called through ctypes, on the other. Four calls per solve, so the
ctypes overhead is negligible at these sizes.

The Fortran comparison needs RTE3D_FORTRAN_REF, as the test suite does; build it with
tests/build_reference.sh.
"""
import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA, Timer  # noqa: E402

import rte3d  # noqa: E402
from rte3d.kdist import read_kdist  # noqa: E402

DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'rte-rrtmgp-cpp', 'rcemip', 'rcemip_input.nc')

# Well-mixed gases, read from the input file's scalars.
SCALAR_GASES = ('co2', 'ch4', 'n2o', 'n2', 'o2')


def read_rcemip(path, ncol):
    """Read the profile and tile it to ncol columns.

    Arrays come back as (layer, column) with layer 0 at the surface, which is how the
    file stores them.
    """
    d = xr.open_dataset(path)

    def tile(name):
        # (lay, y, x) -> (lay, ncol_file) -> (lay, ncol)
        v = d[name].values.reshape(d.sizes[d[name].dims[0]], -1).astype(np.float64)
        reps = -(-ncol // v.shape[1])
        return np.ascontiguousarray(np.tile(v, (1, reps))[:, :ncol])

    atm = dict(
        play=tile('p_lay'), plev=tile('p_lev'),
        tlay=tile('t_lay'), tlev=tile('t_lev'),
        h2o=tile('vmr_h2o'), o3=tile('vmr_o3'),
    )
    atm['tsfc'] = np.full(ncol, float(d['t_sfc'].values.flat[0]))
    atm['mu0'] = float(d['mu0'].values.flat[0])
    atm['tsi'] = float(d['tsi'].values.flat[0])
    atm['sfc_emis'] = float(d['emis_sfc'].values.flat[0])
    atm['sfc_alb'] = float(d['sfc_alb_dir'].values.flat[0])
    atm['scalars'] = {g: float(d[f'vmr_{g}']) for g in SCALAR_GASES}

    return atm


def make_gas_concs(atm):
    g = rte3d.Gas_concs()
    g.set_vmr('h2o', atm['h2o'])
    g.set_vmr('o3', atm['o3'])
    for name, value in atm['scalars'].items():
        g.set_vmr(name, value)
    g.set_vmr('co', 0.0)

    return g


def _grid(kdist):
    a = kdist.arrays()
    return a, dict(
        press_ref_log=a['press_ref_log'], temp_ref=a['temp_ref'],
        press_ref_log_delta=a['press_ref_log_delta'], temp_ref_min=a['temp_ref_min'],
        temp_ref_delta=a['temp_ref_delta'], press_ref_trop_log=a['press_ref_trop_log'],
        neta=kdist.neta, vmr_ref=a['vmr_ref'])


def rte3d_lw(kdist, gas_concs, atm, secants, weights):
    out = rte3d.gas_optics_lw(kdist, gas_concs, atm['play'], atm['plev'],
                              atm['tlay'], atm['tlev'], atm['tsfc'])
    ngpt, _, ncol = out['tau'].shape

    return rte3d.lw_solver_noscat(
        False, secants=secants, weights=weights, tau=out['tau'],
        lay_source=out['lay_source'], lev_source=out['lev_source'],
        sfc_emis=np.full((ngpt, ncol), atm['sfc_emis']),
        sfc_source=out['sfc_source'], inc_flux=np.zeros((ngpt, ncol)))


def fortran_lw(ref, kdist, gas_concs, atm, secants, weights):
    a, grid = _grid(kdist)
    col_gas = rte3d.compute_col_gas(gas_concs, kdist.gas_names, atm['plev'])

    interp = ref.interpolation(flavor=a['flavor'], **grid, play=atm['play'],
                               tlay=atm['tlay'], col_gas=col_gas)
    tau = ref.compute_tau_absorption(a, interp, atm['play'], atm['tlay'], col_gas, kdist.neta)
    src = ref.compute_planck_source(
        {**a, 'temp_ref_min': a['temp_ref_min'], 'totplnk_delta': a['totplnk_delta']},
        interp, atm['tlay'], atm['tlev'], atm['tsfc'], 0, kdist.neta)

    ngpt, _, ncol = tau.shape

    return ref.lw_solver_noscat(
        False, secants, weights, tau, src['lay_source'], src['lev_source'],
        np.full((ngpt, ncol), atm['sfc_emis']), src['sfc_source'],
        np.zeros((ngpt, ncol)))


def rte3d_sw(kdist, gas_concs, atm):
    tau, ssa = rte3d.gas_optics_sw(kdist, gas_concs, atm['play'], atm['plev'], atm['tlay'])
    ngpt, nlay, ncol = tau.shape

    toa = np.ascontiguousarray(np.broadcast_to(kdist.solar_source[:, None], (ngpt, ncol)))
    toa = toa*atm['tsi']/toa.sum(axis=0)[None, :]
    alb = np.full((ngpt, ncol), atm['sfc_alb'])

    return rte3d.sw_solver_2stream(
        False, tau, ssa, np.zeros_like(tau),
        mu0=np.full((nlay, ncol), atm['mu0']),
        sfc_alb_dir=alb, sfc_alb_dif=alb, inc_flux_dir=toa)


def fortran_sw(ref, kdist, gas_concs, atm):
    a, grid = _grid(kdist)
    col_gas = rte3d.compute_col_gas(gas_concs, kdist.gas_names, atm['plev'])

    interp = ref.interpolation(flavor=a['flavor'], **grid, play=atm['play'],
                               tlay=atm['tlay'], col_gas=col_gas)
    tau_abs = ref.compute_tau_absorption(a, interp, atm['play'], atm['tlay'], col_gas, kdist.neta)
    tau_ray = ref.compute_tau_rayleigh(a, interp, col_gas[0], col_gas, kdist.neta)

    tau = tau_abs + tau_ray
    ssa = np.where(tau > 0.0, tau_ray/np.maximum(tau, np.finfo(float).tiny), 0.0)

    ngpt, nlay, ncol = tau.shape
    toa = np.ascontiguousarray(np.broadcast_to(kdist.solar_source[:, None], (ngpt, ncol)))
    toa = toa*atm['tsi']/toa.sum(axis=0)[None, :]
    alb = np.full((ngpt, ncol), atm['sfc_alb'])

    return ref.sw_solver_2stream(
        False, tau, ssa, np.zeros_like(tau),
        np.full((nlay, ncol), atm['mu0']), alb, alb, toa)


def _stages(band, kdist, gas_concs, atm, ngpt, ncol):
    """Gas optics and transport separately, to show where the time goes."""
    nlay = atm['play'].shape[0]

    if band == 'lw':
        out = rte3d.gas_optics_lw(kdist, gas_concs, atm['play'], atm['plev'],
                                  atm['tlay'], atm['tlev'], atm['tsfc'])
        secants = np.full((1, ngpt, ncol), 1.0/0.6096748751)

        return [
            ('gas optics', lambda: rte3d.gas_optics_lw(
                kdist, gas_concs, atm['play'], atm['plev'],
                atm['tlay'], atm['tlev'], atm['tsfc'])),
            ('transport', lambda: rte3d.lw_solver_noscat(
                False, secants=secants, weights=np.array([1.0]), tau=out['tau'],
                lay_source=out['lay_source'], lev_source=out['lev_source'],
                sfc_emis=np.full((ngpt, ncol), atm['sfc_emis']),
                sfc_source=out['sfc_source'], inc_flux=np.zeros((ngpt, ncol)))),
        ]

    tau, ssa = rte3d.gas_optics_sw(kdist, gas_concs, atm['play'], atm['plev'], atm['tlay'])
    toa = np.ascontiguousarray(np.broadcast_to(kdist.solar_source[:, None], (ngpt, ncol)))
    toa = toa*atm['tsi']/toa.sum(axis=0)[None, :]
    alb = np.full((ngpt, ncol), atm['sfc_alb'])

    return [
        ('gas optics', lambda: rte3d.gas_optics_sw(
            kdist, gas_concs, atm['play'], atm['plev'], atm['tlay'])),
        ('transport', lambda: rte3d.sw_solver_2stream(
            False, tau, ssa, np.zeros_like(tau),
            mu0=np.full((nlay, ncol), atm['mu0']),
            sfc_alb_dir=alb, sfc_alb_dif=alb, inc_flux_dir=toa)),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ncol', type=int, default=4096,
                   help='columns to solve; the single profile is tiled to this many '
                        '(default: %(default)s, the full 64x64 domain). Memory grows '
                        'linearly: the longwave needs roughly 3.7 GB at 1024 columns')
    p.add_argument('--band', choices=('lw', 'sw', 'both'), default='both',
                   help='which band to run (default: %(default)s)')
    p.add_argument('--input', default=DEFAULT_INPUT,
                   help='path to rcemip_input.nc (default: the copy in rte-rrtmgp-cpp)')
    p.add_argument('--repeats', type=int, default=3,
                   help='timed repetitions after one warm-up; the best is reported '
                        '(default: %(default)s)')
    p.add_argument('--compare-fortran', action='store_true',
                   help='also time the reference Fortran kernels through ctypes and '
                        'report the ratio. Needs RTE3D_FORTRAN_REF')
    p.add_argument('--breakdown', action='store_true',
                   help='additionally time gas optics and transport separately')
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f'RCEMIP input not found: {args.input}\n'
                         'Pass --input to point at rcemip_input.nc.')

    ref = None
    if args.compare_fortran:
        path = os.environ.get('RTE3D_FORTRAN_REF')
        if path is None:
            raise SystemExit('--compare-fortran needs RTE3D_FORTRAN_REF; '
                             'build it with tests/build_reference.sh')
        import ctypes
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'tests'))
        from reference import Reference
        ref = Reference(ctypes.CDLL(path))

    atm = read_rcemip(args.input, args.ncol)
    gas_concs = make_gas_concs(atm)
    nlay, ncol = atm['play'].shape

    runtime = rte3d.runtime()
    print(f'RCEMIP  {ncol} columns x {nlay} layers   '
          f'[{runtime.backend}, {runtime.precision} precision, '
          f'{runtime.concurrency} threads]')

    if args.compare_fortran and runtime.concurrency > 1:
        print('\nNote: the reference kernels are serial on the host -- their OpenMP and\n'
              '      OpenACC directives target accelerators, not host threads. Set\n'
              '      OMP_NUM_THREADS=1 for a per-thread comparison.')
    print()

    bands = ('lw', 'sw') if args.band == 'both' else (args.band,)
    for band in bands:
        gas_file = 'rrtmgp-gas-lw-g256.nc' if band == 'lw' else 'rrtmgp-gas-sw-g224.nc'
        kdist = rte3d.load_kdist(read_kdist(os.path.join(DATA, gas_file)), gas_concs)
        ngpt = kdist.arrays()['kmajor'].shape[0]

        if band == 'lw':
            secants = np.full((1, ngpt, ncol), 1.0/0.6096748751)
            weights = np.array([1.0])
            mine = lambda: rte3d_lw(kdist, gas_concs, atm, secants, weights)
            theirs = lambda: fortran_lw(ref, kdist, gas_concs, atm, secants, weights)
        else:
            mine = lambda: rte3d_sw(kdist, gas_concs, atm)
            theirs = lambda: fortran_sw(ref, kdist, gas_concs, atm)

        print(f'--- {band} ({ngpt} g-points)')
        t = Timer('rte3d')
        result = t.run(mine, repeats=args.repeats)
        print('   ' + t.report(ncol=ncol))

        if args.breakdown:
            for label, stage in _stages(band, kdist, gas_concs, atm, ngpt, ncol):
                ts = Timer('  ' + label)
                ts.run(stage, repeats=args.repeats)
                print('   ' + ts.report(ncol=ncol))

        if ref is not None:
            tf = Timer('fortran reference')
            expected = tf.run(theirs, repeats=args.repeats)
            print('   ' + tf.report(ncol=ncol))
            print(f'   {"speedup":28s} {t.best and tf.best/t.best:9.2f} x')

            err = max(np.abs(a.sum(axis=0) - b.sum(axis=0)).max()
                      for a, b in zip(result[:2], expected[:2]))
            print(f'   {"max flux difference":28s} {err:9.2e} W/m2')
        print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
