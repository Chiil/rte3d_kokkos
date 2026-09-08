#!/usr/bin/env python
"""Run the RFMIP clear-sky case end to end and compare against the reference fluxes.

    python cases/rfmip/run_rfmip.py [--expt 0] [--plot] [-o out.nc]

Writes broadband up, down and (shortwave) direct fluxes for all 100 RFMIP sites, and
reports the largest difference from the reference. RFMIP's own acceptance threshold,
used by compare-to-reference.py in the reference repository, is 5.8e-2 W/m2.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA, Timer, require_plotting, write_fluxes  # noqa: E402

import rte3d  # noqa: E402
from rte3d.kdist import read_kdist  # noqa: E402
from rte3d.rfmip import (make_gas_concs, read_reference, read_rfmip,  # noqa: E402
                         solve_lw, solve_sw)

RFMIP = os.path.join(DATA, 'examples', 'rfmip-clear-sky')
INPUT = os.path.join(
    RFMIP, 'inputs', 'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc')
THRESHOLD = 5.8e-2


def reference_path(name):
    return os.path.join(RFMIP, 'reference',
                        f'{name}_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--expt', type=int, default=0,
                   help='RFMIP experiment index, 0-17 (default: 0, present day)')
    p.add_argument('--plot', action='store_true', help='also write a figure')
    p.add_argument('-o', '--output', default='rfmip_fluxes.nc')
    args = p.parse_args()

    results = {}
    for band, gas_file in (('lw', 'rrtmgp-gas-lw-g256.nc'), ('sw', 'rrtmgp-gas-sw-g224.nc')):
        f = read_kdist(os.path.join(DATA, gas_file))
        atm = read_rfmip(INPUT, f['gas_names'], expt=args.expt)
        gas_concs = make_gas_concs(rte3d, atm['gases'])
        kdist = rte3d.load_kdist(f, gas_concs)

        timer = Timer(f'{band} solve')
        if band == 'lw':
            up, dn = timer.run(lambda: solve_lw(rte3d, kdist, gas_concs, atm))
            results.update(rlu=up, rld=dn)
        else:
            up, dn, _ = timer.run(lambda: solve_sw(rte3d, kdist, gas_concs, atm))
            results.update(rsu=up, rsd=dn)

        ncol = atm['play'].shape[1]
        print(timer.report(ncol=ncol))
        results['plev'] = atm['plev']

    print(f'\n{ncol} columns, {atm["play"].shape[0]} layers, experiment {args.expt}')
    print(f'{"":6s} {"max |difference|":>18s}   {"threshold":>10s}')

    worst = 0.0
    for name in ('rlu', 'rld', 'rsu', 'rsd'):
        reference = read_reference(reference_path(name), name, expt=args.expt)
        err = np.abs(results[name] - reference).max()
        worst = max(worst, err)
        print(f'{name:6s} {err:18.3e}   {THRESHOLD:10.1e}  {"ok" if err < THRESHOLD else "FAIL"}')

    plev = results.pop('plev')
    write_fluxes(args.output, results, plev,
                 {'case': 'RFMIP clear-sky', 'experiment': args.expt})
    print(f'\nwrote {args.output}')

    if args.plot:
        plot(results, plev, args.output.replace('.nc', '.png'))

    return 0 if worst < THRESHOLD else 1


def plot(results, plev, path):
    require_plotting()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = plev.mean(axis=1)/100.0   # hPa, site mean

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    for ax, (band, pair) in zip(axes, (('longwave', ('rlu', 'rld')),
                                       ('shortwave', ('rsu', 'rsd')))):
        for name, style in zip(pair, ('-', '--')):
            ax.plot(results[name].mean(axis=1), p, style, label=name)
        ax.set_title(band)
        ax.set_xlabel('flux [W m$^{-2}$]')
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('pressure [hPa]')
    axes[0].invert_yaxis()
    axes[0].set_yscale('log')

    fig.suptitle('RFMIP clear-sky, mean over 100 sites')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f'wrote {path}')


if __name__ == '__main__':
    raise SystemExit(main())
