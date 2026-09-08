#!/usr/bin/env python
"""Run the all-sky cloudy case end to end and compare against the reference fluxes.

    python cases/allsky/run_allsky.py [--plot] [-o out.nc]

Unlike RFMIP this reference was generated with the coefficient data that ships beside
it, so agreement is to round-off rather than to RFMIP's 5.8e-2 W/m2 threshold.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA, Timer, require_plotting, write_fluxes  # noqa: E402

import rte3d  # noqa: E402
from rte3d.kdist import read_cloud_optics, read_kdist  # noqa: E402
from rte3d.allsky import (ICERGH, make_gas_concs, read_allsky,  # noqa: E402
                          solve_lw, solve_sw)

REFERENCE = os.path.join(DATA, 'examples', 'all-sky', 'reference')
TOLERANCE = 1e-9


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plot', action='store_true',
                   help='also write a figure alongside the NetCDF, named <output>.png')
    p.add_argument('-o', '--output', default='allsky_fluxes.nc',
                   help='where to write the fluxes (default: %(default)s)')
    args = p.parse_args()

    results = {}
    references = {}
    worst = 0.0

    for band, gas_file, cloud_file, ref_file in (
            ('lw', 'rrtmgp-gas-lw-g256.nc', 'rrtmgp-clouds-lw-bnd.nc',
             'rrtmgp-allsky-lw-no-aerosols.nc'),
            ('sw', 'rrtmgp-gas-sw-g224.nc', 'rrtmgp-clouds-sw-bnd.nc',
             'rrtmgp-allsky-sw-no-aerosols.nc')):

        atm = read_allsky(os.path.join(REFERENCE, ref_file))
        gas_concs = make_gas_concs(rte3d, atm)
        kdist = rte3d.load_kdist(read_kdist(os.path.join(DATA, gas_file)), gas_concs)
        cloud_optics = rte3d.load_cloud_optics(
            read_cloud_optics(os.path.join(DATA, cloud_file)), ICERGH)

        timer = Timer(f'{band} solve')
        if band == 'lw':
            up, dn = timer.run(lambda: solve_lw(rte3d, kdist, cloud_optics, gas_concs, atm))
            computed = {'lw_flux_up': up, 'lw_flux_dn': dn}
        else:
            up, dn, direct = timer.run(
                lambda: solve_sw(rte3d, kdist, cloud_optics, gas_concs, atm))
            computed = {'sw_flux_up': up, 'sw_flux_dn': dn, 'sw_flux_dir': direct}

        ncol = atm['play'].shape[1]
        print(timer.report(ncol=ncol))

        for name, value in computed.items():
            references[name] = atm[name]
            worst = max(worst, np.abs(value - references[name]).max())
        results.update(computed)
        results['plev'] = atm['plev']

    print(f'\n{ncol} columns, {atm["play"].shape[0]} layers, '
          f'{int((atm["lwp"].sum(axis=0) > 0).sum())} cloudy')
    print(f'max |difference| from reference: {worst:.3e} W/m2  '
          f'({"ok" if worst < TOLERANCE else "FAIL"})')

    plev = results.pop('plev')
    write_fluxes(args.output, results, plev, {'case': 'all-sky, no aerosols'})
    print(f'wrote {args.output}')

    if args.plot:
        plot(results, references, plev, atm, args.output.replace('.nc', '.png'))

    return 0 if worst < TOLERANCE else 1


def plot(results, references, plev, atm, path):
    require_plotting()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = plev.mean(axis=1)/100.0
    cloudy = atm['lwp'].sum(axis=0) > 0.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    # rte3d in colour, reference dotted in black on top: black-on-colour is what makes
    # the overlap readable. Only the cloudy columns are drawn, since plotting the clear
    # ones too doubles the curves for nothing.
    for ax, (band, names) in zip(axes, (('longwave', ('lw_flux_up', 'lw_flux_dn')),
                                        ('shortwave', ('sw_flux_up', 'sw_flux_dn')))):
        for name, colour in zip(names, ('C0', 'C1')):
            ax.plot(results[name][:, cloudy].mean(axis=1), p, '-', color=colour, lw=1.6,
                    label=f'{name} rte3d')
            ax.plot(references[name][:, cloudy].mean(axis=1), p, ':', color='k', lw=1.2,
                    zorder=5, label=f'{name} reference')
        ax.set_title(f'{band}, cloudy columns')
        ax.set_xlabel('flux [W m$^{-2}$]')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Cloud water paths, to show where the clouds sit.
    play = atm['play'].mean(axis=1)/100.0
    axes[2].plot(atm['lwp'][:, cloudy].mean(axis=1), play, label='liquid')
    axes[2].plot(atm['iwp'][:, cloudy].mean(axis=1), play, label='ice')
    axes[2].set_title('cloud water path')
    axes[2].set_xlabel('water path [g m$^{-2}$]')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[0].set_ylabel('pressure [hPa]')
    axes[0].invert_yaxis()
    axes[0].set_yscale('log')

    fig.suptitle('All-sky: rte3d (colour) vs reference (black dots)')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f'wrote {path}')


if __name__ == '__main__':
    raise SystemExit(main())
