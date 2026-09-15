#!/usr/bin/env python
"""Draw what the third dimension is worth on the LES cloud field.

    python cases/run_case.py cases/les_cloudfield/les_cloudfield
    python cases/les_cloudfield/plot_solvers.py [-o figure.png]

Reads the case's output file and compares the two transports the case runs, in both
bands: the plane-parallel two-stream solver, which sees each column on its own, and
the Monte Carlo ray tracer, which sees the whole domain. Both switches have to be on
in les_cloudfield.toml, which they are.

The heating rates come from the fluxes the way a model would take them: from the
two-stream's flux divergence over a layer, and from the tracer's absorbed flux per
unit height, which is what it reports. Only the 200 resolved cells are drawn -- the
tracer's cell 200 holds the whole atmosphere above the box, lumped, and has no
plane-parallel counterpart of the same depth.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import require_plotting  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, 'les_cloudfield')

# Specific heat of dry air at constant pressure [J/kg/K], to turn an absorbed flux
# into the heating rate a model would feel, and seconds in a day.
CP = 1005.0
PER_DAY = 86400.0


def rates(d, g, band, nz, dz):
    """Domain-mean heating rate [K/day] from each solver, over the resolved cells."""
    rho = np.asarray(g['rho'])[:nz].reshape(nz, -1).mean(axis=1)

    def field(name):
        a = np.asarray(d[name], dtype=np.float64)
        return a.reshape(a.shape[0], -1)

    # The net flux is downward minus upward, so what a layer absorbs is what enters
    # its top less what leaves its bottom. The surface is index 0.
    net = field(f'{band}_flux_net')
    plane_parallel = (net[1:nz + 1] - net[:nz]).mean(axis=1)/(rho*CP*dz)*PER_DAY

    if band == 'sw':
        per_volume = field('rt_flux_abs_dir') + field('rt_flux_abs_dif')
    else:
        per_volume = field('rt_lw_flux_abs')
    tracer = per_volume[:nz].mean(axis=1)/(rho*CP)*PER_DAY

    return plane_parallel, tracer


def plot(d, g, path):
    require_plotting()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = np.asarray(g['z'])
    nz = z.size
    dz = float(np.diff(np.asarray(g['zh']))[0])

    shape = np.asarray(d['rt_flux_sfc_dir']).shape
    sfc_tracer = (np.asarray(d['rt_flux_sfc_dir'], dtype=np.float64)
                  + np.asarray(d['rt_flux_sfc_dif'], dtype=np.float64)).reshape(-1)
    sfc_plane_parallel = np.asarray(
        d['sw_flux_dn'], dtype=np.float64).reshape(-1, sfc_tracer.size)[0]

    fig = plt.figure(figsize=(12.5, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.35, wspace=0.3)

    # Where the cloud sits, since that is the only place the two transports can
    # differ at all.
    lwp = np.asarray(g['lwp'])[:nz].reshape(nz, -1).mean(axis=1)
    cloud = (z[lwp > 0].min()/1e3, z[lwp > 0].max()/1e3) if (lwp > 0).any() else None

    for col, (band, title, legend_at) in enumerate(
            (('sw', 'shortwave heating', 'upper right'),
             ('lw', 'longwave cooling', 'lower left'))):
        ax = fig.add_subplot(gs[:, col])
        plane_parallel, tracer = rates(d, g, band, nz, dz)

        ax.plot(plane_parallel, z/1e3, '-', color='C0', lw=1.6,
                label='two-stream (1D)')
        ax.plot(tracer, z/1e3, '-', color='C1', lw=1.6, label='ray tracer (3D)')

        # The topmost resolved cell is not converged in either code -- it absorbs an
        # order of magnitude harder than the median -- so it is drawn but is not
        # allowed to set the axis, which would flatten everything below it.
        span = np.concatenate([plane_parallel[:-1], tracer[:-1]])
        pad = 0.06*(span.max() - span.min())
        ax.set_xlim(span.min() - pad, span.max() + pad)

        if cloud:
            ax.axhspan(*cloud, color='0.85', lw=0, zorder=0)
            ax.text(0.97, cloud[1], 'cloud', transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', fontsize=8)

        ax.set_title(f'{title}, domain mean')
        ax.set_xlabel('heating rate [K day$^{-1}$]')
        if col == 0:
            ax.set_ylabel('height [km]')
        ax.legend(fontsize=8, loc=legend_at)
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    difference = (sfc_tracer - sfc_plane_parallel).reshape(shape)

    # A symmetric scale about zero, clipped at the 99th percentile so that a handful
    # of pixels in the deepest shadow do not set it for the whole map.
    limit = np.percentile(np.abs(difference), 99)
    extent = [0.0, float(np.asarray(g['xh'])[-1])/1e3,
              0.0, float(np.asarray(g['yh'])[-1])/1e3]
    image = ax.imshow(difference, origin='lower', cmap='RdBu_r',
                      vmin=-limit, vmax=limit, extent=extent)
    ax.set_title('surface shortwave down, 3D minus 1D')
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    fig.colorbar(image, ax=ax, pad=0.02, label='W m$^{-2}$')

    ax = fig.add_subplot(gs[1, 2])
    lo = min(sfc_plane_parallel.min(), sfc_tracer.min())
    hi = max(sfc_plane_parallel.max(), sfc_tracer.max())
    ax.hexbin(sfc_plane_parallel, sfc_tracer, gridsize=55, bins='log', cmap='Blues',
              extent=(lo, hi, lo, hi), linewidths=0)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)

    bias = sfc_tracer.mean() - sfc_plane_parallel.mean()
    rms = np.sqrt(np.mean((sfc_tracer - sfc_plane_parallel)**2))
    ax.text(0.03, 0.95, f'mean {bias:+.1f} W m$^{{-2}}$\nrms {rms:.1f} W m$^{{-2}}$',
            transform=ax.transAxes, va='top', fontsize=8)

    ax.set_title(f'surface shortwave down, {sfc_tracer.size} columns')
    ax.set_xlabel('two-stream [W m$^{-2}$]')
    ax.set_ylabel('ray tracer [W m$^{-2}$]')
    ax.grid(alpha=0.3)

    nx, ny = shape[1], shape[0]
    fig.suptitle(f'LES cumulus field: {nx} x {ny} x {nz} at {dz:g} m, '
                 'plane-parallel against ray traced')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    print(f'wrote {path}')

    print(f'surface shortwave down   1D {sfc_plane_parallel.mean():7.2f}   '
          f'3D {sfc_tracer.mean():7.2f} W/m2, rms difference {rms:.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-o', '--output', default=os.path.join(HERE, 'les_solvers.png'),
                   help='where to write the figure (default: %(default)s)')
    args = p.parse_args()

    import xarray as xr

    output = CASE + '_output.nc'
    if not os.path.exists(output):
        raise SystemExit(
            f'{output} is not there. Run the case first:\n'
            f'    python cases/run_case.py {CASE}')

    d = xr.open_dataset(output)
    missing = [name for name in ('sw_flux_net', 'lw_flux_net', 'rt_flux_abs_dir',
                                 'rt_lw_flux_abs') if name not in d]
    if missing:
        raise SystemExit(
            f'{output} has no {", ".join(missing)}: the figure compares the two '
            'transports, so both plane-parallel and raytracing have to be on in '
            'les_cloudfield.toml.')

    plot(d, xr.open_dataset(CASE + '_input.nc'), args.output)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
