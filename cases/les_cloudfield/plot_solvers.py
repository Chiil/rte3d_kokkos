#!/usr/bin/env python
"""Draw what the third dimension is worth on the LES cloud field.

    python cases/run_case.py cases/les_cloudfield/les_cloudfield
    python cases/les_cloudfield/plot_solvers.py [-o stem]

Reads the case's output file and compares the two transports the case runs, in both
bands: the plane-parallel two-stream solver, which sees each column on its own, and
the Monte Carlo ray tracer, which sees the whole domain. Both switches have to be on
in les_cloudfield.toml, which they are.

Three figures come out of it, named after the stem:

    <stem>_profiles.png   the heating and cooling profiles, domain mean
    <stem>_sfc_down.png   the downward flux at the surface, both bands, both solvers
                          and their difference
    <stem>_tod_up.png     the upward flux leaving the top of the tracer's domain,
                          laid out the same way

The heating rates come from the fluxes the way a model would take them: from the
two-stream's flux divergence over a layer, and from the tracer's absorbed flux per
unit height, which is what it reports.

The two bands need not trace the same box. Where a band lumps the atmosphere above the
box into the box's top cell, that cell stands for everything above it, so what leaves
the top of that box is what leaves the atmosphere and the two-stream's own top of
atmosphere is what it compares against; where the air above is left outside instead,
the box ends at its own top and so does the comparison. The figures say which.

A lumped cell also costs the cells just below it, so where the case does lump --
lump-above = true in the settings -- the profile figure hatches the region rather than
hide it. The lump carries all the air above the box at one temperature and the box's
own depth, which on this field sends 28 W/m2 less down into the box than the air it
stands in for, and the cells beneath cool too hard by an amount that dies away
downward. The case leaves that air outside the box, and then nothing is hatched.
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

BANDS = (('sw', 'shortwave'), ('lw', 'longwave'))


def field(d, name):
    """A field of the output as (level or cell, column), or (column) if it has no
    vertical dimension."""
    a = np.asarray(d[name], dtype=np.float64)
    return a.reshape(-1) if a.ndim == 2 else a.reshape(a.shape[0], -1)


def tracer_cells(d, band):
    """How many cells the tracer used for this band."""
    return d['rt_flux_abs_dir' if band == 'sw' else 'rt_lw_flux_abs'].shape[0]


def rates(d, g, band, nz, dz):
    """Domain-mean heating rate [K/day] from each solver, over the resolved cells."""
    rho = np.asarray(g['rho'])[:nz].reshape(nz, -1).mean(axis=1)

    # The net flux is downward minus upward, so what a layer absorbs is what enters
    # its top less what leaves its bottom. The surface is index 0.
    net = field(d, f'{band}_flux_net')
    plane_parallel = (net[1:nz + 1] - net[:nz]).mean(axis=1)/(rho*CP*dz)*PER_DAY

    if band == 'sw':
        per_volume = field(d, 'rt_flux_abs_dir') + field(d, 'rt_flux_abs_dif')
    else:
        per_volume = field(d, 'rt_lw_flux_abs')
    tracer = per_volume[:nz].mean(axis=1)/(rho*CP)*PER_DAY

    return plane_parallel, tracer


def surface_down(d, band, nz):
    """The downward flux at the surface, from each solver, and what it is called."""
    plane_parallel = field(d, f'{band}_flux_dn')[0]

    if band == 'sw':
        tracer = field(d, 'rt_flux_sfc_dir') + field(d, 'rt_flux_sfc_dif')
    else:
        tracer = field(d, 'rt_lw_flux_sfc_dn')

    return plane_parallel, tracer, 'at the surface'


def domain_top_up(d, band, nz):
    """The upward flux leaving the top of the tracer's domain, from each solver.

    Which level of the column that is depends on what the band did with the air above
    the box: lumped into the box's top cell, the box stands for the whole atmosphere
    and what leaves it leaves the atmosphere; left outside, the box ends where it ends.
    """
    up = field(d, f'{band}_flux_up')
    lumped = tracer_cells(d, band) > nz

    plane_parallel = up[-1] if lumped else up[nz]
    tracer = field(d, 'rt_flux_tod_up' if band == 'sw' else 'rt_lw_flux_tod_up')
    where = ('at the top of the atmosphere, the air above the box lumped into it'
             if lumped else 'at the top of the box')

    return plane_parallel, tracer, where


def lump_shadow(plane_parallel, tracer, z):
    """Where the lumped cell above the box starts contaminating the profile.

    The two transports differ everywhere by the tracer's Monte Carlo noise, and near
    the top of the box by the lump as well. The lump's part is the one that grows
    towards the boundary, so it is found as the topmost run of cells that stands out
    of the noise the rest of the profile shows. Returns the height it starts at, or
    None where nothing stands out.
    """
    difference = np.abs(tracer - plane_parallel)

    # The noise, taken from the lower three quarters, which the lump does not reach.
    noise = np.median(difference[:3*difference.size//4])

    out = difference > 3*noise
    if not out[-1]:
        return None

    k = difference.size - 1
    while k > 0 and out[k - 1]:
        k -= 1

    return z[k]


def plot_profiles(plt, d, g, z, dz, title, path):
    """The heating and cooling the two transports give, as a model would feel them."""
    nz = z.size

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

    # Where the cloud sits, since that is the only place the two transports can
    # differ at all.
    lwp = np.asarray(g['lwp'])[:nz].reshape(nz, -1).mean(axis=1)
    cloud = (z[lwp > 0].min()/1e3, z[lwp > 0].max()/1e3) if (lwp > 0).any() else None

    for ax, (band, name), legend_at in zip(
            axes, BANDS, ('upper right', 'lower left')):
        plane_parallel, tracer = rates(d, g, band, nz, dz)

        ax.plot(plane_parallel, z/1e3, '-', color='C0', lw=1.6,
                label='two-stream (1D)')
        ax.plot(tracer, z/1e3, '-', color='C1', lw=1.6, label='ray tracer (3D)')

        # The cells a lumped cell above the box spoils are drawn, but they are not
        # allowed to set the axis: the topmost one absorbs an order of magnitude
        # harder than anything else and would flatten the whole profile against it.
        shadow = lump_shadow(plane_parallel, tracer, z)
        clean = slice(None) if shadow is None else (z < shadow)

        span = np.concatenate([plane_parallel[clean], tracer[clean]])
        pad = 0.06*(span.max() - span.min())
        ax.set_xlim(span.min() - pad, span.max() + pad)

        if shadow is not None:
            ax.axhspan(shadow/1e3, z[-1]/1e3, facecolor='none', edgecolor='0.6',
                       hatch='//', lw=0, zorder=0)
            ax.text(0.03, shadow/1e3, 'lumped cell above', fontsize=8, va='bottom',
                    transform=ax.get_yaxis_transform())

        if cloud:
            ax.axhspan(*cloud, color='0.85', lw=0, zorder=0)
            ax.text(0.97, cloud[1], 'cloud', transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', fontsize=8)

        ax.set_title(f'{name}, domain mean')
        ax.set_xlabel('heating rate [K day$^{-1}$]')
        ax.legend(fontsize=8, loc=legend_at)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('height [km]')
    fig.suptitle(title)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    print(f'wrote {path}')


def plot_maps(plt, d, g, which, nz, shape, title, path):
    """One flux, both bands: each solver's map and the difference between them.

    A band gets a row of three: the two solvers on one shared scale, so that the maps
    can be read against each other rather than each against itself, and their
    difference on a scale of its own, symmetric about zero.
    """
    extent = [0.0, float(np.asarray(g['xh'])[-1])/1e3,
              0.0, float(np.asarray(g['yh'])[-1])/1e3]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), layout='constrained')
    wheres = []

    for row, (band, name) in zip(axes, BANDS):
        plane_parallel, tracer, where = which(d, band, nz)
        difference = tracer - plane_parallel
        wheres.append(f'{name}: {where}')

        low = min(plane_parallel.min(), tracer.min())
        high = max(plane_parallel.max(), tracer.max())

        for ax, values, label in zip(row[:2], (plane_parallel, tracer),
                                     ('two-stream (1D)', 'ray tracer (3D)')):
            image = ax.imshow(values.reshape(shape), origin='lower', cmap='viridis',
                              vmin=low, vmax=high, extent=extent)
            ax.set_title(f'{name}, {label}', fontsize=10)

        fig.colorbar(image, ax=row[:2].tolist(), label='W m$^{-2}$')

        # A symmetric scale about zero for the difference, clipped at the 99th
        # percentile so that a handful of pixels do not set it for the whole map.
        limit = np.percentile(np.abs(difference), 99) or 1.0
        image = row[2].imshow(difference.reshape(shape), origin='lower',
                              cmap='RdBu_r', vmin=-limit, vmax=limit, extent=extent)
        row[2].set_title(f'{name}, 3D minus 1D', fontsize=10)
        fig.colorbar(image, ax=row[2], label='W m$^{-2}$')

        bias = difference.mean()
        rms = np.sqrt(np.mean(difference**2))
        row[2].text(0.03, 0.97, f'mean {bias:+.1f}\nrms {rms:.1f} W m$^{{-2}}$',
                    transform=row[2].transAxes, va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=2))

        for ax in row:
            ax.tick_params(labelsize=8)
        row[0].set_ylabel('y [km]', fontsize=9)

    for ax in axes[1]:
        ax.set_xlabel('x [km]', fontsize=9)

    # Where the flux was taken is a property of the band, not of the figure: the two
    # need not have traced the same box.
    fig.suptitle(title + '\n' + ' \u00b7 '.join(wheres), fontsize=11)
    fig.savefig(path, dpi=140)
    print(f'wrote {path}')


def summary(d, which, nz, name):
    """The numbers the maps are made of, for the terminal."""
    for band, band_name in BANDS:
        plane_parallel, tracer, _ = which(d, band, nz)
        difference = tracer - plane_parallel

        print(f'{band_name:10s} {name:22s} 1D {plane_parallel.mean():7.2f}   '
              f'3D {tracer.mean():7.2f} W/m2, mean difference '
              f'{difference.mean():+6.2f}, rms {np.sqrt(np.mean(difference**2)):6.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-o', '--output', default=os.path.join(HERE, 'les_solvers'),
                   help='stem of the three figures (default: %(default)s)')
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
            f'{output} has no {", ".join(missing)}: the figures compare the two '
            'transports, so both plane-parallel and raytracing have to be on in '
            'les_cloudfield.toml.')

    g = xr.open_dataset(CASE + '_input.nc')
    z = np.asarray(g['z'])
    dz = float(np.diff(np.asarray(g['zh']))[0])
    shape = np.asarray(d['rt_flux_sfc_dir']).shape

    require_plotting()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nx, ny = shape[1], shape[0]
    field_size = f'{nx} x {ny} x {z.size} at {dz:g} m'

    plot_profiles(plt, d, g, z, dz,
                  f'Heating the two transports give, LES cumulus field, {field_size}',
                  f'{args.output}_profiles.png')
    plot_maps(plt, d, g, surface_down, z.size, shape,
              f'Downward flux at the surface, LES cumulus field, {field_size}',
              f'{args.output}_sfc_down.png')
    plot_maps(plt, d, g, domain_top_up, z.size, shape,
              f'Upward flux leaving the tracer\'s domain, LES cumulus field, '
              f'{field_size}', f'{args.output}_tod_up.png')

    summary(d, surface_down, z.size, 'down at the surface')
    summary(d, domain_top_up, z.size, 'up at the domain top')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
