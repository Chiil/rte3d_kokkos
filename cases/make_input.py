#!/usr/bin/env python
"""Write a case input file, and its settings, for run_case.py.

    python cases/make_input.py mycase [--nx 64] [--ny 64] [--clouds]

The atmosphere is the RCEMIP radiative-convective-equilibrium profile of Wing et al.
(2018), for a 300 K sea surface: an analytic sounding, so this needs no input data of
its own and is a self-contained starting point. Every column holds the same profile.

The point of the script is the file it writes rather than the profile in it: it is a
worked example of the layout run_case.py reads, documented in rte3d.case and in
cases/README.md, and the easiest way to see what your own writer has to produce.
"""
import argparse
import os

import numpy as np
import xarray as xr

# Band counts of the coefficient files in extern/rrtmgp-data. The surface conditions
# are resolved by band, so a file is written for a particular pair.
NBND_LW = 16
NBND_SW = 14

# The RCEMIP boundary conditions, as in the case's own description.
SST = 300.0
SFC_EMIS = 1.0
SFC_ALBEDO = 0.07
SOLAR_ZENITH_ANGLE = 42.05
TSI = 551.58

# Well-mixed gases, in volume mixing ratio.
FIXED_GASES = dict(co2=348e-6, ch4=1650e-9, n2o=306e-9, n2=0.7808, o2=0.2095)

# A liquid cloud in the boundary layer and an ice cloud in the upper troposphere, by
# height. The water paths are per layer, in g/m2, and the sizes in microns; both sit
# well inside the range the cloud optics tables cover.
LIQUID = dict(bounds=(1.0e3, 3.0e3), water_path=10.0, size=10.0)
ICE = dict(bounds=(8.0e3, 12.0e3), water_path=10.0, size=60.0)


def rcemip_profile(nlay, z_top=70.0e3, sst=SST):
    """Pressure, temperature, water vapour and ozone on nlay layers.

    The layers are equally spaced in height with the levels on their edges, and the
    surface is at index 0. Water vapour is specific humidity in the formulation and
    is converted to the volume mixing ratio RRTMGP wants.
    """
    dz = z_top/nlay
    z = np.arange(dz/2, z_top, dz)
    zh = np.append(np.arange(0.0, z_top - dz/2, dz), z_top)

    def p_q_t(z):
        q_0 = 0.01864       # for a 300 K sea surface
        z_q1, z_q2, z_t = 4.0e3, 7.5e3, 15.0e3
        t_0, gamma = sst, 6.7e-3
        g, rd, p_0 = 9.79764, 287.04, 101480.0

        q = q_0*np.exp(-z/z_q1)*np.exp(-(z/z_q2)**2)

        # Above the tropopause both the moisture and the virtual temperature are held
        # at their tropopause value, which keeps the profile continuous there.
        above = z > z_t
        q[above] = q_0*np.exp(-z_t/z_q1)*np.exp(-(z_t/z_q2)**2)

        tv_0 = (1.0 + 0.608*q_0)*t_0
        tv = tv_0 - gamma*z
        tv_t = tv_0 - gamma*z_t
        tv[above] = tv_t

        p = p_0*(tv/tv_0)**(g/(rd*gamma))
        p[above] = (p_0*(tv_t/tv_0)**(g/(rd*gamma))
                    * np.exp(-(g*(z[above] - z_t))/(rd*tv_t)))

        return p, q, tv/(1.0 + 0.608*q)

    p_lay, q, t_lay = p_q_t(z)
    p_lev, _, t_lev = p_q_t(zh)

    rd_rv = 287.04/461.5
    h2o = q/(rd_rv*(1.0 - q))

    # Ozone from the RCEMIP analytic fit, in ppmv. The floor is what keeps a single
    # precision build from failing on vanishing concentrations aloft.
    p_hpa = p_lay/100.0
    o3 = np.maximum(1e-13, 3.6478*p_hpa**0.83209*np.exp(-p_hpa/11.3515)*1e-6)

    return dict(z=z, zh=zh, p_lay=p_lay, p_lev=p_lev, t_lay=t_lay, t_lev=t_lev,
                h2o=h2o, o3=o3)


def cloud_fields(z):
    """Water paths and particle sizes for the two cloud layers, on the layer grid.

    Ice size is an effective diameter, which is what the current cloud coefficient
    files tabulate and what run_case.py reads as `dei`.
    """
    fields = {}
    for name, (spec, path, size) in dict(
            liquid=(LIQUID, 'lwp', 'rel'), ice=(ICE, 'iwp', 'dei')).items():
        lower, upper = spec['bounds']
        inside = (z >= lower) & (z < upper)
        fields[path] = np.where(inside, spec['water_path'], 0.0)
        fields[size] = np.where(inside, spec['size'], 0.0)

    return fields


def make_input(path, nx=64, ny=64, nlay=256, clouds=False, sst=SST,
               dx=100.0, dy=100.0, rt_nz=None):
    """Write one case input file. Returns its path."""
    profile = rcemip_profile(nlay, sst=sst)

    def field(values):
        """One profile, tiled over the horizontal, as (lay|lev, y, x)."""
        return np.ascontiguousarray(
            np.broadcast_to(values[:, None, None], (values.size, ny, nx)))

    # The Cartesian grid, which only the ray tracer reads. The horizontal spacing is
    # ours to choose; the vertical is the sounding's own, which is equally spaced, as
    # the tracer requires. rt_nz cells are resolved in three dimensions and everything
    # above them is lumped into one more cell on top.
    rt_nz = nlay if rt_nz is None else min(rt_nz, nlay)
    dz = profile['zh'][1] - profile['zh'][0]

    variables = {
        'x': (('x',), (np.arange(nx) + 0.5)*dx),
        'xh': (('xh',), np.arange(nx + 1)*dx),
        'y': (('y',), (np.arange(ny) + 0.5)*dy),
        'yh': (('yh',), np.arange(ny + 1)*dy),
        'z': (('z',), profile['z'][:rt_nz]),
        'zh': (('zh',), profile['zh'][:rt_nz + 1]),

        # Blocks of the null-collision grid the tracer marches on.
        'ngrid_x': ((), np.int32(max(1, nx//4))),
        'ngrid_y': ((), np.int32(max(1, ny//4))),
        'ngrid_z': ((), np.int32(max(1, rt_nz//4))),

        'azi': (('y', 'x'), np.zeros((ny, nx))),

        'z_lay': (('lay',), profile['z']), 'z_lev': (('lev',), profile['zh']),
        'p_lay': (('lay', 'y', 'x'), field(profile['p_lay'])),
        'p_lev': (('lev', 'y', 'x'), field(profile['p_lev'])),
        't_lay': (('lay', 'y', 'x'), field(profile['t_lay'])),
        't_lev': (('lev', 'y', 'x'), field(profile['t_lev'])),
        'vmr_h2o': (('lay', 'y', 'x'), field(profile['h2o'])),
        'vmr_o3': (('lay', 'y', 'x'), field(profile['o3'])),

        't_sfc': (('y', 'x'), np.full((ny, nx), sst)),
        'emis_sfc': (('y', 'x', 'band_lw'), np.full((ny, nx, NBND_LW), SFC_EMIS)),

        'mu0': (('y', 'x'), np.full((ny, nx), np.cos(np.deg2rad(SOLAR_ZENITH_ANGLE)))),
        'sfc_alb_dir': (('y', 'x', 'band_sw'), np.full((ny, nx, NBND_SW), SFC_ALBEDO)),
        'sfc_alb_dif': (('y', 'x', 'band_sw'), np.full((ny, nx, NBND_SW), SFC_ALBEDO)),
        'tsi': (('y', 'x'), np.full((ny, nx), TSI)),
    }

    # The well-mixed gases are scalars, which is one of the three ranks a gas may have.
    for name, value in FIXED_GASES.items():
        variables[f'vmr_{name}'] = ((), np.float64(value))

    if clouds:
        for name, values in cloud_fields(profile['z']).items():
            variables[name] = (('lay', 'y', 'x'), field(values))

    xr.Dataset(variables).to_netcdf(path)

    return path


def make_settings(path, clouds=False):
    """Write the settings file that goes with the input, if there is not one already."""
    if os.path.exists(path):
        return None

    with open(path, 'w') as f:
        f.write('[switches]\n'
                'longwave = true\n'
                'shortwave = true\n'
                f'cloud-optics = {"true" if clouds else "false"}\n'
                '\n'
                '# Which shortwave solver, or both. The ray tracer reads the Cartesian\n'
                '# grid this file was written with; see example.toml for the rest.\n'
                '[shortwave]\n'
                'plane-parallel = true\n'
                'raytracing = false\n'
                'photons-per-pixel = 256\n')

    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('case', help='case name: writes CASE_input.nc and CASE.toml')
    p.add_argument('--nx', type=int, default=64, help='columns in x (default: %(default)s)')
    p.add_argument('--ny', type=int, default=64, help='columns in y (default: %(default)s)')
    p.add_argument('--nlay', type=int, default=256,
                   help='layers, equally spaced up to 70 km (default: %(default)s)')
    p.add_argument('--dx', type=float, default=100.0,
                   help='horizontal grid spacing in x, for the ray tracer '
                        '(default: %(default)s m)')
    p.add_argument('--dy', type=float, default=100.0,
                   help='horizontal grid spacing in y (default: %(default)s m)')
    p.add_argument('--rt-nz', type=int, default=None,
                   help='layers the ray tracer resolves in three dimensions; the rest '
                        'are lumped into one cell on top (default: all of them)')
    p.add_argument('--sst', type=float, default=SST,
                   help='sea surface temperature, which sets the whole sounding '
                        '(default: %(default)s K)')
    p.add_argument('--clouds', action='store_true',
                   help='add a liquid and an ice cloud layer, and switch cloud optics '
                        'on in the settings')
    p.add_argument('--no-settings', action='store_true',
                   help='write only the input file')
    args = p.parse_args()

    path = make_input(f'{args.case}_input.nc', args.nx, args.ny, args.nlay,
                      args.clouds, args.sst, args.dx, args.dy, args.rt_nz)
    print(f'wrote {path}   {args.nx*args.ny} columns, {args.nlay} layers'
          f'{", with clouds" if args.clouds else ""}')

    if not args.no_settings:
        settings = make_settings(f'{args.case}.toml', args.clouds)
        print(f'wrote {settings}' if settings
              else f'{args.case}.toml exists already; left as it is')

    print(f'\n    python cases/run_case.py {args.case}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
