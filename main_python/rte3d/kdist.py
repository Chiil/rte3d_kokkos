"""Read an RRTMGP coefficient file into the arrays Gas_optics::load expects.

This is the only place rte3d touches NetCDF. The C++ side takes plain arrays, so a
host model that cannot run Python can supply them by any other means.

Dimension order is the one thing to get right. rte3d is row-major with the g-point
outermost and the column innermost, which is the reverse of the file's Fortran-facing
order for the multidimensional tables. The transposes below are written with xarray's
dimension *names* rather than axis numbers, so they say what they mean:

    kmajor        file (temperature, pressure_interp, mixing_fraction, gpt)
                  rte3d (gpt, pressure_interp, mixing_fraction, temperature)
    kminor_*      file (temperature, mixing_fraction, contributors)
                  rte3d (contributors, mixing_fraction, temperature)

vmr_ref, key_species, totplnk and the (n, 2) index pairs already match.
"""
import numpy as np
import xarray as xr


def _strings(da):
    """Fixed-width character variables come back as bytes."""
    return [s.decode().strip() for s in da.values]


def read_kdist(path):
    """Return the dict that rte3d.load_kdist takes.

    Indices stay 1-based here, exactly as stored; the C++ loader converts.
    """
    d = xr.open_dataset(path)

    out = dict(
        gas_names=_strings(d['gas_names']),
        gas_minor=_strings(d['gas_minor']),
        identifier_minor=_strings(d['identifier_minor']),
        minor_gases_lower=_strings(d['minor_gases_lower']),
        minor_gases_upper=_strings(d['minor_gases_upper']),
        scaling_gas_lower=_strings(d['scaling_gas_lower']),
        scaling_gas_upper=_strings(d['scaling_gas_upper']),

        key_species=_i4(d['key_species']),
        band2gpt=_i4(d['bnd_limits_gpt']),
        band_lims_wavenum=_f8(d['bnd_limits_wavenumber']),
        press_ref=_f8(d['press_ref']),
        temp_ref=_f8(d['temp_ref']),
        press_ref_trop=float(d['press_ref_trop']),
        temp_ref_p=float(d['absorption_coefficient_ref_P']),
        temp_ref_t=float(d['absorption_coefficient_ref_T']),

        vmr_ref=_f8(d['vmr_ref']),
        kmajor=_f8(d['kmajor'].transpose(
            'gpt', 'pressure_interp', 'mixing_fraction', 'temperature')),

        kminor_lower=_f8(d['kminor_lower'].transpose(
            'contributors_lower', 'mixing_fraction', 'temperature')),
        kminor_upper=_f8(d['kminor_upper'].transpose(
            'contributors_upper', 'mixing_fraction', 'temperature')),
        minor_limits_gpt_lower=_i4(d['minor_limits_gpt_lower']),
        minor_limits_gpt_upper=_i4(d['minor_limits_gpt_upper']),
        minor_scales_with_density_lower=_i1(d['minor_scales_with_density_lower']),
        minor_scales_with_density_upper=_i1(d['minor_scales_with_density_upper']),
        scale_by_complement_lower=_i1(d['scale_by_complement_lower']),
        scale_by_complement_upper=_i1(d['scale_by_complement_upper']),
        kminor_start_lower=_i4(d['kminor_start_lower']),
        kminor_start_upper=_i4(d['kminor_start_upper']),
    )

    # Longwave files carry the Planck tables, shortwave ones the Rayleigh
    # coefficients and the solar source. The variable is misspelled in the file.
    if 'totplnk' in d:
        out['totplnk'] = _f8(d['totplnk'])
        out['planck_frac'] = _f8(d['plank_fraction'].transpose(
            'gpt', 'pressure_interp', 'mixing_fraction', 'temperature'))
        out['optimal_angle_fit'] = _f8(d['optimal_angle_fit'])

    if 'rayl_lower' in d:
        lower = d['rayl_lower'].transpose('gpt', 'mixing_fraction', 'temperature')
        upper = d['rayl_upper'].transpose('gpt', 'mixing_fraction', 'temperature')
        out['rayl'] = np.ascontiguousarray(
            np.stack([lower.values, upper.values]).astype(np.float64))

        out['solar_source_quiet'] = _f8(d['solar_source_quiet'])
        out['solar_source_facular'] = _f8(d['solar_source_facular'])
        out['solar_source_sunspot'] = _f8(d['solar_source_sunspot'])
        out['tsi_default'] = float(d['tsi_default'])
        out['mg_default'] = float(d['mg_default'])
        out['sb_default'] = float(d['sb_default'])

    return out


def _f8(da):
    return np.ascontiguousarray(da.values.astype(np.float64))


def _i4(da):
    return np.ascontiguousarray(da.values.astype(np.int32))


def _i1(da):
    return np.ascontiguousarray(da.values.astype(np.int8))


def read_cloud_optics(path):
    """Read a cloud optics coefficient file into the dict rte3d.load_cloud_optics takes.

    The spectral dimension is whatever the file provides: the -bnd files are resolved
    by band, the -g### files by g-point.

    The ice size bounds are named diamice_lwr/upr in current files and radice_lwr/upr
    in older ones. The numbers are the same, but RRTMGP now documents the quantity as
    an effective *diameter*, so a host model supplying an effective radius has to
    double it.

    The lookup tables themselves are named extliq and the rest in current files and
    lut_extliq in older ones, beside the Pade coefficients rte3d does not use. Same
    arrays, same dimensions; only the prefix moved. Reading both is what lets a case
    written for rte-rrtmgp-cpp run here on its own coefficient files.
    """
    d = xr.open_dataset(path)

    lut = 'lut_' if 'lut_extliq' in d else ''
    spectral = 'ngpt' if 'ngpt' in d[f'{lut}extliq'].dims else 'nband'
    ice_lwr = 'diamice_lwr' if 'diamice_lwr' in d else 'radice_lwr'
    ice_upr = 'diamice_upr' if 'diamice_upr' in d else 'radice_upr'

    def liq(name):
        return _f8(d[f'{lut}{name}'].transpose(spectral, 'nsize_liq'))

    def ice(name):
        return _f8(d[f'{lut}{name}'].transpose('nrghice', spectral, 'nsize_ice'))

    return dict(
        radliq_lwr=float(d['radliq_lwr']), radliq_upr=float(d['radliq_upr']),
        radice_lwr=float(d[ice_lwr]), radice_upr=float(d[ice_upr]),
        extliq=liq('extliq'), ssaliq=liq('ssaliq'), asyliq=liq('asyliq'),
        extice=ice('extice'), ssaice=ice('ssaice'), asyice=ice('asyice'),
        band_lims_wavenum=_f8(d['bnd_limits_wavenumber']),
    )
