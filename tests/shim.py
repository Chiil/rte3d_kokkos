"""ctypes wrapper around the reduction oracle in tests/shim/rte3d_shim.F90.

The reduction inside ty_gas_optics_rrtmgp%load has no bind(C) entry point, so this
shim exposes it. Python passes the same raw arrays to both implementations.

Array shapes below are rte3d's, which for a Fortran column-major array is the same
memory reversed. The one exception is noted where it occurs.
"""
import ctypes

import numpy as np

NAME_LEN = 32
FLOAT = np.float64

_f8 = np.ctypeslib.ndpointer(dtype=FLOAT, flags='C_CONTIGUOUS')
_i4 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
_p = ctypes.c_void_p


def _names_buf(names):
    """Pack names into the fixed-width character buffer the shim unpacks."""
    buf = bytearray(b' ' * (NAME_LEN * len(names)))
    for i, name in enumerate(names):
        encoded = name.encode()[:NAME_LEN]
        buf[i*NAME_LEN:i*NAME_LEN + len(encoded)] = encoded

    return ctypes.create_string_buffer(bytes(buf), NAME_LEN*len(names))


def _i(v):
    return ctypes.byref(ctypes.c_int(int(v)))


def _d(v):
    return ctypes.byref(ctypes.c_double(float(v)))


class Shim:
    def __init__(self, lib):
        self.lib = lib

        lib.rte3d_shim_load.restype = None
        lib.rte3d_shim_load.argtypes = (
            [_p]*14 + [ctypes.c_char_p]*8
            + [_i4, _i4, _f8, _f8, _f8] + [_p]*3
            + [_f8]*4 + [_i4]*2 + [_i4]*4 + [_i4]*2 + [_f8]*3 + [_p])

        lib.rte3d_shim_dims.restype = None
        lib.rte3d_shim_dims.argtypes = [_p]*6

        lib.rte3d_shim_get_main.restype = None
        lib.rte3d_shim_get_main.argtypes = [_i4, _i4, _f8]

        lib.rte3d_shim_get_minor.restype = None
        lib.rte3d_shim_get_minor.argtypes = [_p] + [_i4]*6 + [_f8]

        lib.rte3d_shim_cloud_optics.restype = None
        lib.rte3d_shim_cloud_optics.argtypes = (
            [_p]*8            # nspec nsize_liq nsize_ice nrghice icergh ncol nlay two_stream
            + [_f8]           # band_lims_wvn
            + [_p]*4          # radliq_lwr/upr radice_lwr/upr
            + [_f8]*6         # extliq ssaliq asyliq extice ssaice asyice
            + [_f8]*4         # clwp ciwp reliq reice
            + [_f8]*3         # tau ssa g
            + [_p])           # status

    def load(self, f, available):
        ngas_file = len(f['gas_names'])
        nminor_abs = len(f['gas_minor'])
        nminor_lower = len(f['minor_gases_lower'])
        nminor_upper = len(f['minor_gases_upper'])
        ngpt = f['kmajor'].shape[0]
        nbnd = f['band2gpt'].shape[0]
        npres = f['press_ref'].shape[0]
        ntemp = f['temp_ref'].shape[0]
        neta = f['kmajor'].shape[2]
        nplancktemp = f['totplnk'].shape[1]
        nfit = f['optimal_angle_fit'].shape[1]

        status = ctypes.c_int(-1)

        self.lib.rte3d_shim_load(
            _i(ngas_file), _i(nminor_abs), _i(nminor_lower), _i(nminor_upper),
            _i(ngpt), _i(nbnd), _i(npres), _i(ntemp), _i(neta), _i(nplancktemp), _i(nfit),
            _i(f['kminor_lower'].shape[0]), _i(f['kminor_upper'].shape[0]), _i(len(available)),
            _names_buf(f['gas_names']), _names_buf(f['gas_minor']),
            _names_buf(f['identifier_minor']),
            _names_buf(f['minor_gases_lower']), _names_buf(f['minor_gases_upper']),
            _names_buf(f['scaling_gas_lower']), _names_buf(f['scaling_gas_upper']),
            _names_buf(available),
            f['key_species'], f['band2gpt'], f['band_lims_wavenum'],
            f['press_ref'], f['temp_ref'],
            _d(f['press_ref_trop']), _d(f['temp_ref_p']), _d(f['temp_ref_t']),
            # load() wants the contributor axis fastest in Fortran, i.e. slowest in C;
            # rte3d stores it the other way, matching load()'s own reduced output.
            f['vmr_ref'], f['kmajor'],
            np.ascontiguousarray(f['kminor_lower'].transpose(2, 1, 0)),
            np.ascontiguousarray(f['kminor_upper'].transpose(2, 1, 0)),
            f['minor_limits_gpt_lower'], f['minor_limits_gpt_upper'],
            f['minor_scales_with_density_lower'].astype(np.int32),
            f['minor_scales_with_density_upper'].astype(np.int32),
            f['scale_by_complement_lower'].astype(np.int32),
            f['scale_by_complement_upper'].astype(np.int32),
            f['kminor_start_lower'], f['kminor_start_upper'],
            f['totplnk'], f['planck_frac'], f['optimal_angle_fit'],
            ctypes.byref(status))

        if status.value != 0:
            raise RuntimeError(f'rte3d_shim_load failed with status {status.value}')

        return self._collect(ngpt, ntemp, neta)

    def _collect(self, ngpt, ntemp, neta):
        dims = [ctypes.c_int(0) for _ in range(6)]
        self.lib.rte3d_shim_dims(*[ctypes.byref(d) for d in dims])
        ngas, nflav, nm_lo, nc_lo, nm_up, nc_up = [d.value for d in dims]

        flavor = np.zeros((nflav, 2), dtype=np.int32)
        gpoint_flavor = np.zeros((ngpt, 2), dtype=np.int32)
        vmr_ref = np.zeros((ntemp, ngas + 1, 2), dtype=FLOAT)
        self.lib.rte3d_shim_get_main(flavor, gpoint_flavor, vmr_ref)

        out = dict(gas_names=None, flavor=flavor, gpoint_flavor=gpoint_flavor,
                   vmr_ref=vmr_ref)

        for side, prefix, nm, nc in ((0, 'lower_', nm_lo, nc_lo), (1, 'upper_', nm_up, nc_up)):
            limits = np.zeros((nm, 2), dtype=np.int32)
            density = np.zeros(nm, dtype=np.int32)
            complement = np.zeros(nm, dtype=np.int32)
            idx_minor = np.zeros(nm, dtype=np.int32)
            idx_scaling = np.zeros(nm, dtype=np.int32)
            kminor_start = np.zeros(nm, dtype=np.int32)
            kminor = np.zeros((nc, neta, ntemp), dtype=FLOAT)

            self.lib.rte3d_shim_get_minor(
                _i(side), limits, density, complement,
                idx_minor, idx_scaling, kminor_start, kminor)

            out[prefix + 'minor_limits_gpt'] = limits
            out[prefix + 'scales_with_density'] = density
            out[prefix + 'scale_by_complement'] = complement
            out[prefix + 'idx_minor'] = idx_minor
            out[prefix + 'idx_minor_scaling'] = idx_scaling
            out[prefix + 'kminor_start'] = kminor_start
            out[prefix + 'kminor'] = kminor

        return out

    def cloud_optics(self, f, clwp, ciwp, reliq, reice, icergh=0, two_stream=True):
        """Cloud optical properties from the reference.

        Table shapes here are rte3d's; the reference wants them reversed, which for a
        Fortran column-major array is the same memory, except that the tables are
        indexed (size, spectral) there and (spectral, size) here.
        """
        nlay, ncol = clwp.shape
        nspec, nsize_liq = f['extliq'].shape
        nrghice, _, nsize_ice = f['extice'].shape

        # The reference's cloud optics is band-resolved: it wants one spectral
        # discretisation entry per table column. A g-point-resolved coefficient file
        # has more table columns than bands, so synthesise one interval per column.
        # The wavenumber limits play no part in the lookup -- only the size index does
        # -- so this does not change the answer.
        band_lims = np.ascontiguousarray(f['band_lims_wavenum'])
        if band_lims.shape[0] != nspec:
            edges = np.linspace(1.0, 1.0 + nspec, nspec + 1)
            band_lims = np.ascontiguousarray(
                np.stack([edges[:-1], edges[1:]], axis=1))

        n = nspec*nlay*ncol
        tau = np.zeros(n, dtype=FLOAT)
        ssa = np.zeros(n, dtype=FLOAT)
        g = np.zeros(n, dtype=FLOAT)
        status = ctypes.c_int(-1)

        self.lib.rte3d_shim_cloud_optics(
            _i(nspec), _i(nsize_liq), _i(nsize_ice), _i(nrghice), _i(icergh + 1),
            _i(ncol), _i(nlay), _i(1 if two_stream else 0),
            band_lims,
            _d(f['radliq_lwr']), _d(f['radliq_upr']),
            _d(f['radice_lwr']), _d(f['radice_upr']),
            f['extliq'], f['ssaliq'], f['asyliq'],
            f['extice'], f['ssaice'], f['asyice'],
            clwp, ciwp, reliq, reice,
            tau, ssa, g, ctypes.byref(status))

        if status.value != 0:
            raise RuntimeError(f'rte3d_shim_cloud_optics failed with status {status.value}')

        # The reference writes (ncol, nlay, nspec) in Fortran order, i.e. C order
        # (nspec, nlay, ncol) once reshaped.
        shape = (nspec, nlay, ncol)
        if two_stream:
            return tau.reshape(shape), ssa.reshape(shape), g.reshape(shape)
        return tau.reshape(shape)
