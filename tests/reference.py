"""ctypes wrapper around the deprecated Fortran reference, used as a test oracle.

rte3d never links this. See the `fortran_ref` fixture in conftest.py.

The prototypes follow rte-rrtmgp/rte-kernels/api/rte_kernels.h. Every scalar there is
declared `const int&` / `const Bool&`, so it is passed by reference. `Bool` is C `int`
unless the reference was compiled with -DRTE_USE_CBOOL, and `Float` is double unless it
was compiled with -DRTE_USE_SP.
"""
import ctypes
import os

import numpy as np

FLOAT = np.float64
_BOOL = ctypes.c_char if os.environ.get('RTE3D_FORTRAN_REF_CBOOL') else ctypes.c_int

_f8 = np.ctypeslib.ndpointer(dtype=FLOAT, flags='C_CONTIGUOUS')


def _int(value):
    return ctypes.byref(ctypes.c_int(value))


def _bool(value):
    return ctypes.byref(_BOOL(1 if value else 0))


def _dbl(value):
    return ctypes.byref(ctypes.c_double(value))


class Reference:
    """The reference kernels, called on (ngpt, nlay, ncol) C-order numpy arrays.

    No transposes anywhere: a C-order (ngpt, nlay, ncol) array is byte-for-byte the
    Fortran (ncol, nlay, ngpt) array the kernels expect.
    """

    def __init__(self, lib):
        self.lib = lib

        lib.rte_sw_solver_noscat.restype = None
        lib.rte_sw_solver_noscat.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            _f8, _f8, _f8, _f8]

        lib.rte_sw_solver_2stream.restype = None
        lib.rte_sw_solver_2stream.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            _f8, _f8, _f8, _f8, _f8, _f8, _f8,
            _f8, _f8, _f8,
            ctypes.c_void_p, _f8,
            ctypes.c_void_p, _f8, _f8, _f8]

        lib.rte_lw_solver_noscat.restype = None
        lib.rte_lw_solver_noscat.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, _f8, _f8,
            _f8, _f8, _f8, _f8, _f8, _f8,
            _f8, _f8,
            ctypes.c_void_p, _f8, _f8,
            ctypes.c_void_p, _f8, _f8,
            ctypes.c_void_p, _f8, _f8]

        lib.rte_lw_solver_2stream.restype = None
        lib.rte_lw_solver_2stream.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            _f8, _f8, _f8, _f8, _f8, _f8, _f8, _f8,
            _f8, _f8]

        _i4 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        _p = ctypes.c_void_p  # scalars, all passed by reference

        # (name, number of leading scalar arguments, number of float array arguments)
        for name, n_scalar, n_array in [
                ('rte_delta_scale_2str_k', 3, 3),
                ('rte_delta_scale_2str_f_k', 3, 4),
                ('rte_increment_1scalar_by_1scalar', 3, 2),
                ('rte_increment_1scalar_by_2stream', 3, 3),
                ('rte_increment_2stream_by_1scalar', 3, 3),
                ('rte_increment_2stream_by_2stream', 3, 6),
                ('rte_increment_2stream_by_nstream', 4, 6),
                ('rte_increment_nstream_by_2stream', 4, 6),
                ('rte_increment_nstream_by_nstream', 5, 6),
                ('rte_sum_broadband', 3, 2),
                ('rte_net_broadband_full', 3, 3),
                ('rte_net_broadband_precalc', 2, 3)]:
            fn = getattr(lib, name)
            fn.restype = None
            fn.argtypes = [_p]*n_scalar + [_f8]*n_array

        # The by-band variants take the band limits after the arrays.
        for name, n_array in [
                ('rte_inc_1scalar_by_1scalar_bybnd', 2),
                ('rte_inc_1scalar_by_2stream_bybnd', 3),
                ('rte_inc_2stream_by_1scalar_bybnd', 3),
                ('rte_inc_2stream_by_2stream_bybnd', 6)]:
            fn = getattr(lib, name)
            fn.restype = None
            fn.argtypes = [_p]*3 + [_f8]*n_array + [_p, _i4]

        lib.rrtmgp_interpolation.restype = None
        lib.rrtmgp_interpolation.argtypes = (
            [_p]*7            # ncol nlay ngas nflav neta npres ntemp
            + [_i4]           # flavor
            + [_f8]*2         # press_ref_log temp_ref
            + [_p]*4          # press_ref_log_delta temp_ref_min temp_ref_delta press_ref_trop_log
            + [_f8]*4         # vmr_ref play tlay col_gas
            + [_i4]           # jtemp
            + [_f8]*3         # fmajor fminor col_mix
            + [_i4]*3)        # tropo jeta jpress

        lib.rrtmgp_compute_tau_absorption.restype = None
        lib.rrtmgp_compute_tau_absorption.argtypes = (
            [_p]*14           # ncol nlay nband ngpt ngas nflav neta npres ntemp
                              # nminorlower nminorklower nminorupper nminorkupper idx_h2o
            + [_i4]*2         # gpoint_flavor band_lims_gpt
            + [_f8]*3         # kmajor kminor_lower kminor_upper
            + [_i4]*2         # minor_limits_gpt lower/upper
            + [_i4]*4         # scales_with_density, scale_by_complement (lower/upper)
            + [_i4]*6         # idx_minor, idx_minor_scaling, kminor_start (lower/upper)
            + [_i4]           # tropo
            + [_f8]*3         # col_mix fmajor fminor
            + [_f8]*3         # play tlay col_gas
            + [_i4]*3         # jeta jtemp jpress
            + [_f8])          # tau

        lib.rrtmgp_compute_tau_rayleigh.restype = None
        lib.rrtmgp_compute_tau_rayleigh.argtypes = (
            [_p]*9            # ncol nlay nband ngpt ngas nflav neta npres ntemp
            + [_i4]*2         # gpoint_flavor band_lims_gpt
            + [_f8]           # krayl
            + [_p]            # idx_h2o
            + [_f8]*3         # col_dry col_gas fminor
            + [_i4]*3         # jeta tropo jtemp
            + [_f8])          # tau_rayleigh

        lib.rrtmgp_compute_Planck_source.restype = None
        lib.rrtmgp_compute_Planck_source.argtypes = (
            [_p]*9            # ncol nlay nbnd ngpt nflav neta npres ntemp nPlanckTemp
            + [_f8]*3         # tlay tlev tsfc
            + [_p]            # sfc_lay
            + [_f8]           # fmajor
            + [_i4]*4         # jeta tropo jtemp jpress
            + [_i4]*2         # gpoint_bands band_lims_gpt
            + [_f8]           # pfracin
            + [_p]*2          # temp_ref_min totplnk_delta
            + [_f8]           # totplnk
            + [_i4]           # gpoint_flavor
            + [_f8]*4)        # sfc_src lay_src lev_src sfc_source_Jac

        # rte_sum_byband / rte_net_byband_full live in extensions/mo_fluxes_byband.F90,
        # not in the kernels this library is built from. They are plain band-wise sums,
        # so the tests check them against numpy instead.

    def sw_solver_noscat(self, top_at_1, tau, mu0, inc_flux_dir):
        ngpt, nlay, ncol = tau.shape
        flux_dir = np.zeros((ngpt, nlay+1, ncol), dtype=FLOAT)

        self.lib.rte_sw_solver_noscat(
            _int(ncol), _int(nlay), _int(ngpt), _bool(top_at_1),
            tau, mu0, inc_flux_dir, flux_dir)

        return flux_dir

    def sw_solver_2stream(self, top_at_1, tau, ssa, g, mu0,
                          sfc_alb_dir, sfc_alb_dif, inc_flux_dir, inc_flux_dif=None):
        ngpt, nlay, ncol = tau.shape
        nlev = nlay + 1

        has_dif_bc = inc_flux_dif is not None
        if not has_dif_bc:
            inc_flux_dif = np.zeros((ngpt, ncol), dtype=FLOAT)

        flux_up = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)
        flux_dn = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)
        flux_dir = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)

        # Unused when do_broadband is false, but the arguments must still be addressable.
        broadband = [np.zeros((nlev, ncol), dtype=FLOAT) for _ in range(3)]

        self.lib.rte_sw_solver_2stream(
            _int(ncol), _int(nlay), _int(ngpt), _bool(top_at_1),
            tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
            flux_up, flux_dn, flux_dir,
            _bool(has_dif_bc), inc_flux_dif,
            _bool(False), *broadband)

        return flux_up, flux_dn, flux_dir

    def lw_solver_noscat(self, top_at_1, secants, weights, tau,
                         lay_source, lev_source, sfc_emis, sfc_source, inc_flux,
                         sfc_source_jac=None):
        ngpt, nlay, ncol = tau.shape
        nlev = nlay + 1
        nmus = weights.shape[0]

        do_jacobians = sfc_source_jac is not None
        if not do_jacobians:
            sfc_source_jac = np.zeros((ngpt, ncol), dtype=FLOAT)

        flux_up = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)
        flux_dn = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)

        # Note: rte_kernels.h documents flux_upJac as (ncol, nlay+1, ngpt), but the
        # Fortran declares it (ncol, nlay+1). Only broadband Jacobians are provided.
        flux_up_jac = np.zeros((nlev, ncol), dtype=FLOAT)

        # Unused when do_broadband is false, but must still be addressable.
        broadband = [np.zeros((nlev, ncol), dtype=FLOAT) for _ in range(2)]

        # ssa and g are referenced only when do_rescaling is true.
        ssa = np.zeros((ngpt, nlay, ncol), dtype=FLOAT)
        g = np.zeros((ngpt, nlay, ncol), dtype=FLOAT)

        self.lib.rte_lw_solver_noscat(
            _int(ncol), _int(nlay), _int(ngpt), _bool(top_at_1),
            _int(nmus), secants, weights,
            tau, lay_source, lev_source, sfc_emis, sfc_source, inc_flux,
            flux_up, flux_dn,
            _bool(False), *broadband,
            _bool(do_jacobians), sfc_source_jac, flux_up_jac,
            _bool(False), ssa, g)

        return flux_up, flux_dn, (flux_up_jac if do_jacobians else None)

    def lw_solver_2stream(self, top_at_1, tau, ssa, g,
                          lay_source, lev_source, sfc_emis, sfc_source, inc_flux):
        ngpt, nlay, ncol = tau.shape
        nlev = nlay + 1

        flux_up = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)
        flux_dn = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)

        self.lib.rte_lw_solver_2stream(
            _int(ncol), _int(nlay), _int(ngpt), _bool(top_at_1),
            tau, ssa, g, lay_source, lev_source, sfc_emis, sfc_source, inc_flux,
            flux_up, flux_dn)

        return flux_up, flux_dn

    # --- optical properties -------------------------------------------------

    def delta_scale_2str(self, tau, ssa, g, f=None):
        ngpt, nlay, ncol = tau.shape
        tau, ssa, g = tau.copy(), ssa.copy(), g.copy()

        if f is None:
            self.lib.rte_delta_scale_2str_k(_int(ncol), _int(nlay), _int(ngpt), tau, ssa, g)
        else:
            self.lib.rte_delta_scale_2str_f_k(_int(ncol), _int(nlay), _int(ngpt), tau, ssa, g, f)

        return tau, ssa, g

    def _sizes(self, tau1):
        ngpt, nlay, ncol = tau1.shape
        return _int(ncol), _int(nlay), _int(ngpt)

    def increment_1scalar_by_1scalar(self, tau1, tau2):
        tau1 = tau1.copy()
        self.lib.rte_increment_1scalar_by_1scalar(*self._sizes(tau1), tau1, tau2)
        return tau1

    def increment_1scalar_by_2stream(self, tau1, tau2, ssa2):
        tau1 = tau1.copy()
        self.lib.rte_increment_1scalar_by_2stream(*self._sizes(tau1), tau1, tau2, ssa2)
        return tau1

    def increment_2stream_by_1scalar(self, tau1, ssa1, tau2):
        tau1, ssa1 = tau1.copy(), ssa1.copy()
        self.lib.rte_increment_2stream_by_1scalar(*self._sizes(tau1), tau1, ssa1, tau2)
        return tau1, ssa1

    def increment_2stream_by_2stream(self, tau1, ssa1, g1, tau2, ssa2, g2):
        tau1, ssa1, g1 = tau1.copy(), ssa1.copy(), g1.copy()
        self.lib.rte_increment_2stream_by_2stream(
            *self._sizes(tau1), tau1, ssa1, g1, tau2, ssa2, g2)
        return tau1, ssa1, g1

    def increment_2stream_by_nstream(self, tau1, ssa1, g1, tau2, ssa2, p2):
        tau1, ssa1, g1 = tau1.copy(), ssa1.copy(), g1.copy()
        nmom2 = p2.shape[3]
        self.lib.rte_increment_2stream_by_nstream(
            *self._sizes(tau1), _int(nmom2), tau1, ssa1, g1, tau2, ssa2, p2)
        return tau1, ssa1, g1

    def increment_nstream_by_2stream(self, tau1, ssa1, p1, tau2, ssa2, g2):
        tau1, ssa1, p1 = tau1.copy(), ssa1.copy(), p1.copy()
        nmom1 = p1.shape[3]
        self.lib.rte_increment_nstream_by_2stream(
            *self._sizes(tau1), _int(nmom1), tau1, ssa1, p1, tau2, ssa2, g2)
        return tau1, ssa1, p1

    def increment_nstream_by_nstream(self, tau1, ssa1, p1, tau2, ssa2, p2):
        tau1, ssa1, p1 = tau1.copy(), ssa1.copy(), p1.copy()
        nmom1, nmom2 = p1.shape[3], p2.shape[3]
        self.lib.rte_increment_nstream_by_nstream(
            *self._sizes(tau1), _int(nmom1), _int(nmom2), tau1, ssa1, p1, tau2, ssa2, p2)
        return tau1, ssa1, p1

    # --- flux reduction -----------------------------------------------------

    def sum_broadband(self, spectral_flux):
        ngpt, nlev, ncol = spectral_flux.shape
        out = np.zeros((nlev, ncol), dtype=FLOAT)
        self.lib.rte_sum_broadband(_int(ncol), _int(nlev), _int(ngpt), spectral_flux, out)
        return out

    def net_broadband_full(self, flux_dn, flux_up):
        ngpt, nlev, ncol = flux_dn.shape
        out = np.zeros((nlev, ncol), dtype=FLOAT)
        self.lib.rte_net_broadband_full(
            _int(ncol), _int(nlev), _int(ngpt), flux_dn, flux_up, out)
        return out

    def net_broadband_precalc(self, flux_dn, flux_up):
        nlev, ncol = flux_dn.shape
        out = np.zeros((nlev, ncol), dtype=FLOAT)
        self.lib.rte_net_broadband_precalc(_int(ncol), _int(nlev), flux_dn, flux_up, out)
        return out

    # --- by-band increments -------------------------------------------------
    #
    # The reference declares gpt_lims as Fortran (2, nbnd). Column-major, that is the
    # same memory as a C-order (nbnd, 2) array, so these take rte3d's (nbnd, 2) shape
    # with 1-based indices.

    def inc_1scalar_by_1scalar_bybnd(self, tau1, tau2, gpt_lims):
        tau1 = tau1.copy()
        nbnd = gpt_lims.shape[0]
        self.lib.rte_inc_1scalar_by_1scalar_bybnd(
            *self._sizes(tau1), tau1, tau2, _int(nbnd), gpt_lims)
        return tau1

    def inc_1scalar_by_2stream_bybnd(self, tau1, tau2, ssa2, gpt_lims):
        tau1 = tau1.copy()
        nbnd = gpt_lims.shape[0]
        self.lib.rte_inc_1scalar_by_2stream_bybnd(
            *self._sizes(tau1), tau1, tau2, ssa2, _int(nbnd), gpt_lims)
        return tau1

    def inc_2stream_by_1scalar_bybnd(self, tau1, ssa1, tau2, gpt_lims):
        tau1, ssa1 = tau1.copy(), ssa1.copy()
        nbnd = gpt_lims.shape[0]
        self.lib.rte_inc_2stream_by_1scalar_bybnd(
            *self._sizes(tau1), tau1, ssa1, tau2, _int(nbnd), gpt_lims)
        return tau1, ssa1

    def inc_2stream_by_2stream_bybnd(self, tau1, ssa1, g1, tau2, ssa2, g2, gpt_lims):
        tau1, ssa1, g1 = tau1.copy(), ssa1.copy(), g1.copy()
        nbnd = gpt_lims.shape[0]
        self.lib.rte_inc_2stream_by_2stream_bybnd(
            *self._sizes(tau1), tau1, ssa1, g1, tau2, ssa2, g2, _int(nbnd), gpt_lims)
        return tau1, ssa1, g1

    # --- gas optics ---------------------------------------------------------

    def interpolation(self, flavor, press_ref_log, temp_ref, press_ref_log_delta,
                      temp_ref_min, temp_ref_delta, press_ref_trop_log, neta,
                      vmr_ref, play, tlay, col_gas):
        """Shapes here are rte3d's, which are the reference's reversed and therefore
        the same memory. Outputs come back in the reference's own index order; the
        tests transpose them to rte3d's."""
        nlay, ncol = play.shape
        nflav = flavor.shape[0]
        ntemp = temp_ref.shape[0]
        npres = press_ref_log.shape[0]
        ngas = col_gas.shape[0] - 1

        jtemp = np.zeros((nlay, ncol), dtype=np.int32)
        jpress = np.zeros((nlay, ncol), dtype=np.int32)
        tropo = np.zeros((nlay, ncol), dtype=np.int32)
        jeta = np.zeros((nflav, nlay, ncol, 2), dtype=np.int32)
        col_mix = np.zeros((nflav, nlay, ncol, 2), dtype=FLOAT)
        fminor = np.zeros((nflav, nlay, ncol, 2, 2), dtype=FLOAT)
        fmajor = np.zeros((nflav, nlay, ncol, 2, 2, 2), dtype=FLOAT)

        self.lib.rrtmgp_interpolation(
            _int(ncol), _int(nlay), _int(ngas), _int(nflav), _int(neta),
            _int(npres), _int(ntemp),
            flavor, press_ref_log, temp_ref,
            _dbl(press_ref_log_delta), _dbl(temp_ref_min), _dbl(temp_ref_delta),
            _dbl(press_ref_trop_log),
            vmr_ref, play, tlay, col_gas,
            jtemp, fmajor, fminor, col_mix, tropo, jeta, jpress)

        return dict(jtemp=jtemp, jpress=jpress, tropo=tropo, jeta=jeta,
                    col_mix=col_mix, fminor=fminor, fmajor=fmajor)

    def compute_tau_absorption(self, kdist, interp, play, tlay, col_gas, neta):
        """kdist and interp use rte3d's 0-based indices; this converts to the
        reference's 1-based ones and passes its own interpolation output straight
        through, so only the tau kernel is under test."""
        nlay, ncol = play.shape
        ngpt = kdist['kmajor'].shape[0]
        nbnd = kdist['band_lims_gpt'].shape[0]
        ngas = col_gas.shape[0] - 1
        nflav = kdist['flavor'].shape[0]
        ntemp = kdist['kmajor'].shape[3]
        npres = kdist['kmajor'].shape[1] - 1

        lo = {k[6:]: v for k, v in kdist.items() if k.startswith('lower_')}
        up = {k[6:]: v for k, v in kdist.items() if k.startswith('upper_')}

        tau = np.zeros((ngpt, nlay, ncol), dtype=FLOAT)

        def i32(a, base=1):
            return np.ascontiguousarray((a + base).astype(np.int32))

        self.lib.rrtmgp_compute_tau_absorption(
            _int(ncol), _int(nlay), _int(nbnd), _int(ngpt),
            _int(ngas), _int(nflav), _int(neta), _int(npres), _int(ntemp),
            _int(lo['minor_limits_gpt'].shape[0]), _int(lo['kminor'].shape[0]),
            _int(up['minor_limits_gpt'].shape[0]), _int(up['kminor'].shape[0]),
            _int(kdist['idx_h2o']),
            i32(kdist['gpoint_flavor']), i32(kdist['band_lims_gpt']),
            kdist['kmajor'], lo['kminor'], up['kminor'],
            i32(lo['minor_limits_gpt']), i32(up['minor_limits_gpt']),
            i32(lo['scales_with_density'], 0), i32(up['scales_with_density'], 0),
            i32(lo['scale_by_complement'], 0), i32(up['scale_by_complement'], 0),
            i32(lo['idx_minor'], 0), i32(up['idx_minor'], 0),
            i32(lo['idx_minor_scaling'], 0), i32(up['idx_minor_scaling'], 0),
            i32(lo['kminor_start']), i32(up['kminor_start']),
            interp['tropo'],
            interp['col_mix'], interp['fmajor'], interp['fminor'],
            play, tlay, col_gas,
            interp['jeta'], interp['jtemp'], interp['jpress'],
            tau)

        return tau

    def compute_tau_rayleigh(self, kdist, interp, col_dry, col_gas, neta):
        nlay, ncol = col_dry.shape
        ngpt = kdist['krayl'].shape[1]
        nbnd = kdist['band_lims_gpt'].shape[0]
        ngas = col_gas.shape[0] - 1
        nflav = kdist['flavor'].shape[0]
        ntemp = kdist['krayl'].shape[3]

        tau = np.zeros((ngpt, nlay, ncol), dtype=FLOAT)

        def i32(a, base=1):
            return np.ascontiguousarray((a + base).astype(np.int32))

        self.lib.rrtmgp_compute_tau_rayleigh(
            _int(ncol), _int(nlay), _int(nbnd), _int(ngpt),
            # npres is declared but unused by this kernel: krayl has no pressure axis.
            _int(ngas), _int(nflav), _int(neta), _int(0), _int(ntemp),
            i32(kdist['gpoint_flavor']), i32(kdist['band_lims_gpt']),
            kdist['krayl'], _int(kdist['idx_h2o']),
            col_dry, col_gas, interp['fminor'],
            interp['jeta'], interp['tropo'], interp['jtemp'],
            tau)

        return tau

    def compute_planck_source(self, kdist, interp, tlay, tlev, tsfc, sfc_lay, neta):
        """sfc_lay is 0-based here, as in rte3d; the reference wants it 1-based."""
        nlay, ncol = tlay.shape
        ngpt = kdist['pfracin'].shape[0]
        nbnd = kdist['band_lims_gpt'].shape[0]
        nflav = kdist['flavor'].shape[0]
        ntemp = kdist['pfracin'].shape[3]
        npres = kdist['pfracin'].shape[1] - 1
        nplancktemp = kdist['totplnk'].shape[1]

        sfc_src = np.zeros((ngpt, ncol), dtype=FLOAT)
        lay_src = np.zeros((ngpt, nlay, ncol), dtype=FLOAT)
        lev_src = np.zeros((ngpt, nlay + 1, ncol), dtype=FLOAT)
        sfc_jac = np.zeros((ngpt, ncol), dtype=FLOAT)

        def i32(a, base=1):
            return np.ascontiguousarray((a + base).astype(np.int32))

        self.lib.rrtmgp_compute_Planck_source(
            _int(ncol), _int(nlay), _int(nbnd), _int(ngpt),
            _int(nflav), _int(neta), _int(npres), _int(ntemp), _int(nplancktemp),
            tlay, tlev, tsfc, _int(sfc_lay + 1),
            interp['fmajor'], interp['jeta'], interp['tropo'],
            interp['jtemp'], interp['jpress'],
            i32(kdist['gpt_band']), i32(kdist['band_lims_gpt']),
            kdist['pfracin'],
            _dbl(kdist['temp_ref_min']), _dbl(kdist['totplnk_delta']),
            kdist['totplnk'], i32(kdist['gpoint_flavor']),
            sfc_src, lay_src, lev_src, sfc_jac)

        return dict(lay_source=lay_src, lev_source=lev_src,
                    sfc_source=sfc_src, sfc_source_jac=sfc_jac)
