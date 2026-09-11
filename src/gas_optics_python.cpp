#include <optional>

#include <pybind11/stl.h>

#include "gas_optics.h"
#include "runtime.h"


namespace
{
    template<typename T>
    Numpy::In<T> item(const py::dict& d, const char* key)
    {
        if (!d.contains(key))
            throw std::invalid_argument(std::string("k-distribution is missing '") + key + "'");

        return d[key].cast<Numpy::In<T>>();
    }

    Minor_absorbers minor_from_dict(const py::dict& d, const char* prefix)
    {
        const auto name = [&](const char* suffix) { return std::string(prefix) + suffix; };

        Minor_absorbers m;
        m.kminor = Numpy::to_device_3d<TF>(item<TF>(d, name("kminor").c_str()), "kminor");
        m.minor_limits_gpt = Numpy::to_device_2d<int>(
                item<int>(d, name("minor_limits_gpt").c_str()), "minor_limits_gpt");
        m.scales_with_density = Numpy::to_device_1d<Bool>(
                item<Bool>(d, name("scales_with_density").c_str()), "scales_with_density");
        m.scale_by_complement = Numpy::to_device_1d<Bool>(
                item<Bool>(d, name("scale_by_complement").c_str()), "scale_by_complement");
        m.idx_minor = Numpy::to_device_1d<int>(item<int>(d, name("idx_minor").c_str()), "idx_minor");
        m.idx_minor_scaling = Numpy::to_device_1d<int>(
                item<int>(d, name("idx_minor_scaling").c_str()), "idx_minor_scaling");
        m.kminor_start = Numpy::to_device_1d<int>(
                item<int>(d, name("kminor_start").c_str()), "kminor_start");

        return m;
    }
}


void Gas_optics::init_python_bindings(py::module_& m)
{
    m.def("interpolation",
        [](const Numpy::In<int>& flavor,
           const Numpy::In<TF>& press_ref_log,
           const Numpy::In<TF>& temp_ref,
           const TF press_ref_log_delta,
           const TF temp_ref_min,
           const TF temp_ref_delta,
           const TF press_ref_trop_log,
           const int neta,
           const Numpy::In<TF>& vmr_ref,
           const Numpy::In<TF>& play,
           const Numpy::In<TF>& tlay,
           const Numpy::In<TF>& col_gas) -> py::dict
        {
            Runtime::get();

            auto flavor_d = Numpy::to_device_2d<int>(flavor, "flavor");
            auto play_d = Numpy::to_device_2d<TF>(play, "play");

            const int nflav = static_cast<int>(flavor_d.extent(0));
            const int nlay = static_cast<int>(play_d.extent(0));
            const int ncol = static_cast<int>(play_d.extent(1));

            // store_eta: this entry point exists to be compared against the
            // reference, which returns jeta, feta and col_mix.
            Interp_state state = Interp_state::create(nflav, nlay, ncol, true);

            Gas_optics::interpolation(
                    flavor_d,
                    Numpy::to_device_1d<TF>(press_ref_log, "press_ref_log"),
                    Numpy::to_device_1d<TF>(temp_ref, "temp_ref"),
                    press_ref_log_delta, temp_ref_min, temp_ref_delta, press_ref_trop_log,
                    neta,
                    Numpy::to_device_3d<TF>(vmr_ref, "vmr_ref"),
                    play_d,
                    Numpy::to_device_2d<TF>(tlay, "tlay"),
                    Numpy::to_device_3d<TF>(col_gas, "col_gas"),
                    state);

            // Test support: the reference returns fmajor and fminor, which we
            // reconstruct on demand rather than store.
            Array_5d<TF> fminor(Kokkos::view_alloc("fminor", Kokkos::WithoutInitializing),
                                nflav, 2, 2, nlay, ncol);
            Array_6d<TF> fmajor(Kokkos::view_alloc("fmajor", Kokkos::WithoutInitializing),
                                nflav, 2, 2, 2, nlay, ncol);
            Gas_optics::expand_weights(state, fminor, fmajor);
            Kokkos::fence();

            py::dict out;
            out["jtemp"] = Numpy::from_device(state.jtemp);
            out["ftemp"] = Numpy::from_device(state.ftemp);
            out["jpress"] = Numpy::from_device(state.jpress);
            out["fpress"] = Numpy::from_device(state.fpress);
            out["tropo"] = Numpy::from_device(state.tropo);
            out["jeta"] = Numpy::from_device(state.jeta);
            out["feta"] = Numpy::from_device(state.feta);
            out["col_mix"] = Numpy::from_device(state.col_mix);
            out["fminor"] = Numpy::from_device(fminor);
            out["fmajor"] = Numpy::from_device(fmajor);

            return out;
        },
        py::arg("flavor"), py::arg("press_ref_log"), py::arg("temp_ref"),
        py::arg("press_ref_log_delta"), py::arg("temp_ref_min"), py::arg("temp_ref_delta"),
        py::arg("press_ref_trop_log"), py::arg("neta"), py::arg("vmr_ref"),
        py::arg("play"), py::arg("tlay"), py::arg("col_gas"),
        "Locate each (layer, column) in the k-distribution grids. Returns a dict of the "
        "interpolation state. jtemp, jpress and jeta are 0-based, unlike the reference. "
        "fmajor and fminor are materialised for testing only.");

    m.def("compute_tau_absorption",
        [](const py::dict& kdist,
           const Numpy::In<TF>& press_ref_log,
           const Numpy::In<TF>& temp_ref,
           const TF press_ref_log_delta,
           const TF temp_ref_min,
           const TF temp_ref_delta,
           const TF press_ref_trop_log,
           const int neta,
           const Numpy::In<TF>& vmr_ref,
           const Numpy::In<TF>& play,
           const Numpy::In<TF>& tlay,
           const Numpy::In<TF>& col_gas)
        {
            Runtime::get();

            Kdist_gas k;
            k.gpoint_flavor = Numpy::to_device_2d<int>(item<int>(kdist, "gpoint_flavor"), "gpoint_flavor");
            k.band_lims_gpt = Numpy::to_device_2d<int>(item<int>(kdist, "band_lims_gpt"), "band_lims_gpt");
            k.kmajor = Numpy::to_device_4d<TF>(item<TF>(kdist, "kmajor"), "kmajor");
            k.gpt_band = Numpy::to_device_1d<int>(item<int>(kdist, "gpt_band"), "gpt_band");
            k.band_gpt_start = Numpy::to_device_1d<int>(item<int>(kdist, "band_gpt_start"), "band_gpt_start");
            k.idx_h2o = kdist["idx_h2o"].cast<int>();

            k.lower = minor_from_dict(kdist, "lower_");
            k.upper = minor_from_dict(kdist, "upper_");

            const int ngpt = static_cast<int>(k.kmajor.extent(0));
            k.lower.build_map(k.gpoint_flavor, ngpt, 0);
            k.upper.build_map(k.gpoint_flavor, ngpt, 1);

            auto flavor_d = Numpy::to_device_2d<int>(item<int>(kdist, "flavor"), "flavor");
            auto play_d = Numpy::to_device_2d<TF>(play, "play");
            auto tlay_d = Numpy::to_device_2d<TF>(tlay, "tlay");
            auto col_gas_d = Numpy::to_device_3d<TF>(col_gas, "col_gas");

            // The kernels rebuild the binary-species interpolation themselves.
            k.flavor = flavor_d;
            k.vmr_ref = Numpy::to_device_3d<TF>(vmr_ref, "vmr_ref");
            k.neta = neta;

            const int nflav = static_cast<int>(flavor_d.extent(0));
            const int nlay = static_cast<int>(play_d.extent(0));
            const int ncol = static_cast<int>(play_d.extent(1));

            Interp_state state = Interp_state::create(nflav, nlay, ncol);

            Gas_optics::interpolation(
                    flavor_d,
                    Numpy::to_device_1d<TF>(press_ref_log, "press_ref_log"),
                    Numpy::to_device_1d<TF>(temp_ref, "temp_ref"),
                    press_ref_log_delta, temp_ref_min, temp_ref_delta, press_ref_trop_log,
                    neta,
                    Numpy::to_device_3d<TF>(vmr_ref, "vmr_ref"),
                    play_d, tlay_d, col_gas_d, state);

            // Each g-point writes its own slice, so nothing here needs zeroing.
            Array_3d<TF> tau(Kokkos::view_alloc("tau", Kokkos::WithoutInitializing), ngpt, nlay, ncol);

            for (int igpt=0; igpt<ngpt; ++igpt)
                Gas_optics::compute_tau_absorption(
                        k, state, play_d, tlay_d, col_gas_d, igpt, slice_2d(tau, igpt));
            Kokkos::fence();

            return Numpy::from_device(tau);
        },
        py::arg("kdist"), py::arg("press_ref_log"), py::arg("temp_ref"),
        py::arg("press_ref_log_delta"), py::arg("temp_ref_min"), py::arg("temp_ref_delta"),
        py::arg("press_ref_trop_log"), py::arg("neta"), py::arg("vmr_ref"),
        py::arg("play"), py::arg("tlay"), py::arg("col_gas"),
        "Interpolate and compute the absorption optical depth. Returns tau "
        "(ngpt, nlay, ncol). All indices in kdist are 0-based.");

    m.def("compute_tau_rayleigh",
        [](const py::dict& kdist,
           const Numpy::In<TF>& press_ref_log, const Numpy::In<TF>& temp_ref,
           const TF press_ref_log_delta, const TF temp_ref_min, const TF temp_ref_delta,
           const TF press_ref_trop_log, const int neta,
           const Numpy::In<TF>& vmr_ref,
           const Numpy::In<TF>& play, const Numpy::In<TF>& tlay,
           const Numpy::In<TF>& col_dry, const Numpy::In<TF>& col_gas)
        {
            Runtime::get();

            Kdist_gas k;
            k.gpoint_flavor = Numpy::to_device_2d<int>(item<int>(kdist, "gpoint_flavor"), "gpoint_flavor");
            k.gpt_band = Numpy::to_device_1d<int>(item<int>(kdist, "gpt_band"), "gpt_band");
            k.band_gpt_start = Numpy::to_device_1d<int>(item<int>(kdist, "band_gpt_start"), "band_gpt_start");
            k.krayl = Numpy::to_device_4d<TF>(item<TF>(kdist, "krayl"), "krayl");
            k.idx_h2o = kdist["idx_h2o"].cast<int>();

            auto flavor_d = Numpy::to_device_2d<int>(item<int>(kdist, "flavor"), "flavor");
            auto play_d = Numpy::to_device_2d<TF>(play, "play");
            auto col_gas_d = Numpy::to_device_3d<TF>(col_gas, "col_gas");

            k.flavor = flavor_d;
            k.vmr_ref = Numpy::to_device_3d<TF>(vmr_ref, "vmr_ref");
            k.neta = neta;

            const int nflav = static_cast<int>(flavor_d.extent(0));
            const int nlay = static_cast<int>(play_d.extent(0));
            const int ncol = static_cast<int>(play_d.extent(1));
            const int ngpt = static_cast<int>(k.krayl.extent(1));

            Interp_state state = Interp_state::create(nflav, nlay, ncol);
            Gas_optics::interpolation(
                    flavor_d,
                    Numpy::to_device_1d<TF>(press_ref_log, "press_ref_log"),
                    Numpy::to_device_1d<TF>(temp_ref, "temp_ref"),
                    press_ref_log_delta, temp_ref_min, temp_ref_delta, press_ref_trop_log,
                    neta, Numpy::to_device_3d<TF>(vmr_ref, "vmr_ref"),
                    play_d, Numpy::to_device_2d<TF>(tlay, "tlay"), col_gas_d, state);

            Array_3d<TF> tau(Kokkos::view_alloc("tau_rayleigh", Kokkos::WithoutInitializing),
                             ngpt, nlay, ncol);

            const auto col_dry_d = Numpy::to_device_2d<TF>(col_dry, "col_dry");

            for (int igpt=0; igpt<ngpt; ++igpt)
                Gas_optics::compute_tau_rayleigh(
                        k, state, col_dry_d, col_gas_d, igpt, slice_2d(tau, igpt));
            Kokkos::fence();

            return Numpy::from_device(tau);
        },
        py::arg("kdist"), py::arg("press_ref_log"), py::arg("temp_ref"),
        py::arg("press_ref_log_delta"), py::arg("temp_ref_min"), py::arg("temp_ref_delta"),
        py::arg("press_ref_trop_log"), py::arg("neta"), py::arg("vmr_ref"),
        py::arg("play"), py::arg("tlay"), py::arg("col_dry"), py::arg("col_gas"),
        "Interpolate and compute the Rayleigh scattering optical depth. "
        "Returns tau_rayleigh (ngpt, nlay, ncol).");

    m.def("compute_planck_source",
        [](const py::dict& kdist,
           const Numpy::In<TF>& press_ref_log, const Numpy::In<TF>& temp_ref,
           const TF press_ref_log_delta, const TF temp_ref_min, const TF temp_ref_delta,
           const TF press_ref_trop_log, const int neta,
           const Numpy::In<TF>& vmr_ref,
           const Numpy::In<TF>& play, const Numpy::In<TF>& tlay,
           const Numpy::In<TF>& tlev, const Numpy::In<TF>& tsfc,
           const int sfc_lay,
           const Numpy::In<TF>& col_gas) -> py::dict
        {
            Runtime::get();

            Kdist_gas k;
            k.gpoint_flavor = Numpy::to_device_2d<int>(item<int>(kdist, "gpoint_flavor"), "gpoint_flavor");
            k.gpt_band = Numpy::to_device_1d<int>(item<int>(kdist, "gpt_band"), "gpt_band");
            k.band_gpt_start = Numpy::to_device_1d<int>(item<int>(kdist, "band_gpt_start"), "band_gpt_start");
            k.pfracin = Numpy::to_device_4d<TF>(item<TF>(kdist, "pfracin"), "pfracin");
            k.totplnk = Numpy::to_device_2d<TF>(item<TF>(kdist, "totplnk"), "totplnk");
            k.totplnk_delta = kdist["totplnk_delta"].cast<TF>();
            k.temp_ref_min = temp_ref_min;

            auto flavor_d = Numpy::to_device_2d<int>(item<int>(kdist, "flavor"), "flavor");
            auto play_d = Numpy::to_device_2d<TF>(play, "play");
            auto col_gas_d = Numpy::to_device_3d<TF>(col_gas, "col_gas");

            k.flavor = flavor_d;
            k.vmr_ref = Numpy::to_device_3d<TF>(vmr_ref, "vmr_ref");
            k.neta = neta;

            const int nflav = static_cast<int>(flavor_d.extent(0));
            const int nlay = static_cast<int>(play_d.extent(0));
            const int ncol = static_cast<int>(play_d.extent(1));
            const int ngpt = static_cast<int>(k.pfracin.extent(0));

            Interp_state state = Interp_state::create(nflav, nlay, ncol);
            Gas_optics::interpolation(
                    flavor_d,
                    Numpy::to_device_1d<TF>(press_ref_log, "press_ref_log"),
                    Numpy::to_device_1d<TF>(temp_ref, "temp_ref"),
                    press_ref_log_delta, temp_ref_min, temp_ref_delta, press_ref_trop_log,
                    neta, Numpy::to_device_3d<TF>(vmr_ref, "vmr_ref"),
                    play_d, Numpy::to_device_2d<TF>(tlay, "tlay"), col_gas_d, state);

            const auto sources = Source_func_lw_spectral::create(ngpt, nlay, ncol, true);

            const auto tlay_d = Numpy::to_device_2d<TF>(tlay, "tlay");
            const auto tlev_d = Numpy::to_device_2d<TF>(tlev, "tlev");
            const auto tsfc_d = Numpy::to_device_1d<TF>(tsfc, "tsfc");

            Array_2d<TF> pfrac(
                    Kokkos::view_alloc("pfrac", Kokkos::WithoutInitializing),
                    tlay_d.extent(0), tlay_d.extent(1));

            for (int igpt=0; igpt<ngpt; ++igpt)
            {
                // The solve path takes pfrac from compute_tau_lw; this entry point
                // tests the Planck sources alone, so it computes it on its own.
                Gas_optics::compute_pfrac(k, state, col_gas_d, igpt, pfrac);
                Gas_optics::compute_planck_source(
                        k, tlay_d, tlev_d, tsfc_d, sfc_lay, igpt, sources.gpt(igpt), pfrac);
            }
            Kokkos::fence();

            py::dict out;
            out["lay_source"] = Numpy::from_device(sources.lay_source);
            out["lev_source"] = Numpy::from_device(sources.lev_source);
            out["sfc_source"] = Numpy::from_device(sources.sfc_source);
            out["sfc_source_jac"] = Numpy::from_device(sources.sfc_source_jac);

            return out;
        },
        py::arg("kdist"), py::arg("press_ref_log"), py::arg("temp_ref"),
        py::arg("press_ref_log_delta"), py::arg("temp_ref_min"), py::arg("temp_ref_delta"),
        py::arg("press_ref_trop_log"), py::arg("neta"), py::arg("vmr_ref"),
        py::arg("play"), py::arg("tlay"), py::arg("tlev"), py::arg("tsfc"),
        py::arg("sfc_lay"), py::arg("col_gas"),
        "Interpolate and compute the Planck sources. sfc_lay is 0-based. Returns a "
        "dict with lay_source, lev_source, sfc_source and sfc_source_jac, in the "
        "layouts Source_func_lw_spectral uses.");

    m.def("gas_optics_lw",
        [](const Kdist_gas& k, const Gas_concs& gas_concs,
           const Numpy::In<TF>& play, const Numpy::In<TF>& plev,
           const Numpy::In<TF>& tlay, const Numpy::In<TF>& tlev,
           const Numpy::In<TF>& tsfc,
           const std::optional<Numpy::In<TF>>& col_dry) -> py::dict
        {
            Runtime::get();

            auto play_d = Numpy::to_device_2d<TF>(play, "play");
            const int nlay = static_cast<int>(play_d.extent(0));
            const int ncol = static_cast<int>(play_d.extent(1));
            const int ngpt = static_cast<int>(k.kmajor.extent(0));

            const auto no_init = Kokkos::WithoutInitializing;
            Array_3d<TF> tau(Kokkos::view_alloc("tau", no_init), ngpt, nlay, ncol);

            const auto sources = Source_func_lw_spectral::create(ngpt, nlay, ncol, true);

            Gas_optics::gas_optics_lw(
                    k, gas_concs, play_d,
                    Numpy::to_device_2d<TF>(plev, "plev"),
                    Numpy::to_device_2d<TF>(tlay, "tlay"),
                    Numpy::to_device_2d<TF>(tlev, "tlev"),
                    Numpy::to_device_1d<TF>(tsfc, "tsfc"),
                    col_dry.has_value() ? Numpy::to_device_2d<TF>(*col_dry, "col_dry")
                                        : Array_2d<TF>(),
                    tau, sources);
            Kokkos::fence();

            py::dict out;
            out["tau"] = Numpy::from_device(tau);
            out["lay_source"] = Numpy::from_device(sources.lay_source);
            out["lev_source"] = Numpy::from_device(sources.lev_source);
            out["sfc_source"] = Numpy::from_device(sources.sfc_source);
            out["sfc_source_jac"] = Numpy::from_device(sources.sfc_source_jac);

            return out;
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("play"), py::arg("plev"),
        py::arg("tlay"), py::arg("tlev"), py::arg("tsfc"), py::arg("col_dry") = py::none(),
        "Longwave gas optics. Returns tau and the Planck sources, in the layouts the "
        "longwave solvers take.");

    m.def("gas_optics_sw",
        [](const Kdist_gas& k, const Gas_concs& gas_concs,
           const Numpy::In<TF>& play, const Numpy::In<TF>& plev, const Numpy::In<TF>& tlay,
           const std::optional<Numpy::In<TF>>& col_dry) -> py::tuple
        {
            Runtime::get();

            auto play_d = Numpy::to_device_2d<TF>(play, "play");
            const int nlay = static_cast<int>(play_d.extent(0));
            const int ncol = static_cast<int>(play_d.extent(1));
            const int ngpt = static_cast<int>(k.kmajor.extent(0));

            const auto no_init = Kokkos::WithoutInitializing;
            Array_3d<TF> tau(Kokkos::view_alloc("tau", no_init), ngpt, nlay, ncol);
            Array_3d<TF> ssa(Kokkos::view_alloc("ssa", no_init), ngpt, nlay, ncol);

            Gas_optics::gas_optics_sw(
                    k, gas_concs, play_d,
                    Numpy::to_device_2d<TF>(plev, "plev"),
                    Numpy::to_device_2d<TF>(tlay, "tlay"),
                    col_dry.has_value() ? Numpy::to_device_2d<TF>(*col_dry, "col_dry")
                                        : Array_2d<TF>(),
                    tau, ssa);
            Kokkos::fence();

            return py::make_tuple(Numpy::from_device(tau), Numpy::from_device(ssa));
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("play"), py::arg("plev"),
        py::arg("tlay"), py::arg("col_dry") = py::none(),
        "Shortwave gas optics. Returns (tau, ssa), where tau is the total extinction "
        "and ssa the Rayleigh fraction of it.");
}
