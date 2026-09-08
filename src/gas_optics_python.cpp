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

            Interp_state state = Interp_state::create(nflav, nlay, ncol);

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

            // The reference accumulates into tau, so start from zero.
            Array_3d<TF> tau("tau", ngpt, nlay, ncol);

            Gas_optics::compute_tau_absorption(k, state, play_d, tlay_d, col_gas_d, tau);
            Kokkos::fence();

            return Numpy::from_device(tau);
        },
        py::arg("kdist"), py::arg("press_ref_log"), py::arg("temp_ref"),
        py::arg("press_ref_log_delta"), py::arg("temp_ref_min"), py::arg("temp_ref_delta"),
        py::arg("press_ref_trop_log"), py::arg("neta"), py::arg("vmr_ref"),
        py::arg("play"), py::arg("tlay"), py::arg("col_gas"),
        "Interpolate and compute the absorption optical depth. Returns tau "
        "(ngpt, nlay, ncol). All indices in kdist are 0-based.");
}
