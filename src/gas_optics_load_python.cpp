#include <optional>

#include <pybind11/stl.h>

#include "gas_optics.h"
#include "raytracer.h"
#include "runtime.h"


namespace
{
    template<typename T>
    Numpy::In<T> get(const py::dict& d, const char* key)
    {
        if (!d.contains(key))
            throw std::invalid_argument(std::string("Coefficient file is missing '") + key + "'");

        return d[key].cast<Numpy::In<T>>();
    }

    std::vector<std::string> get_names(const py::dict& d, const char* key)
    {
        if (!d.contains(key))
            throw std::invalid_argument(std::string("Coefficient file is missing '") + key + "'");

        return d[key].cast<std::vector<std::string>>();
    }
}


namespace
{
    py::dict kdist_arrays(const Kdist_gas& k);
}


void Gas_optics::init_load_python_bindings(py::module_& m)
{
    py::class_<Kdist_gas>(m, "Kdist_gas",
            "A k-distribution reduced to one set of gases. Opaque; use arrays() to "
            "inspect the reduced index arrays.")
        .def_property_readonly("gas_names",
            [](const Kdist_gas& k) { return k.gas_names; },
            "The gases this was reduced to, in the order col_gas uses. Entry 0 of that "
            "dimension is dry air, so gas_names[i] sits at col_gas index i+1.")
        .def_property_readonly("ngpt",
            [](const Kdist_gas& k) { return static_cast<int>(k.kmajor.extent(0)); })
        .def_property_readonly("nbnd",
            [](const Kdist_gas& k) { return static_cast<int>(k.band_lims_gpt.extent(0)); })
        .def_property_readonly("neta", [](const Kdist_gas& k) { return k.neta; })
        .def_property_readonly("idx_h2o", [](const Kdist_gas& k) { return k.idx_h2o; })
        .def_property_readonly("solar_source",
            [](const Kdist_gas& k) -> py::object
            {
                if (k.solar_source.size() == 0)
                    return py::none();
                return Numpy::from_device(k.solar_source);
            },
            "Spectral solar source at the top of the atmosphere, (ngpt). None for a "
            "longwave k-distribution. Scale is arbitrary; renormalise to your own "
            "total solar irradiance.")
        .def("arrays", [](const Kdist_gas& k) { return kdist_arrays(k); },
            "Every reduced array, 0-based. For testing against the reference.");

    m.def("load_kdist",
        [](const py::dict& f, const Gas_concs& gas_concs) -> Kdist_gas
        {
            Runtime::get();

            Kdist_file file;
            file.gas_names = get_names(f, "gas_names");
            file.gas_minor = get_names(f, "gas_minor");
            file.identifier_minor = get_names(f, "identifier_minor");
            file.minor_gases_lower = get_names(f, "minor_gases_lower");
            file.minor_gases_upper = get_names(f, "minor_gases_upper");
            file.scaling_gas_lower = get_names(f, "scaling_gas_lower");
            file.scaling_gas_upper = get_names(f, "scaling_gas_upper");

            file.key_species = Numpy::to_host_3d<int>(get<int>(f, "key_species"), "key_species");
            file.band2gpt = Numpy::to_host_2d<int>(get<int>(f, "band2gpt"), "band2gpt");
            file.press_ref = Numpy::to_host_1d<TF>(get<TF>(f, "press_ref"), "press_ref");
            file.temp_ref = Numpy::to_host_1d<TF>(get<TF>(f, "temp_ref"), "temp_ref");
            file.press_ref_trop = f["press_ref_trop"].cast<TF>();
            file.temp_ref_p = f["temp_ref_p"].cast<TF>();
            file.temp_ref_t = f["temp_ref_t"].cast<TF>();
            file.vmr_ref = Numpy::to_host_3d<TF>(get<TF>(f, "vmr_ref"), "vmr_ref");
            file.kmajor = Numpy::to_host_4d<TF>(get<TF>(f, "kmajor"), "kmajor");

            file.kminor_lower = Numpy::to_host_3d<TF>(get<TF>(f, "kminor_lower"), "kminor_lower");
            file.kminor_upper = Numpy::to_host_3d<TF>(get<TF>(f, "kminor_upper"), "kminor_upper");
            file.minor_limits_gpt_lower = Numpy::to_host_2d<int>(
                    get<int>(f, "minor_limits_gpt_lower"), "minor_limits_gpt_lower");
            file.minor_limits_gpt_upper = Numpy::to_host_2d<int>(
                    get<int>(f, "minor_limits_gpt_upper"), "minor_limits_gpt_upper");
            file.minor_scales_with_density_lower = Numpy::to_host_1d<Bool>(
                    get<Bool>(f, "minor_scales_with_density_lower"), "msd_lower");
            file.minor_scales_with_density_upper = Numpy::to_host_1d<Bool>(
                    get<Bool>(f, "minor_scales_with_density_upper"), "msd_upper");
            file.scale_by_complement_lower = Numpy::to_host_1d<Bool>(
                    get<Bool>(f, "scale_by_complement_lower"), "sbc_lower");
            file.scale_by_complement_upper = Numpy::to_host_1d<Bool>(
                    get<Bool>(f, "scale_by_complement_upper"), "sbc_upper");
            file.kminor_start_lower = Numpy::to_host_1d<int>(
                    get<int>(f, "kminor_start_lower"), "kminor_start_lower");
            file.kminor_start_upper = Numpy::to_host_1d<int>(
                    get<int>(f, "kminor_start_upper"), "kminor_start_upper");

            if (f.contains("totplnk"))
            {
                file.totplnk = Numpy::to_host_2d<TF>(get<TF>(f, "totplnk"), "totplnk");
                file.planck_frac = Numpy::to_host_4d<TF>(get<TF>(f, "planck_frac"), "planck_frac");
            }
            if (f.contains("rayl"))
            {
                file.rayl = Numpy::to_host_4d<TF>(get<TF>(f, "rayl"), "rayl");
                file.solar_source_quiet = Numpy::to_host_1d<TF>(
                        get<TF>(f, "solar_source_quiet"), "solar_source_quiet");
                file.solar_source_facular = Numpy::to_host_1d<TF>(
                        get<TF>(f, "solar_source_facular"), "solar_source_facular");
                file.solar_source_sunspot = Numpy::to_host_1d<TF>(
                        get<TF>(f, "solar_source_sunspot"), "solar_source_sunspot");
                file.mg_default = f["mg_default"].cast<TF>();
                file.sb_default = f["sb_default"].cast<TF>();
            }

            return Gas_optics::load(file, gas_concs);
        },
        py::arg("file"), py::arg("gas_concs"),
        "Reduce a coefficient file to the gases in gas_concs and build every derived "
        "index array. Input indices are 1-based, as in the file; the result is "
        "0-based throughout.");
}


namespace
{
    py::dict kdist_arrays(const Kdist_gas& k)
    {
            py::dict out;
            out["gas_names"] = k.gas_names;
            out["flavor"] = Numpy::from_device(k.flavor);
            out["gpoint_flavor"] = Numpy::from_device(k.gpoint_flavor);
            out["vmr_ref"] = Numpy::from_device(k.vmr_ref);
            out["band_lims_gpt"] = Numpy::from_device(k.band_lims_gpt);
            out["gpt_band"] = Numpy::from_device(k.gpt_band);
            out["band_gpt_start"] = Numpy::from_device(k.band_gpt_start);
            out["idx_h2o"] = k.idx_h2o;

            out["press_ref_log"] = Numpy::from_device(k.press_ref_log);
            out["temp_ref"] = Numpy::from_device(k.temp_ref);
            out["press_ref_log_delta"] = k.press_ref_log_delta;
            out["temp_ref_min"] = k.temp_ref_min;
            out["temp_ref_max"] = k.temp_ref_max;
            out["temp_ref_delta"] = k.temp_ref_delta;
            out["press_ref_trop_log"] = k.press_ref_trop_log;
            out["kmajor"] = Numpy::from_device(k.kmajor);

            if (k.totplnk.size() > 0)
            {
                out["totplnk"] = Numpy::from_device(k.totplnk);
                out["pfracin"] = Numpy::from_device(k.pfracin);
                out["totplnk_delta"] = k.totplnk_delta;
            }
            if (k.krayl.size() > 0)
                out["krayl"] = Numpy::from_device(k.krayl);

            for (const auto& [prefix, m_] : {std::pair{"lower_", &k.lower}, std::pair{"upper_", &k.upper}})
            {
                out[(std::string(prefix) + "kminor").c_str()] = Numpy::from_device(m_->kminor);
                out[(std::string(prefix) + "minor_limits_gpt").c_str()] =
                        Numpy::from_device(m_->minor_limits_gpt);
                out[(std::string(prefix) + "scales_with_density").c_str()] =
                        Numpy::from_device(m_->scales_with_density);
                out[(std::string(prefix) + "scale_by_complement").c_str()] =
                        Numpy::from_device(m_->scale_by_complement);
                out[(std::string(prefix) + "idx_minor").c_str()] = Numpy::from_device(m_->idx_minor);
                out[(std::string(prefix) + "idx_minor_scaling").c_str()] =
                        Numpy::from_device(m_->idx_minor_scaling);
                out[(std::string(prefix) + "kminor_start").c_str()] =
                        Numpy::from_device(m_->kminor_start);
            }

            return out;
    }
}


namespace
{
    // An optional array argument. None becomes an empty View, which the solve reads as
    // "absent" throughout.
    Array_2d<TF> optional_2d(const std::optional<Numpy::In<TF>>& a, const std::string& name)
    {
        return a.has_value() ? Numpy::to_device_2d<TF>(*a, name) : Array_2d<TF>();
    }

    Array_3d<TF> optional_3d(const std::optional<Numpy::In<TF>>& a, const std::string& name)
    {
        return a.has_value() ? Numpy::to_device_3d<TF>(*a, name) : Array_3d<TF>();
    }

    Gas_optics::Fluxes_out fluxes_out(
            const int nlev, const int ncol, const int nbnd,
            const bool byband, const bool do_dir, const bool do_jacobian)
    {
        const auto no_init = Kokkos::WithoutInitializing;

        // The broadband totals are accumulated into, so the solve zeroes them; these
        // need no initialisation here.
        Gas_optics::Fluxes_out f;
        f.up = Array_2d<TF>(Kokkos::view_alloc("flux_up", no_init), nlev, ncol);
        f.dn = Array_2d<TF>(Kokkos::view_alloc("flux_dn", no_init), nlev, ncol);

        if (do_dir)
            f.dir = Array_2d<TF>(Kokkos::view_alloc("flux_dir", no_init), nlev, ncol);
        if (do_jacobian)
            f.up_jac = Array_2d<TF>(Kokkos::view_alloc("flux_up_jac", no_init), nlev, ncol);

        if (byband)
        {
            f.up_byband = Array_3d<TF>(Kokkos::view_alloc("flux_up_byband", no_init), nbnd, nlev, ncol);
            f.dn_byband = Array_3d<TF>(Kokkos::view_alloc("flux_dn_byband", no_init), nbnd, nlev, ncol);

            if (do_dir)
                f.dir_byband = Array_3d<TF>(
                        Kokkos::view_alloc("flux_dir_byband", no_init), nbnd, nlev, ncol);
        }

        return f;
    }

    py::dict fluxes_dict(const Gas_optics::Fluxes_out& f)
    {
        py::dict out;
        out["flux_up"] = Numpy::from_device(f.up);
        out["flux_dn"] = Numpy::from_device(f.dn);

        if (f.dir.size() > 0)
            out["flux_dir"] = Numpy::from_device(f.dir);
        if (f.up_jac.size() > 0)
            out["flux_up_jac"] = Numpy::from_device(f.up_jac);

        if (f.up_byband.size() > 0)
        {
            out["flux_up_byband"] = Numpy::from_device(f.up_byband);
            out["flux_dn_byband"] = Numpy::from_device(f.dn_byband);

            if (f.dir_byband.size() > 0)
                out["flux_dir_byband"] = Numpy::from_device(f.dir_byband);
        }

        return out;
    }
}


void Gas_optics::init_frontend_python_bindings(py::module_& m)
{
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

    m.def("solve_lw",
        [](const Kdist_gas& k, const Gas_concs& gas_concs, const bool top_at_1,
           const Numpy::In<TF>& play, const Numpy::In<TF>& plev,
           const Numpy::In<TF>& tlay, const Numpy::In<TF>& tlev,
           const Numpy::In<TF>& tsfc,
           const Numpy::In<TF>& secants, const Numpy::In<TF>& weights,
           const Numpy::In<TF>& sfc_emis,
           const std::optional<Numpy::In<TF>>& inc_flux,
           const std::optional<Numpy::In<TF>>& cloud_tau,
           const std::optional<Numpy::In<TF>>& col_dry,
           const bool byband, const bool jacobian) -> py::dict
        {
            Runtime::get();

            Gas_optics::Atmosphere atm;
            atm.play = Numpy::to_device_2d<TF>(play, "play");
            atm.plev = Numpy::to_device_2d<TF>(plev, "plev");
            atm.tlay = Numpy::to_device_2d<TF>(tlay, "tlay");
            atm.tlev = Numpy::to_device_2d<TF>(tlev, "tlev");
            atm.tsfc = Numpy::to_device_1d<TF>(tsfc, "tsfc");
            atm.col_dry = optional_2d(col_dry, "col_dry");

            const int nlev = static_cast<int>(atm.plev.extent(0));
            const int ncol = static_cast<int>(atm.plev.extent(1));
            const int nbnd = static_cast<int>(k.band_lims_gpt.extent(0));

            Gas_optics::Band_props clouds;
            clouds.tau = optional_3d(cloud_tau, "cloud_tau");

            const auto fluxes = fluxes_out(nlev, ncol, nbnd, byband, false, jacobian);

            Gas_optics::solve_lw(
                    k, gas_concs, atm, top_at_1,
                    Numpy::to_device_2d<TF>(secants, "secants"),
                    Numpy::to_device_1d<TF>(weights, "weights"),
                    Numpy::to_device_2d<TF>(sfc_emis, "sfc_emis"),
                    optional_2d(inc_flux, "inc_flux"),
                    clouds, fluxes);
            Kokkos::fence();

            return fluxes_dict(fluxes);
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("top_at_1"),
        py::arg("play"), py::arg("plev"), py::arg("tlay"), py::arg("tlev"), py::arg("tsfc"),
        py::arg("secants"), py::arg("weights"), py::arg("sfc_emis"),
        py::arg("inc_flux") = py::none(), py::arg("cloud_tau") = py::none(),
        py::arg("col_dry") = py::none(),
        py::arg("byband") = false, py::arg("jacobian") = false,
        "Longwave gas optics, clouds and transport, one g-point at a time. Nothing "
        "allocated here carries a g-point dimension. secants is (nmus, ncol) and "
        "sfc_emis (ngpt, ncol); cloud_tau, if given, is (nbnd, nlay, ncol). Returns a "
        "dict with flux_up and flux_dn, plus flux_up_byband / flux_dn_byband when "
        "byband, and flux_up_jac when jacobian.");

    m.def("solve_sw",
        [](const Kdist_gas& k, const Gas_concs& gas_concs, const bool top_at_1,
           const Numpy::In<TF>& play, const Numpy::In<TF>& plev, const Numpy::In<TF>& tlay,
           const Numpy::In<TF>& mu0,
           const Numpy::In<TF>& sfc_alb_dir, const Numpy::In<TF>& sfc_alb_dif,
           const Numpy::In<TF>& inc_flux_dir,
           const std::optional<Numpy::In<TF>>& inc_flux_dif,
           const std::optional<Numpy::In<TF>>& cloud_tau,
           const std::optional<Numpy::In<TF>>& cloud_ssa,
           const std::optional<Numpy::In<TF>>& cloud_g,
           const std::optional<Numpy::In<TF>>& col_dry,
           const bool byband) -> py::dict
        {
            Runtime::get();

            Gas_optics::Atmosphere atm;
            atm.play = Numpy::to_device_2d<TF>(play, "play");
            atm.plev = Numpy::to_device_2d<TF>(plev, "plev");
            atm.tlay = Numpy::to_device_2d<TF>(tlay, "tlay");
            atm.col_dry = optional_2d(col_dry, "col_dry");

            const int nlev = static_cast<int>(atm.plev.extent(0));
            const int ncol = static_cast<int>(atm.plev.extent(1));
            const int nbnd = static_cast<int>(k.band_lims_gpt.extent(0));

            Gas_optics::Band_props clouds;
            clouds.tau = optional_3d(cloud_tau, "cloud_tau");
            clouds.ssa = optional_3d(cloud_ssa, "cloud_ssa");
            clouds.g = optional_3d(cloud_g, "cloud_g");

            const auto fluxes = fluxes_out(nlev, ncol, nbnd, byband, true, false);

            Gas_optics::solve_sw(
                    k, gas_concs, atm, top_at_1,
                    Numpy::to_device_2d<TF>(mu0, "mu0"),
                    Numpy::to_device_2d<TF>(sfc_alb_dir, "sfc_alb_dir"),
                    Numpy::to_device_2d<TF>(sfc_alb_dif, "sfc_alb_dif"),
                    Numpy::to_device_2d<TF>(inc_flux_dir, "inc_flux_dir"),
                    optional_2d(inc_flux_dif, "inc_flux_dif"),
                    clouds, fluxes);
            Kokkos::fence();

            return fluxes_dict(fluxes);
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("top_at_1"),
        py::arg("play"), py::arg("plev"), py::arg("tlay"), py::arg("mu0"),
        py::arg("sfc_alb_dir"), py::arg("sfc_alb_dif"), py::arg("inc_flux_dir"),
        py::arg("inc_flux_dif") = py::none(),
        py::arg("cloud_tau") = py::none(), py::arg("cloud_ssa") = py::none(),
        py::arg("cloud_g") = py::none(), py::arg("col_dry") = py::none(),
        py::arg("byband") = false,
        "Shortwave gas optics, clouds and transport, one g-point at a time. mu0 is "
        "(nlay, ncol); the boundary conditions are (ngpt, ncol); the cloud properties, "
        "if given, are (nbnd, nlay, ncol) and already delta-scaled. Returns a dict "
        "with flux_up, flux_dn and flux_dir, plus the by-band totals when byband.");


    m.def("solve_sw_rt",
        [](const Kdist_gas& k, const Gas_concs& gas_concs, const bool top_at_1,
           const Numpy::In<TF>& play, const Numpy::In<TF>& plev, const Numpy::In<TF>& tlay,
           const TF mu0, const TF azi,
           const Numpy::In<TF>& sfc_alb_dir, const Numpy::In<TF>& toa_src,
           const int nx, const int ny, const int nz,
           const TF dx, const TF dy, const TF dz,
           const int photons_per_pixel, const bool independent_column,
           const std::optional<Numpy::In<TF>>& cloud_tau,
           const std::optional<Numpy::In<TF>>& cloud_ssa,
           const std::optional<Numpy::In<TF>>& cloud_g,
           const std::optional<Numpy::In<TF>>& col_dry,
           const int kn_x, const int kn_y, const int kn_z) -> py::dict
        {
            Runtime::get();

            Gas_optics::Atmosphere atm;
            atm.play = Numpy::to_device_2d<TF>(play, "play");
            atm.plev = Numpy::to_device_2d<TF>(plev, "plev");
            atm.tlay = Numpy::to_device_2d<TF>(tlay, "tlay");
            atm.col_dry = optional_2d(col_dry, "col_dry");

            const Raytracer::Grid grid =
                    Raytracer::Grid::make(nx, ny, nz, dx, dy, dz, kn_x, kn_y, kn_z);

            const int nlay = static_cast<int>(atm.play.extent(0));
            const int ncol = static_cast<int>(atm.play.extent(1));

            if (ncol != grid.ncol())
                throw std::invalid_argument("The atmosphere has " + std::to_string(ncol)
                        + " columns where the grid has " + std::to_string(grid.ncol()));
            if (nz > nlay)
                throw std::invalid_argument("The ray-tracing grid is deeper than the atmosphere");

            Gas_optics::Band_props clouds;
            clouds.tau = optional_3d(cloud_tau, "cloud_tau");
            clouds.ssa = optional_3d(cloud_ssa, "cloud_ssa");
            clouds.g = optional_3d(cloud_g, "cloud_g");

            const auto fluxes = Raytracer::Fluxes_rt::make(grid);

            Gas_optics::solve_sw_rt(
                    k, gas_concs, atm, top_at_1, grid,
                    photons_per_pixel, independent_column, mu0, azi,
                    Numpy::to_host_1d<TF>(toa_src, "toa_src"),
                    Numpy::to_device_2d<TF>(sfc_alb_dir, "sfc_alb_dir"),
                    clouds, fluxes);
            Kokkos::fence();

            py::dict out;
            out["rt_flux_tod_dn"] = Numpy::from_device(fluxes.tod_dn);
            out["rt_flux_tod_up"] = Numpy::from_device(fluxes.tod_up);
            out["rt_flux_sfc_dir"] = Numpy::from_device(fluxes.sfc_dir);
            out["rt_flux_sfc_dif"] = Numpy::from_device(fluxes.sfc_dif);
            out["rt_flux_sfc_up"] = Numpy::from_device(fluxes.sfc_up);
            out["rt_flux_abs_dir"] = Numpy::from_device(fluxes.abs_dir);
            out["rt_flux_abs_dif"] = Numpy::from_device(fluxes.abs_dif);

            return out;
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("top_at_1"),
        py::arg("play"), py::arg("plev"), py::arg("tlay"),
        py::arg("mu0"), py::arg("azi"),
        py::arg("sfc_alb_dir"), py::arg("toa_src"),
        py::arg("nx"), py::arg("ny"), py::arg("nz"),
        py::arg("dx"), py::arg("dy"), py::arg("dz"),
        py::arg("photons_per_pixel") = 256, py::arg("independent_column") = false,
        py::arg("cloud_tau") = py::none(), py::arg("cloud_ssa") = py::none(),
        py::arg("cloud_g") = py::none(), py::arg("col_dry") = py::none(),
        py::arg("kn_x") = 0, py::arg("kn_y") = 0, py::arg("kn_z") = 0,
        "Shortwave gas optics, clouds and the Monte Carlo ray tracer, one g-point at a "
        "time. The columns are the tracer's horizontal grid, ncol = nx*ny with the "
        "column index i + j*nx, and layers from nz-1 upward are lumped into the top "
        "cell. The sun is one direction for the whole domain; toa_src is (ngpt) and "
        "sfc_alb_dir (ngpt, ncol). Returns a dict of the surface and top-of-domain "
        "fluxes, (ncol) each, and the absorbed flux per unit height, (nz, ncol).");
}
