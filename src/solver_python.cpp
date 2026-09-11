#include <optional>

#include <pybind11/stl.h>

#include "raytracer.h"
#include "raytracer_lw.h"
#include "runtime.h"
#include "solver.h"


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

    Solver::Fluxes_out fluxes_out(
            const int nlev, const int ncol, const int nbnd,
            const bool byband, const bool do_dir, const bool do_jacobian)
    {
        const auto no_init = Kokkos::WithoutInitializing;

        // The broadband totals are accumulated into, so the solve zeroes them; these
        // need no initialisation here.
        Solver::Fluxes_out f;
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

    py::dict fluxes_dict(const Solver::Fluxes_out& f)
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


void Solver::init_python_bindings(py::module_& m)
{
    m.def("solve_lw",
        [](const Kdist_gas& k, const Gas_concs& gas_concs, const bool top_at_1,
           const Numpy::In<TF>& play, const Numpy::In<TF>& plev,
           const Numpy::In<TF>& tlay, const Numpy::In<TF>& tlev,
           const Numpy::In<TF>& tsfc,
           const Numpy::In<TF>& secants, const Numpy::In<TF>& weights,
           const Numpy::In<TF>& sfc_emis,
           const std::optional<Numpy::In<TF>>& inc_flux,
           const std::optional<Numpy::In<TF>>& cloud_tau,
           const std::optional<Numpy::In<TF>>& cloud_ssa,
           const std::optional<Numpy::In<TF>>& cloud_g,
           const std::optional<Numpy::In<TF>>& col_dry,
           const bool scattering,
           const bool byband, const bool jacobian) -> py::dict
        {
            Runtime::get();

            Solver::Atmosphere atm;
            atm.play = Numpy::to_device_2d<TF>(play, "play");
            atm.plev = Numpy::to_device_2d<TF>(plev, "plev");
            atm.tlay = Numpy::to_device_2d<TF>(tlay, "tlay");
            atm.tlev = Numpy::to_device_2d<TF>(tlev, "tlev");
            atm.tsfc = Numpy::to_device_1d<TF>(tsfc, "tsfc");
            atm.col_dry = optional_2d(col_dry, "col_dry");

            const int nlev = static_cast<int>(atm.plev.extent(0));
            const int ncol = static_cast<int>(atm.plev.extent(1));
            const int nbnd = static_cast<int>(k.band_lims_gpt.extent(0));

            Solver::Band_props clouds;
            clouds.tau = optional_3d(cloud_tau, "cloud_tau");
            clouds.ssa = optional_3d(cloud_ssa, "cloud_ssa");
            clouds.g = optional_3d(cloud_g, "cloud_g");

            const auto fluxes = fluxes_out(nlev, ncol, nbnd, byband, false, jacobian);

            Solver::solve_lw(
                    k, gas_concs, atm, top_at_1,
                    Numpy::to_device_2d<TF>(secants, "secants"),
                    Numpy::to_device_1d<TF>(weights, "weights"),
                    Numpy::to_device_2d<TF>(sfc_emis, "sfc_emis"),
                    optional_2d(inc_flux, "inc_flux"),
                    clouds, scattering, fluxes);
            Kokkos::fence();

            return fluxes_dict(fluxes);
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("top_at_1"),
        py::arg("play"), py::arg("plev"), py::arg("tlay"), py::arg("tlev"), py::arg("tsfc"),
        py::arg("secants"), py::arg("weights"), py::arg("sfc_emis"),
        py::arg("inc_flux") = py::none(), py::arg("cloud_tau") = py::none(),
        py::arg("cloud_ssa") = py::none(), py::arg("cloud_g") = py::none(),
        py::arg("col_dry") = py::none(), py::arg("scattering") = false,
        py::arg("byband") = false, py::arg("jacobian") = false,
        "Longwave gas optics, clouds and transport, one g-point at a time. Nothing "
        "allocated here carries a g-point dimension. secants is (nmus, ncol) and "
        "sfc_emis (ngpt, ncol); cloud_tau, if given, is (nbnd, nlay, ncol). Returns a "
        "dict with flux_up and flux_dn, plus flux_up_byband / flux_dn_byband when "
        "byband, and flux_up_jac when jacobian. scattering solves with the two-stream "
        "solver instead of the quadrature, and then takes cloud_ssa and cloud_g too; "
        "it has no Jacobian.");

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

            Solver::Atmosphere atm;
            atm.play = Numpy::to_device_2d<TF>(play, "play");
            atm.plev = Numpy::to_device_2d<TF>(plev, "plev");
            atm.tlay = Numpy::to_device_2d<TF>(tlay, "tlay");
            atm.col_dry = optional_2d(col_dry, "col_dry");

            const int nlev = static_cast<int>(atm.plev.extent(0));
            const int ncol = static_cast<int>(atm.plev.extent(1));
            const int nbnd = static_cast<int>(k.band_lims_gpt.extent(0));

            Solver::Band_props clouds;
            clouds.tau = optional_3d(cloud_tau, "cloud_tau");
            clouds.ssa = optional_3d(cloud_ssa, "cloud_ssa");
            clouds.g = optional_3d(cloud_g, "cloud_g");

            const auto fluxes = fluxes_out(nlev, ncol, nbnd, byband, true, false);

            Solver::solve_sw(
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


    m.def("solve_lw_rt",
        [](const Kdist_gas& k, const Gas_concs& gas_concs, const bool top_at_1,
           const Numpy::In<TF>& play, const Numpy::In<TF>& plev,
           const Numpy::In<TF>& tlay, const Numpy::In<TF>& tlev,
           const Numpy::In<TF>& tsfc, const Numpy::In<TF>& sfc_emis,
           const Numpy::In<TF>& secants, const Numpy::In<TF>& weights,
           const TF min_mfp_grid_ratio,
           const int nx, const int ny, const int nz,
           const TF dx, const TF dy, const TF dz,
           const int photons_per_pixel, const bool independent_column,
           const std::optional<Numpy::In<TF>>& cloud_tau,
           const std::optional<Numpy::In<TF>>& cloud_ssa,
           const std::optional<Numpy::In<TF>>& cloud_g,
           const std::optional<Numpy::In<TF>>& col_dry,
           const bool scattering,
           const int kn_x, const int kn_y, const int kn_z) -> py::dict
        {
            Runtime::get();

            Solver::Atmosphere atm;
            atm.play = Numpy::to_device_2d<TF>(play, "play");
            atm.plev = Numpy::to_device_2d<TF>(plev, "plev");
            atm.tlay = Numpy::to_device_2d<TF>(tlay, "tlay");
            atm.tlev = Numpy::to_device_2d<TF>(tlev, "tlev");
            atm.tsfc = Numpy::to_device_1d<TF>(tsfc, "tsfc");
            atm.col_dry = optional_2d(col_dry, "col_dry");

            const Raytracer_lw::Grid grid =
                    Raytracer_lw::Grid::make(nx, ny, nz, dx, dy, dz, kn_x, kn_y, kn_z);

            const int nlay = static_cast<int>(atm.play.extent(0));
            const int ncol = static_cast<int>(atm.play.extent(1));

            if (ncol != grid.ncol())
                throw std::invalid_argument("The atmosphere has " + std::to_string(ncol)
                        + " columns where the grid has " + std::to_string(grid.ncol()));
            if (nz > nlay)
                throw std::invalid_argument("The ray-tracing grid is deeper than the atmosphere");

            Solver::Band_props clouds;
            clouds.tau = optional_3d(cloud_tau, "cloud_tau");
            clouds.ssa = optional_3d(cloud_ssa, "cloud_ssa");
            clouds.g = optional_3d(cloud_g, "cloud_g");

            const auto fluxes = Raytracer_lw::Fluxes_lw::make(grid);

            const int traced = Solver::solve_lw_rt(
                    k, gas_concs, atm, top_at_1, grid,
                    photons_per_pixel, independent_column,
                    Numpy::to_device_2d<TF>(sfc_emis, "sfc_emis"),
                    Numpy::to_device_2d<TF>(secants, "secants"),
                    Numpy::to_device_1d<TF>(weights, "weights"),
                    min_mfp_grid_ratio, clouds, scattering, fluxes);
            Kokkos::fence();

            py::dict out;
            // Named as rte-rrtmgp-cpp names them in its own output, so that a run
            // here and a run of test_rte_rrtmgp_rt can be compared variable for
            // variable. Its "abs" is this net flux, not gross absorption.
            out["rt_lw_flux_toa_dn"] = Numpy::from_device(fluxes.toa_dn);
            out["rt_lw_flux_toa_up"] = Numpy::from_device(fluxes.toa_up);
            out["rt_lw_flux_tod_dn"] = Numpy::from_device(fluxes.tod_dn);
            out["rt_lw_flux_tod_up"] = Numpy::from_device(fluxes.tod_up);
            out["rt_lw_flux_sfc_dn"] = Numpy::from_device(fluxes.sfc_dn);
            out["rt_lw_flux_sfc_up"] = Numpy::from_device(fluxes.sfc_up);
            out["rt_lw_flux_abs"] = Numpy::from_device(fluxes.flux_net);
            out["n_gpt_traced"] = traced;

            return out;
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("top_at_1"),
        py::arg("play"), py::arg("plev"), py::arg("tlay"), py::arg("tlev"),
        py::arg("tsfc"), py::arg("sfc_emis"),
        py::arg("secants"), py::arg("weights"),
        py::arg("min_mfp_grid_ratio") = TF(0.),
        py::arg("nx"), py::arg("ny"), py::arg("nz"),
        py::arg("dx"), py::arg("dy"), py::arg("dz"),
        py::arg("photons_per_pixel") = 256, py::arg("independent_column") = false,
        py::arg("cloud_tau") = py::none(), py::arg("cloud_ssa") = py::none(),
        py::arg("cloud_g") = py::none(), py::arg("col_dry") = py::none(),
        py::arg("scattering") = false,
        py::arg("kn_x") = 0, py::arg("kn_y") = 0, py::arg("kn_z") = 0,
        "Longwave gas optics, Planck sources, clouds and the Monte Carlo ray tracer, "
        "one g-point at a time. The columns are the tracer's horizontal grid, "
        "ncol = nx*ny with the column index i + j*nx, and layers from nz-1 upward are "
        "lumped into the top cell, their emission included. sfc_emis is (ngpt, ncol). "
        "cloud_ssa and cloud_g may be omitted, which is a cloud that only absorbs. "
        "Returns a dict of the surface and top-of-domain fluxes, (ncol) each, and "
        "rt_lw_flux_abs, the absorbed minus emitted flux per unit height, "
        "(nz, ncol) -- the names are rte-rrtmgp-cpp's own -- plus n_gpt_traced, how "
        "many g-points were actually traced. min_mfp_grid_ratio skips the g-points "
        "whose gas is opaque within a grid cell and solves those plane-parallel "
        "instead, which is what secants and weights are for; zero traces everything.");


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

            Solver::Atmosphere atm;
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

            Solver::Band_props clouds;
            clouds.tau = optional_3d(cloud_tau, "cloud_tau");
            clouds.ssa = optional_3d(cloud_ssa, "cloud_ssa");
            clouds.g = optional_3d(cloud_g, "cloud_g");

            const auto fluxes = Raytracer::Fluxes_rt::make(grid);

            Solver::solve_sw_rt(
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
