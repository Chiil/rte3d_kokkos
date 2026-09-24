#include <optional>

#include <pybind11/stl.h>

#include "cloud_optics.h"
#include "optical_props.h"
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

    // The clouds as the caller gives them: water paths and particle sizes, uploaded and
    // with their optical properties allocated, but not computed. That is left to
    // cloud_props, so that it can run inside the timed region, as rte-rrtmgp-cpp's
    // cloud optics does inside its own.
    struct Cloud_input
    {
        const Cloud_optics* optics = nullptr;
        Array_2d<TF> clwp, ciwp, reliq, reice;
        Array_3d<TF> tau, ssa, g;   // (nbnd, nlay, ncol); ssa and g only for two_stream
        bool delta_scale = false;
    };

    Cloud_input upload_clouds(
            const Cloud_optics* optics,
            const std::optional<Numpy::In<TF>>& clwp, const std::optional<Numpy::In<TF>>& ciwp,
            const std::optional<Numpy::In<TF>>& reliq, const std::optional<Numpy::In<TF>>& reice,
            const bool two_stream, const bool delta_scale)
    {
        Cloud_input c;
        if (optics == nullptr)
            return c;

        if (!(clwp.has_value() && ciwp.has_value() && reliq.has_value() && reice.has_value()))
            throw std::invalid_argument("cloud_optics needs clwp, ciwp, reliq and reice");

        c.optics = optics;
        c.clwp = Numpy::to_device_2d<TF>(*clwp, "clwp");
        c.ciwp = Numpy::to_device_2d<TF>(*ciwp, "ciwp");
        c.reliq = Numpy::to_device_2d<TF>(*reliq, "reliq");
        c.reice = Numpy::to_device_2d<TF>(*reice, "reice");

        const int nspec = static_cast<int>(optics->lut_extliq.extent(0));
        const int nlay = static_cast<int>(c.clwp.extent(0));
        const int ncol = static_cast<int>(c.clwp.extent(1));

        const auto no_init = Kokkos::WithoutInitializing;
        c.tau = Array_3d<TF>(Kokkos::view_alloc("cloud_tau", no_init), nspec, nlay, ncol);

        if (two_stream)
        {
            c.ssa = Array_3d<TF>(Kokkos::view_alloc("cloud_ssa", no_init), nspec, nlay, ncol);
            c.g = Array_3d<TF>(Kokkos::view_alloc("cloud_g", no_init), nspec, nlay, ncol);
            c.delta_scale = delta_scale;
        }

        return c;
    }

    // The cloud optical properties by band, delta-scaled when asked. Without ssa and g
    // the clouds only absorb, which is what a no-scattering longwave solve takes.
    Solver::Band_props cloud_props(const Cloud_input& c)
    {
        Solver::Band_props props;
        if (c.optics == nullptr)
            return props;

        if (c.ssa.size() == 0)
        {
            Clouds::compute(*c.optics, c.clwp, c.ciwp, c.reliq, c.reice, c.tau);
            props.tau = c.tau;
            return props;
        }

        Clouds::compute(*c.optics, c.clwp, c.ciwp, c.reliq, c.reice, c.tau, c.ssa, c.g);
        if (c.delta_scale)
            Optical_props::delta_scale_2str(Optical_props_2str{c.tau, c.ssa, c.g});

        props.tau = c.tau;
        props.ssa = c.ssa;
        props.g = c.g;
        return props;
    }

    // The duration of solve in seconds, fenced at both ends, with every input already
    // on the device and every output left there. That is the span rte-rrtmgp-cpp's
    // drivers time with CUDA events, so the two can be compared directly.
    template<typename F>
    double timed(F&& solve)
    {
        Kokkos::fence();
        Kokkos::Timer timer;
        solve();
        Kokkos::fence();
        return timer.seconds();
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
           const Cloud_optics* cloud_optics,
           const std::optional<Numpy::In<TF>>& clwp, const std::optional<Numpy::In<TF>>& ciwp,
           const std::optional<Numpy::In<TF>>& reliq, const std::optional<Numpy::In<TF>>& reice,
           const bool delta_cloud,
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

            // Without scattering the clouds only absorb; with it they are two-stream,
            // and delta_cloud then decides whether they are delta-scaled.
            const Cloud_input clouds = upload_clouds(
                    cloud_optics, clwp, ciwp, reliq, reice, scattering, delta_cloud);

            const auto secants_d = Numpy::to_device_2d<TF>(secants, "secants");
            const auto weights_d = Numpy::to_device_1d<TF>(weights, "weights");
            const auto sfc_emis_d = Numpy::to_device_2d<TF>(sfc_emis, "sfc_emis");
            const auto inc_flux_d = optional_2d(inc_flux, "inc_flux");

            const auto fluxes = fluxes_out(nlev, ncol, nbnd, byband, false, jacobian);

            const double solve_time = timed([&]
            {
                Solver::solve_lw(
                        k, gas_concs, atm, top_at_1,
                        secants_d, weights_d, sfc_emis_d, inc_flux_d,
                        cloud_props(clouds), scattering, fluxes);
            });

            py::dict out = fluxes_dict(fluxes);
            out["solve_time"] = solve_time;
            return out;
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("top_at_1"),
        py::arg("play"), py::arg("plev"), py::arg("tlay"), py::arg("tlev"), py::arg("tsfc"),
        py::arg("secants"), py::arg("weights"), py::arg("sfc_emis"),
        py::arg("inc_flux") = py::none(), py::arg("cloud_optics") = py::none(),
        py::arg("clwp") = py::none(), py::arg("ciwp") = py::none(),
        py::arg("reliq") = py::none(), py::arg("reice") = py::none(),
        py::arg("delta_cloud") = true,
        py::arg("col_dry") = py::none(), py::arg("scattering") = false,
        py::arg("byband") = false, py::arg("jacobian") = false,
        "Longwave gas optics, clouds and transport, one g-point at a time. Nothing "
        "allocated here carries a g-point dimension. secants is (nmus, ncol) and "
        "sfc_emis (ngpt, ncol). Clouds, if cloud_optics is given, come from clwp, "
        "ciwp, reliq and reice, (nlay, ncol) each, and cloud_optics must be a -bnd "
        "table. Returns a dict with flux_up and flux_dn, plus flux_up_byband / "
        "flux_dn_byband when byband, flux_up_jac when jacobian, and solve_time. "
        "scattering solves with the two-stream solver instead of the quadrature, "
        "and the clouds then scatter too, delta-scaled if delta_cloud; it has no "
        "Jacobian. "
        "solve_time is the device time of the clouds, gas optics and transport, in "
        "seconds, without the copies to and from numpy.");

    m.def("solve_sw",
        [](const Kdist_gas& k, const Gas_concs& gas_concs, const bool top_at_1,
           const Numpy::In<TF>& play, const Numpy::In<TF>& plev, const Numpy::In<TF>& tlay,
           const Numpy::In<TF>& mu0,
           const Numpy::In<TF>& sfc_alb_dir, const Numpy::In<TF>& sfc_alb_dif,
           const Numpy::In<TF>& inc_flux_dir,
           const std::optional<Numpy::In<TF>>& inc_flux_dif,
           const Cloud_optics* cloud_optics,
           const std::optional<Numpy::In<TF>>& clwp, const std::optional<Numpy::In<TF>>& ciwp,
           const std::optional<Numpy::In<TF>>& reliq, const std::optional<Numpy::In<TF>>& reice,
           const bool delta_cloud,
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

            const Cloud_input clouds = upload_clouds(
                    cloud_optics, clwp, ciwp, reliq, reice, true, delta_cloud);

            const auto mu0_d = Numpy::to_device_2d<TF>(mu0, "mu0");
            const auto sfc_alb_dir_d = Numpy::to_device_2d<TF>(sfc_alb_dir, "sfc_alb_dir");
            const auto sfc_alb_dif_d = Numpy::to_device_2d<TF>(sfc_alb_dif, "sfc_alb_dif");
            const auto inc_flux_dir_d =
                    Numpy::to_device_2d<TF>(inc_flux_dir, "inc_flux_dir");
            const auto inc_flux_dif_d = optional_2d(inc_flux_dif, "inc_flux_dif");

            const auto fluxes = fluxes_out(nlev, ncol, nbnd, byband, true, false);

            const double solve_time = timed([&]
            {
                Solver::solve_sw(
                        k, gas_concs, atm, top_at_1,
                        mu0_d, sfc_alb_dir_d, sfc_alb_dif_d, inc_flux_dir_d, inc_flux_dif_d,
                        cloud_props(clouds), fluxes);
            });

            py::dict out = fluxes_dict(fluxes);
            out["solve_time"] = solve_time;
            return out;
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("top_at_1"),
        py::arg("play"), py::arg("plev"), py::arg("tlay"), py::arg("mu0"),
        py::arg("sfc_alb_dir"), py::arg("sfc_alb_dif"), py::arg("inc_flux_dir"),
        py::arg("inc_flux_dif") = py::none(),
        py::arg("cloud_optics") = py::none(),
        py::arg("clwp") = py::none(), py::arg("ciwp") = py::none(),
        py::arg("reliq") = py::none(), py::arg("reice") = py::none(),
        py::arg("delta_cloud") = true,
        py::arg("col_dry") = py::none(),
        py::arg("byband") = false,
        "Shortwave gas optics, clouds and transport, one g-point at a time. mu0 is "
        "(nlay, ncol); the boundary conditions are (ngpt, ncol). Clouds come from "
        "clwp, ciwp, reliq and reice as for solve_lw, delta-scaled if delta_cloud. "
        "Returns a dict with flux_up, flux_dn and flux_dir, plus the by-band totals "
        "when byband, and solve_time. "
        "solve_time is the device time of the clouds, gas optics and transport, in "
        "seconds, without the copies to and from numpy.");


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
           const Cloud_optics* cloud_optics,
           const std::optional<Numpy::In<TF>>& clwp, const std::optional<Numpy::In<TF>>& ciwp,
           const std::optional<Numpy::In<TF>>& reliq, const std::optional<Numpy::In<TF>>& reice,
           const bool delta_cloud,
           const std::optional<Numpy::In<TF>>& col_dry,
           const bool scattering, const bool lump_above,
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

            const Cloud_input clouds = upload_clouds(
                    cloud_optics, clwp, ciwp, reliq, reice, scattering, delta_cloud);

            const auto sfc_emis_d = Numpy::to_device_2d<TF>(sfc_emis, "sfc_emis");
            const auto secants_d = Numpy::to_device_2d<TF>(secants, "secants");
            const auto weights_d = Numpy::to_device_1d<TF>(weights, "weights");

            const auto fluxes = Raytracer_lw::Fluxes_lw::make(grid);

            int traced = 0;
            const double solve_time = timed([&]
            {
                traced = Solver::solve_lw_rt(
                        k, gas_concs, atm, top_at_1, grid,
                        photons_per_pixel, independent_column,
                        sfc_emis_d, secants_d, weights_d,
                        min_mfp_grid_ratio, cloud_props(clouds), scattering, lump_above,
                        fluxes);
            });

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
            out["solve_time"] = solve_time;

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
        py::arg("cloud_optics") = py::none(),
        py::arg("clwp") = py::none(), py::arg("ciwp") = py::none(),
        py::arg("reliq") = py::none(), py::arg("reice") = py::none(),
        py::arg("delta_cloud") = true,
        py::arg("col_dry") = py::none(),
        py::arg("scattering") = false, py::arg("lump_above") = true,
        py::arg("kn_x") = 0, py::arg("kn_y") = 0, py::arg("kn_z") = 0,
        "Longwave gas optics, Planck sources, clouds and the Monte Carlo ray tracer, "
        "one g-point at a time. The columns are the tracer's horizontal grid, "
        "ncol = nx*ny with the column index i + j*nx, and layers from nz-1 upward are "
        "lumped into the top cell, their emission included. sfc_emis is (ngpt, ncol). "
        "Clouds come from clwp, ciwp, reliq and reice as for solve_lw; without "
        "scattering they only absorb. "
        "Returns a dict of the surface and top-of-domain fluxes, (ncol) each, and "
        "rt_lw_flux_abs, the absorbed minus emitted flux per unit height, "
        "(nz, ncol) -- the names are rte-rrtmgp-cpp's own -- plus n_gpt_traced, how "
        "many g-points were actually traced, and solve_time. min_mfp_grid_ratio skips "
        "the g-points whose gas is opaque within a grid cell and solves those "
        "plane-parallel instead, which is what secants and weights are for; zero "
        "traces everything. "
        "lump_above puts the atmosphere above the box into the box\'s top cell; "
        "without it nz must count only the resolved cells and the air above enters as "
        "the downward flux a plane-parallel solve of the full column leaves there. "
        "solve_time is the device time of the clouds, gas optics and transport, in "
        "seconds, without the copies to and from numpy.");


    m.def("solve_sw_rt",
        [](const Kdist_gas& k, const Gas_concs& gas_concs, const bool top_at_1,
           const Numpy::In<TF>& play, const Numpy::In<TF>& plev, const Numpy::In<TF>& tlay,
           const TF mu0, const TF azi,
           const Numpy::In<TF>& sfc_alb_dir, const Numpy::In<TF>& toa_src,
           const int nx, const int ny, const int nz,
           const TF dx, const TF dy, const TF dz,
           const int photons_per_pixel, const bool independent_column,
           const Cloud_optics* cloud_optics,
           const std::optional<Numpy::In<TF>>& clwp, const std::optional<Numpy::In<TF>>& ciwp,
           const std::optional<Numpy::In<TF>>& reliq, const std::optional<Numpy::In<TF>>& reice,
           const bool delta_cloud,
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

            const Cloud_input clouds = upload_clouds(
                    cloud_optics, clwp, ciwp, reliq, reice, true, delta_cloud);

            const auto toa_src_h = Numpy::to_host_1d<TF>(toa_src, "toa_src");
            const auto sfc_alb_dir_d = Numpy::to_device_2d<TF>(sfc_alb_dir, "sfc_alb_dir");

            const auto fluxes = Raytracer::Fluxes_rt::make(grid);

            const double solve_time = timed([&]
            {
                Solver::solve_sw_rt(
                        k, gas_concs, atm, top_at_1, grid,
                        photons_per_pixel, independent_column, mu0, azi,
                        toa_src_h, sfc_alb_dir_d, cloud_props(clouds), fluxes);
            });

            py::dict out;
            out["rt_flux_tod_dn"] = Numpy::from_device(fluxes.tod_dn);
            out["rt_flux_tod_up"] = Numpy::from_device(fluxes.tod_up);
            out["rt_flux_sfc_dir"] = Numpy::from_device(fluxes.sfc_dir);
            out["rt_flux_sfc_dif"] = Numpy::from_device(fluxes.sfc_dif);
            out["rt_flux_sfc_up"] = Numpy::from_device(fluxes.sfc_up);
            out["rt_flux_abs_dir"] = Numpy::from_device(fluxes.abs_dir);
            out["rt_flux_abs_dif"] = Numpy::from_device(fluxes.abs_dif);
            out["solve_time"] = solve_time;

            return out;
        },
        py::arg("kdist"), py::arg("gas_concs"), py::arg("top_at_1"),
        py::arg("play"), py::arg("plev"), py::arg("tlay"),
        py::arg("mu0"), py::arg("azi"),
        py::arg("sfc_alb_dir"), py::arg("toa_src"),
        py::arg("nx"), py::arg("ny"), py::arg("nz"),
        py::arg("dx"), py::arg("dy"), py::arg("dz"),
        py::arg("photons_per_pixel") = 256, py::arg("independent_column") = false,
        py::arg("cloud_optics") = py::none(),
        py::arg("clwp") = py::none(), py::arg("ciwp") = py::none(),
        py::arg("reliq") = py::none(), py::arg("reice") = py::none(),
        py::arg("delta_cloud") = true,
        py::arg("col_dry") = py::none(),
        py::arg("kn_x") = 0, py::arg("kn_y") = 0, py::arg("kn_z") = 0,
        "Shortwave gas optics, clouds and the Monte Carlo ray tracer, one g-point at a "
        "time. The columns are the tracer's horizontal grid, ncol = nx*ny with the "
        "column index i + j*nx, and layers from nz-1 upward are lumped into the top "
        "cell. The sun is one direction for the whole domain; toa_src is (ngpt) and "
        "sfc_alb_dir (ngpt, ncol). Returns a dict of the surface and top-of-domain "
        "fluxes, (ncol) each, the absorbed flux per unit height, (nz, ncol), and "
        "solve_time. Clouds come from clwp, ciwp, reliq and reice as for solve_sw. "
        "solve_time is the device time of the clouds, gas optics and transport, in "
        "seconds, without the copies to and from numpy.");
}
