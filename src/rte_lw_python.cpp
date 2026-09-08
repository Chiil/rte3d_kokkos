#include <optional>

#include <pybind11/stl.h>

#include "rte_lw.h"
#include "runtime.h"


// The solvers run one g-point at a time. These bindings keep the spectrally resolved
// interface the tests and the reference kernels use, and drive the g-point loop here:
// slicing an (ngpt, ...) array is free, since LayoutRight makes each g-point's block
// contiguous.
void Rte_lw::init_python_bindings(py::module_& m)
{
    m.def("lw_solver_noscat",
        [](const bool top_at_1,
           const Numpy::In<TF>& secants,
           const Numpy::In<TF>& weights,
           const Numpy::In<TF>& tau,
           const Numpy::In<TF>& lay_source,
           const Numpy::In<TF>& lev_source,
           const Numpy::In<TF>& sfc_emis,
           const Numpy::In<TF>& sfc_source,
           const Numpy::In<TF>& inc_flux,
           const std::optional<Numpy::In<TF>>& sfc_source_jac) -> py::tuple
        {
            Runtime::get();

            const bool do_jacobians = sfc_source_jac.has_value();

            auto secants_d = Numpy::to_device_3d<TF>(secants, "secants");
            auto weights_d = Numpy::to_device_1d<TF>(weights, "weights");
            auto tau_d = Numpy::to_device_3d<TF>(tau, "tau");
            auto sfc_emis_d = Numpy::to_device_2d<TF>(sfc_emis, "sfc_emis");
            auto inc_flux_d = Numpy::to_device_2d<TF>(inc_flux, "inc_flux");

            Source_func_lw_spectral sources;
            sources.lay_source = Numpy::to_device_3d<TF>(lay_source, "lay_source");
            sources.lev_source = Numpy::to_device_3d<TF>(lev_source, "lev_source");
            sources.sfc_source = Numpy::to_device_2d<TF>(sfc_source, "sfc_source");
            sources.sfc_source_jac = do_jacobians
                    ? Numpy::to_device_2d<TF>(*sfc_source_jac, "sfc_source_jac")
                    : Array_2d<TF>();

            const int ngpt = static_cast<int>(tau_d.extent(0));
            const int nlay = static_cast<int>(tau_d.extent(1));
            const int ncol = static_cast<int>(tau_d.extent(2));
            const int nmus = static_cast<int>(weights_d.extent(0));

            // The secants arrive as (nmus, ngpt, ncol), matching the reference; the
            // solver wants one g-point's (nmus, ncol) contiguous, so reorder once.
            Array_3d<TF> secants_g(
                    Kokkos::view_alloc("secants_g", Kokkos::WithoutInitializing), ngpt, nmus, ncol);
            parallel_for_3d("reorder_secants", {0, 0, 0}, {ngpt, nmus, ncol},
                KOKKOS_LAMBDA(const int igpt, const int imu, const int icol)
                {
                    secants_g(igpt, imu, icol) = secants_d(imu, igpt, icol);
                });

            Array_3d<TF> flux_up(
                    Kokkos::view_alloc("flux_up", Kokkos::WithoutInitializing), ngpt, nlay+1, ncol);
            Array_3d<TF> flux_dn(
                    Kokkos::view_alloc("flux_dn", Kokkos::WithoutInitializing), ngpt, nlay+1, ncol);

            // Spectrally integrated, so the solver accumulates into it across g-points.
            Array_2d<TF> flux_up_jac(
                    "flux_up_jac", do_jacobians ? nlay+1 : 0, do_jacobians ? ncol : 0);

            for (int igpt=0; igpt<ngpt; ++igpt)
                Rte_lw::solver_noscat(
                        top_at_1,
                        slice_2d(secants_g, igpt), weights_d, slice_2d(tau_d, igpt),
                        sources.gpt(igpt),
                        slice_1d(sfc_emis_d, igpt), slice_1d(inc_flux_d, igpt),
                        slice_2d(flux_up, igpt), slice_2d(flux_dn, igpt),
                        flux_up_jac);
            Kokkos::fence();

            if (!do_jacobians)
                return py::make_tuple(Numpy::from_device(flux_up), Numpy::from_device(flux_dn), py::none());

            return py::make_tuple(
                    Numpy::from_device(flux_up),
                    Numpy::from_device(flux_dn),
                    Numpy::from_device(flux_up_jac));
        },
        py::arg("top_at_1"), py::arg("secants"), py::arg("weights"), py::arg("tau"),
        py::arg("lay_source"), py::arg("lev_source"), py::arg("sfc_emis"),
        py::arg("sfc_source"), py::arg("inc_flux"),
        py::arg("sfc_source_jac") = py::none(),
        "Longwave no-scattering solver with multi-angle quadrature. Returns "
        "(flux_up, flux_dn, flux_up_jac); the Jacobian is spectrally integrated, so "
        "(nlev, ncol), and is None unless sfc_source_jac is given.");

    m.def("lw_solver_2stream",
        [](const bool top_at_1,
           const Numpy::In<TF>& tau,
           const Numpy::In<TF>& ssa,
           const Numpy::In<TF>& g,
           const Numpy::In<TF>& lay_source,
           const Numpy::In<TF>& lev_source,
           const Numpy::In<TF>& sfc_emis,
           const Numpy::In<TF>& sfc_source,
           const Numpy::In<TF>& inc_flux) -> py::tuple
        {
            Runtime::get();

            auto tau_d = Numpy::to_device_3d<TF>(tau, "tau");
            auto ssa_d = Numpy::to_device_3d<TF>(ssa, "ssa");
            auto g_d = Numpy::to_device_3d<TF>(g, "g");
            auto sfc_emis_d = Numpy::to_device_2d<TF>(sfc_emis, "sfc_emis");
            auto inc_flux_d = Numpy::to_device_2d<TF>(inc_flux, "inc_flux");

            Source_func_lw_spectral sources;
            sources.lay_source = Numpy::to_device_3d<TF>(lay_source, "lay_source");
            sources.lev_source = Numpy::to_device_3d<TF>(lev_source, "lev_source");
            sources.sfc_source = Numpy::to_device_2d<TF>(sfc_source, "sfc_source");

            const int ngpt = static_cast<int>(tau_d.extent(0));
            const int nlay = static_cast<int>(tau_d.extent(1));
            const int ncol = static_cast<int>(tau_d.extent(2));

            Array_3d<TF> flux_up(
                    Kokkos::view_alloc("flux_up", Kokkos::WithoutInitializing), ngpt, nlay+1, ncol);
            Array_3d<TF> flux_dn(
                    Kokkos::view_alloc("flux_dn", Kokkos::WithoutInitializing), ngpt, nlay+1, ncol);

            for (int igpt=0; igpt<ngpt; ++igpt)
                Rte_lw::solver_2stream(
                        top_at_1,
                        slice_2d(tau_d, igpt), slice_2d(ssa_d, igpt), slice_2d(g_d, igpt),
                        sources.gpt(igpt),
                        slice_1d(sfc_emis_d, igpt), slice_1d(inc_flux_d, igpt),
                        slice_2d(flux_up, igpt), slice_2d(flux_dn, igpt));
            Kokkos::fence();

            return py::make_tuple(Numpy::from_device(flux_up), Numpy::from_device(flux_dn));
        },
        py::arg("top_at_1"), py::arg("tau"), py::arg("ssa"), py::arg("g"),
        py::arg("lay_source"), py::arg("lev_source"), py::arg("sfc_emis"),
        py::arg("sfc_source"), py::arg("inc_flux"),
        "Two-stream longwave solver. Returns (flux_up, flux_dn), each (ngpt, nlev, ncol).");
}
