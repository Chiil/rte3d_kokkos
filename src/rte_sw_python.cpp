#include <optional>

#include <pybind11/stl.h>

#include "rte_sw.h"
#include "runtime.h"


namespace
{
    // An optional (ngpt, ncol) boundary condition. None becomes an empty View, which
    // the solvers read as "zero".
    Array_2d<TF> optional_2d(const std::optional<Numpy::In<TF>>& a, const std::string& name)
    {
        if (!a.has_value())
            return Array_2d<TF>();

        return Numpy::to_device_2d<TF>(*a, name);
    }

    Array_map_1d<TF> optional_slice(const Array_2d<TF>& a, const int igpt)
    {
        if (a.size() == 0)
            return Array_map_1d<TF>();

        return slice_1d(a, igpt);
    }
}


// The solvers run one g-point at a time; these bindings keep the spectrally resolved
// interface and drive the g-point loop here. See the note in rte_lw_python.cpp.
void Rte_sw::init_python_bindings(py::module_& m)
{
    m.def("sw_solver_noscat",
        [](const bool top_at_1,
           const Numpy::In<TF>& tau,
           const Numpy::In<TF>& mu0,
           const Numpy::In<TF>& inc_flux_dir)
        {
            Runtime::get();

            auto tau_d = Numpy::to_device_3d<TF>(tau, "tau");
            auto mu0_d = Numpy::to_device_2d<TF>(mu0, "mu0");
            auto inc_d = Numpy::to_device_2d<TF>(inc_flux_dir, "inc_flux_dir");

            const int ngpt = static_cast<int>(tau_d.extent(0));
            const int nlay = static_cast<int>(tau_d.extent(1));
            const int ncol = static_cast<int>(tau_d.extent(2));

            Array_3d<TF> flux_dir(
                    Kokkos::view_alloc("flux_dir", Kokkos::WithoutInitializing), ngpt, nlay+1, ncol);

            for (int igpt=0; igpt<ngpt; ++igpt)
                Rte_sw::solver_noscat(
                        top_at_1, slice_2d(tau_d, igpt), mu0_d,
                        slice_1d(inc_d, igpt), slice_2d(flux_dir, igpt));
            Kokkos::fence();

            return Numpy::from_device(flux_dir);
        },
        py::arg("top_at_1"), py::arg("tau"), py::arg("mu0"), py::arg("inc_flux_dir"),
        "Direct-beam shortwave solver without scattering. Returns flux_dir (ngpt, nlev, ncol).");

    m.def("sw_solver_2stream",
        [](const bool top_at_1,
           const Numpy::In<TF>& tau,
           const Numpy::In<TF>& ssa,
           const Numpy::In<TF>& g,
           const Numpy::In<TF>& mu0,
           const Numpy::In<TF>& sfc_alb_dir,
           const Numpy::In<TF>& sfc_alb_dif,
           const Numpy::In<TF>& inc_flux_dir,
           const std::optional<Numpy::In<TF>>& inc_flux_dif)
        {
            Runtime::get();

            auto tau_d = Numpy::to_device_3d<TF>(tau, "tau");
            auto ssa_d = Numpy::to_device_3d<TF>(ssa, "ssa");
            auto g_d = Numpy::to_device_3d<TF>(g, "g");
            auto mu0_d = Numpy::to_device_2d<TF>(mu0, "mu0");
            auto alb_dir_d = Numpy::to_device_2d<TF>(sfc_alb_dir, "sfc_alb_dir");
            auto alb_dif_d = Numpy::to_device_2d<TF>(sfc_alb_dif, "sfc_alb_dif");
            auto inc_dir_d = Numpy::to_device_2d<TF>(inc_flux_dir, "inc_flux_dir");
            auto inc_dif_d = optional_2d(inc_flux_dif, "inc_flux_dif");

            const int ngpt = static_cast<int>(tau_d.extent(0));
            const int nlay = static_cast<int>(tau_d.extent(1));
            const int ncol = static_cast<int>(tau_d.extent(2));

            Array_3d<TF> flux_up(
                    Kokkos::view_alloc("flux_up", Kokkos::WithoutInitializing), ngpt, nlay+1, ncol);
            Array_3d<TF> flux_dn(
                    Kokkos::view_alloc("flux_dn", Kokkos::WithoutInitializing), ngpt, nlay+1, ncol);
            Array_3d<TF> flux_dir(
                    Kokkos::view_alloc("flux_dir", Kokkos::WithoutInitializing), ngpt, nlay+1, ncol);

            // Built once: the solver's scratch does not depend on the g-point, and
            // allocating it per call would be a cudaMalloc/cudaFree pair per g-point.
            const auto scratch = Rte_sw::Two_stream_scratch::make(nlay, ncol);

            for (int igpt=0; igpt<ngpt; ++igpt)
                Rte_sw::solver_2stream(
                        top_at_1,
                        slice_2d(tau_d, igpt), slice_2d(ssa_d, igpt), slice_2d(g_d, igpt), mu0_d,
                        slice_1d(alb_dir_d, igpt), slice_1d(alb_dif_d, igpt),
                        slice_1d(inc_dir_d, igpt), optional_slice(inc_dif_d, igpt),
                        slice_2d(flux_up, igpt), slice_2d(flux_dn, igpt), slice_2d(flux_dir, igpt),
                        scratch);
            Kokkos::fence();

            return py::make_tuple(
                    Numpy::from_device(flux_up),
                    Numpy::from_device(flux_dn),
                    Numpy::from_device(flux_dir));
        },
        py::arg("top_at_1"), py::arg("tau"), py::arg("ssa"), py::arg("g"), py::arg("mu0"),
        py::arg("sfc_alb_dir"), py::arg("sfc_alb_dif"), py::arg("inc_flux_dir"),
        py::arg("inc_flux_dif") = py::none(),
        "Two-stream shortwave solver. Returns (flux_up, flux_dn, flux_dir), each "
        "(ngpt, nlev, ncol). flux_dn is the total downward flux, diffuse plus direct.");
}
