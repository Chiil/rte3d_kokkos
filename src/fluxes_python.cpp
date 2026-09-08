#include <optional>

#include <pybind11/stl.h>

#include "fluxes.h"
#include "runtime.h"


void Fluxes::init_python_bindings(py::module_& m)
{
    m.def("sum_broadband",
        [](const Numpy::In<TF>& spectral_flux)
        {
            Runtime::get();
            auto in_d = Numpy::to_device_3d<TF>(spectral_flux, "spectral_flux");

            Array_2d<TF> out(Kokkos::view_alloc("broadband_flux", Kokkos::WithoutInitializing),
                             in_d.extent(1), in_d.extent(2));

            Fluxes::sum_broadband(in_d, out);
            Kokkos::fence();

            return Numpy::from_device(out);
        },
        py::arg("spectral_flux"),
        "Sum a (ngpt, nlev, ncol) flux over g-points. Returns (nlev, ncol).");

    m.def("net_broadband",
        [](const Numpy::In<TF>& flux_dn, const Numpy::In<TF>& flux_up)
        {
            Runtime::get();

            const int ndim = static_cast<int>(flux_dn.request().ndim);

            if (ndim == 3)
            {
                auto dn_d = Numpy::to_device_3d<TF>(flux_dn, "flux_dn");
                auto up_d = Numpy::to_device_3d<TF>(flux_up, "flux_up");

                Array_2d<TF> out(Kokkos::view_alloc("net", Kokkos::WithoutInitializing),
                                 dn_d.extent(1), dn_d.extent(2));

                Fluxes::net_broadband(dn_d, up_d, out);
                Kokkos::fence();

                return Numpy::from_device(out);
            }

            auto dn_d = Numpy::to_device_2d<TF>(flux_dn, "flux_dn");
            auto up_d = Numpy::to_device_2d<TF>(flux_up, "flux_up");

            Array_2d<TF> out(Kokkos::view_alloc("net", Kokkos::WithoutInitializing),
                             dn_d.extent(0), dn_d.extent(1));

            Fluxes::net_broadband(dn_d, up_d, out);
            Kokkos::fence();

            return Numpy::from_device(out);
        },
        py::arg("flux_dn"), py::arg("flux_up"),
        "Net (down minus up) flux, summed over g-points. Accepts spectrally-resolved "
        "(ngpt, nlev, ncol) inputs or already-integrated (nlev, ncol) ones. "
        "Returns (nlev, ncol).");

    m.def("sum_byband",
        [](const Numpy::In<int>& band_lims, const Numpy::In<TF>& spectral_flux)
        {
            Runtime::get();
            auto lims_d = Numpy::to_device_2d<int>(band_lims, "band_lims");
            auto in_d = Numpy::to_device_3d<TF>(spectral_flux, "spectral_flux");

            Array_3d<TF> out(Kokkos::view_alloc("byband_flux", Kokkos::WithoutInitializing),
                             lims_d.extent(0), in_d.extent(1), in_d.extent(2));

            Fluxes::sum_byband(lims_d, in_d, out);
            Kokkos::fence();

            return Numpy::from_device(out);
        },
        py::arg("band_lims"), py::arg("spectral_flux"),
        "Sum a flux within each band. band_lims is (nbnd, 2), 0-based and inclusive. "
        "Returns (nbnd, nlev, ncol).");

    m.def("net_byband",
        [](const Numpy::In<int>& band_lims,
           const Numpy::In<TF>& flux_dn, const Numpy::In<TF>& flux_up)
        {
            Runtime::get();
            auto lims_d = Numpy::to_device_2d<int>(band_lims, "band_lims");
            auto dn_d = Numpy::to_device_3d<TF>(flux_dn, "flux_dn");
            auto up_d = Numpy::to_device_3d<TF>(flux_up, "flux_up");

            Array_3d<TF> out(Kokkos::view_alloc("byband_net", Kokkos::WithoutInitializing),
                             lims_d.extent(0), dn_d.extent(1), dn_d.extent(2));

            Fluxes::net_byband(lims_d, dn_d, up_d, out);
            Kokkos::fence();

            return Numpy::from_device(out);
        },
        py::arg("band_lims"), py::arg("flux_dn"), py::arg("flux_up"),
        "Net flux within each band. Returns (nbnd, nlev, ncol).");
}
