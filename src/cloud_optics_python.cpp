#include <pybind11/stl.h>

#include "cloud_optics.h"
#include "runtime.h"


void Clouds::init_python_bindings(py::module_& m)
{
    py::class_<Cloud_optics>(m, "Cloud_optics",
            "Cloud optical property lookup tables for one ice roughness type.")
        .def_property_readonly("nroughness_types",
            [](const Cloud_optics& c) { return c.nroughness_types; })
        .def_property_readonly("size_bounds",
            [](const Cloud_optics& c)
            {
                return py::make_tuple(c.radliq_lwr, c.radliq_upr, c.radice_lwr, c.radice_upr);
            },
            "(liquid lower, liquid upper, ice lower, ice upper) in microns.");

    m.def("load_cloud_optics",
        [](const py::dict& f, const int icergh)
        {
            Runtime::get();

            const auto get = [&](const char* key)
            {
                if (!f.contains(key))
                    throw std::invalid_argument(
                            std::string("Cloud coefficient file is missing '") + key + "'");
                return f[key].cast<Numpy::In<TF>>();
            };

            return Cloud_optics::load_lut(
                    f["radliq_lwr"].cast<TF>(), f["radliq_upr"].cast<TF>(),
                    f["radice_lwr"].cast<TF>(), f["radice_upr"].cast<TF>(),
                    Numpy::to_host_2d<TF>(get("extliq"), "extliq"),
                    Numpy::to_host_2d<TF>(get("ssaliq"), "ssaliq"),
                    Numpy::to_host_2d<TF>(get("asyliq"), "asyliq"),
                    Numpy::to_host_3d<TF>(get("extice"), "extice"),
                    Numpy::to_host_3d<TF>(get("ssaice"), "ssaice"),
                    Numpy::to_host_3d<TF>(get("asyice"), "asyice"),
                    icergh);
        },
        py::arg("file"), py::arg("icergh") = 0,
        "Build the cloud optics tables. icergh selects the ice roughness type, 0-based; "
        "the reference defaults to the smoothest.");

    m.def("cloud_optics",
        [](const Cloud_optics& c,
           const Numpy::In<TF>& clwp, const Numpy::In<TF>& ciwp,
           const Numpy::In<TF>& reliq, const Numpy::In<TF>& reice,
           const bool two_stream) -> py::object
        {
            Runtime::get();

            auto clwp_d = Numpy::to_device_2d<TF>(clwp, "clwp");
            auto ciwp_d = Numpy::to_device_2d<TF>(ciwp, "ciwp");
            auto reliq_d = Numpy::to_device_2d<TF>(reliq, "reliq");
            auto reice_d = Numpy::to_device_2d<TF>(reice, "reice");

            const int nspec = static_cast<int>(c.lut_extliq.extent(0));
            const int nlay = static_cast<int>(clwp_d.extent(0));
            const int ncol = static_cast<int>(clwp_d.extent(1));

            const auto no_init = Kokkos::WithoutInitializing;
            Array_3d<TF> tau(Kokkos::view_alloc("tau", no_init), nspec, nlay, ncol);

            if (!two_stream)
            {
                Clouds::compute(c, clwp_d, ciwp_d, reliq_d, reice_d, tau);
                Kokkos::fence();
                return Numpy::from_device(tau);
            }

            Array_3d<TF> ssa(Kokkos::view_alloc("ssa", no_init), nspec, nlay, ncol);
            Array_3d<TF> g(Kokkos::view_alloc("g", no_init), nspec, nlay, ncol);

            Clouds::compute(c, clwp_d, ciwp_d, reliq_d, reice_d, tau, ssa, g);
            Kokkos::fence();

            return py::make_tuple(Numpy::from_device(tau), Numpy::from_device(ssa),
                                  Numpy::from_device(g));
        },
        py::arg("cloud_optics"), py::arg("clwp"), py::arg("ciwp"),
        py::arg("reliq"), py::arg("reice"), py::arg("two_stream") = true,
        "Cloud optical properties. With two_stream, returns (tau, ssa, g); otherwise "
        "the absorption optical depth alone, which is what a no-scattering longwave "
        "calculation takes.");
}
