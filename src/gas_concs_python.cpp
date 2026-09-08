#include <optional>

#include <pybind11/stl.h>

#include "gas_concs.h"
#include "gas_optics.h"
#include "runtime.h"


void Gas_concs::init_python_bindings(py::module_& m)
{
    py::class_<Gas_concs>(m, "Gas_concs",
            "Volume mixing ratios by gas name. Accepts a scalar, a (nlay) profile or a "
            "(nlay, ncol) field. Names are case-insensitive.")
        .def(py::init<>())
        .def("set_vmr",
            [](Gas_concs& g, const std::string& name, const py::object& value)
            {
                Runtime::get();

                if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value))
                {
                    g.set_vmr(name, value.cast<TF>());
                    return;
                }

                const auto a = value.cast<Numpy::In<TF>>();
                const int ndim = static_cast<int>(a.request().ndim);

                if (ndim == 1)
                    g.set_vmr(name, Numpy::to_device_1d<TF>(a, "vmr"));
                else if (ndim == 2)
                    g.set_vmr(name, Numpy::to_device_2d<TF>(a, "vmr"));
                else
                    throw std::invalid_argument(
                            "A gas concentration must be a scalar, a (nlay) profile or a "
                            "(nlay, ncol) field.");
            },
            py::arg("name"), py::arg("value"))
        .def("contains", &Gas_concs::contains, py::arg("name"))
        .def("names", &Gas_concs::names)
        .def("get_vmr",
            [](const Gas_concs& g, const std::string& name, const int nlay, const int ncol)
            {
                Runtime::get();

                Array_2d<TF> out(Kokkos::view_alloc("vmr", Kokkos::WithoutInitializing), nlay, ncol);
                g.get_vmr(name, out);
                Kokkos::fence();

                return Numpy::from_device(out);
            },
            py::arg("name"), py::arg("nlay"), py::arg("ncol"),
            "Expand this gas to a (nlay, ncol) field.");

    m.def("compute_col_dry",
        [](const Numpy::In<TF>& vmr_h2o, const Numpy::In<TF>& plev,
           const std::optional<Numpy::In<TF>>& latitude)
        {
            Runtime::get();

            auto vmr_d = Numpy::to_device_2d<TF>(vmr_h2o, "vmr_h2o");
            Array_2d<TF> col_dry(Kokkos::view_alloc("col_dry", Kokkos::WithoutInitializing),
                                 vmr_d.extent(0), vmr_d.extent(1));

            Gas_optics::compute_col_dry(
                    vmr_d, Numpy::to_device_2d<TF>(plev, "plev"),
                    latitude.has_value() ? Numpy::to_device_1d<TF>(*latitude, "latitude")
                                         : Array_1d<TF>(),
                    col_dry);
            Kokkos::fence();

            return Numpy::from_device(col_dry);
        },
        py::arg("vmr_h2o"), py::arg("plev"), py::arg("latitude") = py::none(),
        "Dry air column amount [molecules/cm2]. Returns (nlay, ncol).");

    m.def("compute_col_gas",
        [](const Gas_concs& gas_concs, const std::vector<std::string>& gas_names,
           const Numpy::In<TF>& plev,
           const std::optional<Numpy::In<TF>>& col_dry,
           const std::optional<Numpy::In<TF>>& latitude)
        {
            Runtime::get();

            auto plev_d = Numpy::to_device_2d<TF>(plev, "plev");
            const int nlay = static_cast<int>(plev_d.extent(0)) - 1;
            const int ncol = static_cast<int>(plev_d.extent(1));

            Array_3d<TF> col_gas(Kokkos::view_alloc("col_gas", Kokkos::WithoutInitializing),
                                 static_cast<int>(gas_names.size()) + 1, nlay, ncol);

            Gas_optics::compute_col_gas(
                    gas_concs, gas_names, plev_d,
                    col_dry.has_value() ? Numpy::to_device_2d<TF>(*col_dry, "col_dry")
                                        : Array_2d<TF>(),
                    latitude.has_value() ? Numpy::to_device_1d<TF>(*latitude, "latitude")
                                         : Array_1d<TF>(),
                    col_gas);
            Kokkos::fence();

            return Numpy::from_device(col_gas);
        },
        py::arg("gas_concs"), py::arg("gas_names"), py::arg("plev"),
        py::arg("col_dry") = py::none(), py::arg("latitude") = py::none(),
        "Column gas amounts in the order given by gas_names, with dry air at index 0. "
        "Returns (ngas+1, nlay, ncol).");
}
