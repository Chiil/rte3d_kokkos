#include <optional>

#include <pybind11/stl.h>

#include "optical_props.h"
#include "runtime.h"


namespace
{
    // gpt_lims, when given, is (nbnd, 2) holding the 0-based inclusive first and last
    // g-point of each band. Absent, the two sets share a spectral resolution.
    Array_1d<int> make_map(const std::optional<Numpy::In<int>>& gpt_lims, const int ngpt)
    {
        if (!gpt_lims.has_value())
            return Optical_props::identity_map(ngpt);

        return Optical_props::band_map(Numpy::to_device_2d<int>(*gpt_lims, "gpt_lims"), ngpt);
    }
}


void Optical_props::init_python_bindings(py::module_& m)
{
    m.def("delta_scale_2str",
        [](const Numpy::In<TF>& tau, const Numpy::In<TF>& ssa, const Numpy::In<TF>& g,
           const std::optional<Numpy::In<TF>>& f) -> py::tuple
        {
            Runtime::get();

            Optical_props_2str props;
            props.tau = Numpy::to_device_3d<TF>(tau, "tau");
            props.ssa = Numpy::to_device_3d<TF>(ssa, "ssa");
            props.g = Numpy::to_device_3d<TF>(g, "g");

            if (f.has_value())
                Optical_props::delta_scale_2str_f(props, Numpy::to_device_3d<TF>(*f, "f"));
            else
                Optical_props::delta_scale_2str(props);
            Kokkos::fence();

            return py::make_tuple(Numpy::from_device(props.tau), Numpy::from_device(props.ssa),
                                  Numpy::from_device(props.g));
        },
        py::arg("tau"), py::arg("ssa"), py::arg("g"), py::arg("f") = py::none(),
        "Delta-scale two-stream optical properties. Without f, assumes f = g*g. "
        "Returns the scaled (tau, ssa, g).");

    m.def("increment_1scalar_by_1scalar",
        [](const Numpy::In<TF>& tau1, const Numpy::In<TF>& tau2,
           const std::optional<Numpy::In<int>>& gpt_lims)
        {
            Runtime::get();
            auto tau1_d = Numpy::to_device_3d<TF>(tau1, "tau1");
            auto tau2_d = Numpy::to_device_3d<TF>(tau2, "tau2");

            Optical_props::increment_1scalar_by_1scalar(
                    tau1_d, tau2_d, make_map(gpt_lims, static_cast<int>(tau1_d.extent(0))));
            Kokkos::fence();

            return Numpy::from_device(tau1_d);
        },
        py::arg("tau1"), py::arg("tau2"), py::arg("gpt_lims") = py::none(),
        "Increment absorption optical depth by a second value. Returns tau1.");

    m.def("increment_1scalar_by_2stream",
        [](const Numpy::In<TF>& tau1, const Numpy::In<TF>& tau2, const Numpy::In<TF>& ssa2,
           const std::optional<Numpy::In<int>>& gpt_lims)
        {
            Runtime::get();
            auto tau1_d = Numpy::to_device_3d<TF>(tau1, "tau1");
            auto tau2_d = Numpy::to_device_3d<TF>(tau2, "tau2");
            auto ssa2_d = Numpy::to_device_3d<TF>(ssa2, "ssa2");

            Optical_props::increment_1scalar_by_2stream(
                    tau1_d, tau2_d, ssa2_d, make_map(gpt_lims, static_cast<int>(tau1_d.extent(0))));
            Kokkos::fence();

            return Numpy::from_device(tau1_d);
        },
        py::arg("tau1"), py::arg("tau2"), py::arg("ssa2"), py::arg("gpt_lims") = py::none(),
        "Increment absorption optical depth by the absorbing part of a scattering set. "
        "Also covers the reference's increment_1scalar_by_nstream, whose body is identical. "
        "Returns tau1.");

    m.def("increment_2stream_by_1scalar",
        [](const Numpy::In<TF>& tau1, const Numpy::In<TF>& ssa1, const Numpy::In<TF>& tau2,
           const std::optional<Numpy::In<int>>& gpt_lims) -> py::tuple
        {
            Runtime::get();
            auto tau1_d = Numpy::to_device_3d<TF>(tau1, "tau1");
            auto ssa1_d = Numpy::to_device_3d<TF>(ssa1, "ssa1");
            auto tau2_d = Numpy::to_device_3d<TF>(tau2, "tau2");

            Optical_props::increment_2stream_by_1scalar(
                    tau1_d, ssa1_d, tau2_d, make_map(gpt_lims, static_cast<int>(tau1_d.extent(0))));
            Kokkos::fence();

            return py::make_tuple(Numpy::from_device(tau1_d), Numpy::from_device(ssa1_d));
        },
        py::arg("tau1"), py::arg("ssa1"), py::arg("tau2"), py::arg("gpt_lims") = py::none(),
        "Increment two-stream properties by a purely absorbing set. Also covers the "
        "reference's increment_nstream_by_1scalar, whose body is identical: adding a "
        "non-scattering set leaves the phase function untouched. Returns (tau1, ssa1).");

    m.def("increment_2stream_by_2stream",
        [](const Numpy::In<TF>& tau1, const Numpy::In<TF>& ssa1, const Numpy::In<TF>& g1,
           const Numpy::In<TF>& tau2, const Numpy::In<TF>& ssa2, const Numpy::In<TF>& g2,
           const std::optional<Numpy::In<int>>& gpt_lims) -> py::tuple
        {
            Runtime::get();
            auto tau1_d = Numpy::to_device_3d<TF>(tau1, "tau1");
            auto ssa1_d = Numpy::to_device_3d<TF>(ssa1, "ssa1");
            auto g1_d = Numpy::to_device_3d<TF>(g1, "g1");

            Optical_props::increment_2stream_by_2stream(
                    tau1_d, ssa1_d, g1_d,
                    Numpy::to_device_3d<TF>(tau2, "tau2"),
                    Numpy::to_device_3d<TF>(ssa2, "ssa2"),
                    Numpy::to_device_3d<TF>(g2, "g2"),
                    make_map(gpt_lims, static_cast<int>(tau1_d.extent(0))));
            Kokkos::fence();

            return py::make_tuple(Numpy::from_device(tau1_d), Numpy::from_device(ssa1_d),
                                  Numpy::from_device(g1_d));
        },
        py::arg("tau1"), py::arg("ssa1"), py::arg("g1"),
        py::arg("tau2"), py::arg("ssa2"), py::arg("g2"), py::arg("gpt_lims") = py::none(),
        "Increment two-stream properties by another two-stream set. Returns (tau1, ssa1, g1).");

    m.def("extract_subset",
        [](const Numpy::In<TF>& array_in, const int col_start, const int col_end)
        {
            Runtime::get();
            auto in_d = Numpy::to_device_3d<TF>(array_in, "array_in");

            Array_3d<TF> out(Kokkos::view_alloc("array_out", Kokkos::WithoutInitializing),
                             in_d.extent(0), in_d.extent(1), col_end - col_start);

            Optical_props::extract_subset(in_d, col_start, col_end, out);
            Kokkos::fence();

            return Numpy::from_device(out);
        },
        py::arg("array_in"), py::arg("col_start"), py::arg("col_end"),
        "Extract a range of columns. col_start is 0-based, col_end exclusive.");

    m.def("extract_subset_absorption_tau",
        [](const Numpy::In<TF>& tau_in, const Numpy::In<TF>& ssa_in,
           const int col_start, const int col_end)
        {
            Runtime::get();
            auto tau_d = Numpy::to_device_3d<TF>(tau_in, "tau_in");
            auto ssa_d = Numpy::to_device_3d<TF>(ssa_in, "ssa_in");

            Array_3d<TF> out(Kokkos::view_alloc("tau_out", Kokkos::WithoutInitializing),
                             tau_d.extent(0), tau_d.extent(1), col_end - col_start);

            Optical_props::extract_subset_absorption_tau(tau_d, ssa_d, col_start, col_end, out);
            Kokkos::fence();

            return Numpy::from_device(out);
        },
        py::arg("tau_in"), py::arg("ssa_in"), py::arg("col_start"), py::arg("col_end"),
        "Absorption optical thickness tau*(1 - ssa) for a range of columns.");

    m.def("increment_2stream_by_nstream",
        [](const Numpy::In<TF>& tau1, const Numpy::In<TF>& ssa1, const Numpy::In<TF>& g1,
           const Numpy::In<TF>& tau2, const Numpy::In<TF>& ssa2, const Numpy::In<TF>& p2,
           const std::optional<Numpy::In<int>>& gpt_lims) -> py::tuple
        {
            Runtime::get();
            auto tau1_d = Numpy::to_device_3d<TF>(tau1, "tau1");
            auto ssa1_d = Numpy::to_device_3d<TF>(ssa1, "ssa1");
            auto g1_d = Numpy::to_device_3d<TF>(g1, "g1");

            Optical_props::increment_2stream_by_nstream(
                    tau1_d, ssa1_d, g1_d,
                    Numpy::to_device_3d<TF>(tau2, "tau2"),
                    Numpy::to_device_3d<TF>(ssa2, "ssa2"),
                    Numpy::to_device_4d<TF>(p2, "p2"),
                    make_map(gpt_lims, static_cast<int>(tau1_d.extent(0))));
            Kokkos::fence();

            return py::make_tuple(Numpy::from_device(tau1_d), Numpy::from_device(ssa1_d),
                                  Numpy::from_device(g1_d));
        },
        py::arg("tau1"), py::arg("ssa1"), py::arg("g1"),
        py::arg("tau2"), py::arg("ssa2"), py::arg("p2"), py::arg("gpt_lims") = py::none(),
        "Increment two-stream properties by an n-stream set; p2 is (ngpt, nlay, ncol, nmom). "
        "Returns (tau1, ssa1, g1).");

    m.def("increment_nstream_by_2stream",
        [](const Numpy::In<TF>& tau1, const Numpy::In<TF>& ssa1, const Numpy::In<TF>& p1,
           const Numpy::In<TF>& tau2, const Numpy::In<TF>& ssa2, const Numpy::In<TF>& g2,
           const std::optional<Numpy::In<int>>& gpt_lims) -> py::tuple
        {
            Runtime::get();
            auto tau1_d = Numpy::to_device_3d<TF>(tau1, "tau1");
            auto ssa1_d = Numpy::to_device_3d<TF>(ssa1, "ssa1");
            auto p1_d = Numpy::to_device_4d<TF>(p1, "p1");

            Optical_props::increment_nstream_by_2stream(
                    tau1_d, ssa1_d, p1_d,
                    Numpy::to_device_3d<TF>(tau2, "tau2"),
                    Numpy::to_device_3d<TF>(ssa2, "ssa2"),
                    Numpy::to_device_3d<TF>(g2, "g2"),
                    make_map(gpt_lims, static_cast<int>(tau1_d.extent(0))));
            Kokkos::fence();

            return py::make_tuple(Numpy::from_device(tau1_d), Numpy::from_device(ssa1_d),
                                  Numpy::from_device(p1_d));
        },
        py::arg("tau1"), py::arg("ssa1"), py::arg("p1"),
        py::arg("tau2"), py::arg("ssa2"), py::arg("g2"), py::arg("gpt_lims") = py::none(),
        "Increment n-stream properties by a two-stream set, whose phase function "
        "expands to moments g, g^2, g^3, ... Returns (tau1, ssa1, p1).");

    m.def("increment_nstream_by_nstream",
        [](const Numpy::In<TF>& tau1, const Numpy::In<TF>& ssa1, const Numpy::In<TF>& p1,
           const Numpy::In<TF>& tau2, const Numpy::In<TF>& ssa2, const Numpy::In<TF>& p2,
           const std::optional<Numpy::In<int>>& gpt_lims) -> py::tuple
        {
            Runtime::get();
            auto tau1_d = Numpy::to_device_3d<TF>(tau1, "tau1");
            auto ssa1_d = Numpy::to_device_3d<TF>(ssa1, "ssa1");
            auto p1_d = Numpy::to_device_4d<TF>(p1, "p1");

            Optical_props::increment_nstream_by_nstream(
                    tau1_d, ssa1_d, p1_d,
                    Numpy::to_device_3d<TF>(tau2, "tau2"),
                    Numpy::to_device_3d<TF>(ssa2, "ssa2"),
                    Numpy::to_device_4d<TF>(p2, "p2"),
                    make_map(gpt_lims, static_cast<int>(tau1_d.extent(0))));
            Kokkos::fence();

            return py::make_tuple(Numpy::from_device(tau1_d), Numpy::from_device(ssa1_d),
                                  Numpy::from_device(p1_d));
        },
        py::arg("tau1"), py::arg("ssa1"), py::arg("p1"),
        py::arg("tau2"), py::arg("ssa2"), py::arg("p2"), py::arg("gpt_lims") = py::none(),
        "Increment n-stream properties by another n-stream set. Moments beyond the "
        "shorter expansion are left untouched. Returns (tau1, ssa1, p1).");

    m.def("extract_subset_4d",
        [](const Numpy::In<TF>& array_in, const int col_start, const int col_end)
        {
            Runtime::get();
            auto in_d = Numpy::to_device_4d<TF>(array_in, "array_in");

            Array_4d<TF> out(Kokkos::view_alloc("array_out", Kokkos::WithoutInitializing),
                             in_d.extent(0), in_d.extent(1), col_end - col_start, in_d.extent(3));

            Optical_props::extract_subset(in_d, col_start, col_end, out);
            Kokkos::fence();

            return Numpy::from_device(out);
        },
        py::arg("array_in"), py::arg("col_start"), py::arg("col_end"),
        "Extract a range of columns from a (ngpt, nlay, ncol, nmom) array.");
}
