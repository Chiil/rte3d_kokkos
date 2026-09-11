#include <optional>

#include <pybind11/stl.h>

#include "raytracer_lw.h"
#include "runtime.h"


namespace
{
    Array_2d<TF> optional_2d(const std::optional<Numpy::In<TF>>& a, const std::string& name)
    {
        if (!a.has_value())
            return Array_2d<TF>();

        return Numpy::to_device_2d<TF>(*a, name);
    }
}


void Raytracer_lw::init_python_bindings(py::module_& m)
{
    m.def("trace_rays_lw",
        [](const Numpy::In<TF>& tau_gas, const Numpy::In<TF>& ssa_gas,
           const Numpy::In<TF>& lay_source, const Numpy::In<TF>& sfc_source,
           const Numpy::In<TF>& sfc_emis,
           const int nx, const int ny, const int nz,
           const TF dx, const TF dy, const TF dz,
           const int photons_per_pixel,
           const std::optional<Numpy::In<TF>>& tau_cld,
           const std::optional<Numpy::In<TF>>& ssa_cld,
           const std::optional<Numpy::In<TF>>& asy_cld,
           const TF inc_dif,
           const bool top_at_1, const bool independent_column,
           const int kn_x, const int kn_y, const int kn_z) -> py::dict
        {
            Runtime::get();

            const Grid grid = Grid::make(nx, ny, nz, dx, dy, dz, kn_x, kn_y, kn_z);

            auto tau_gas_d = Numpy::to_device_2d<TF>(tau_gas, "tau_gas");
            auto ssa_gas_d = Numpy::to_device_2d<TF>(ssa_gas, "ssa_gas");
            auto lay_source_d = Numpy::to_device_2d<TF>(lay_source, "lay_source");
            auto sfc_source_d = Numpy::to_device_1d<TF>(sfc_source, "sfc_source");
            auto sfc_emis_d = Numpy::to_device_1d<TF>(sfc_emis, "sfc_emis");
            auto tau_cld_d = optional_2d(tau_cld, "tau_cld");
            auto ssa_cld_d = optional_2d(ssa_cld, "ssa_cld");
            auto asy_cld_d = optional_2d(asy_cld, "asy_cld");

            const int nlay = static_cast<int>(tau_gas_d.extent(0));
            const int ncol = static_cast<int>(tau_gas_d.extent(1));

            if (ncol != grid.ncol())
                throw std::invalid_argument("tau_gas has " + std::to_string(ncol)
                        + " columns where the grid has " + std::to_string(grid.ncol()));
            if (nz > nlay)
                throw std::invalid_argument("The ray-tracing grid is deeper than the atmosphere");

            const auto fluxes = Fluxes_lw::make(grid);
            fluxes.zero();
            const auto scratch = Scratch::make(grid);

            const auto as_const_2d = [](const Array_2d<TF>& v)
            {
                return Array_map_2d<const TF>(v.data(), v.extent(0), v.extent(1));
            };

            trace_rays(
                    grid, top_at_1, independent_column, photons_per_pixel, 0,
                    as_const_2d(tau_gas_d), as_const_2d(ssa_gas_d),
                    as_const_2d(tau_cld_d), as_const_2d(ssa_cld_d), as_const_2d(asy_cld_d),
                    as_const_2d(lay_source_d),
                    Array_map_1d<const TF>(sfc_source_d.data(), ncol),
                    Array_map_1d<const TF>(sfc_emis_d.data(), ncol),
                    inc_dif, fluxes, scratch);
            Kokkos::fence();

            py::dict out;
            out["toa_dn"] = Numpy::from_device(fluxes.toa_dn);
            out["toa_up"] = Numpy::from_device(fluxes.toa_up);
            out["tod_dn"] = Numpy::from_device(fluxes.tod_dn);
            out["tod_up"] = Numpy::from_device(fluxes.tod_up);
            out["sfc_dn"] = Numpy::from_device(fluxes.sfc_dn);
            out["sfc_up"] = Numpy::from_device(fluxes.sfc_up);
            out["flux_net"] = Numpy::from_device(fluxes.flux_net);

            return out;
        },
        py::arg("tau_gas"), py::arg("ssa_gas"),
        py::arg("lay_source"), py::arg("sfc_source"), py::arg("sfc_emis"),
        py::arg("nx"), py::arg("ny"), py::arg("nz"),
        py::arg("dx"), py::arg("dy"), py::arg("dz"),
        py::arg("photons_per_pixel") = 256,
        py::arg("tau_cld") = py::none(), py::arg("ssa_cld") = py::none(),
        py::arg("asy_cld") = py::none(),
        py::arg("inc_dif") = TF(0.),
        py::arg("top_at_1") = false, py::arg("independent_column") = false,
        py::arg("kn_x") = 0, py::arg("kn_y") = 0, py::arg("kn_z") = 0,
        "Trace one longwave g-point through a three-dimensional box. The optical "
        "properties and the Planck sources are (nlay, ncol) with ncol = nx*ny and the "
        "column index i + j*nx; layers from nz-1 upward are lumped into the top cell, "
        "their emission included. sfc_source and sfc_emis are (ncol). Returns a dict "
        "with the surface and top-of-domain fluxes, (ncol) each, and flux_net, the "
        "absorbed minus emitted flux per unit height, (nz, ncol).");
}
