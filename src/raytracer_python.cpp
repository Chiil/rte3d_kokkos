#include <optional>

#include <pybind11/stl.h>

#include "raytracer.h"
#include "runtime.h"


namespace
{
    Array_2d<TF> optional_2d(const std::optional<Numpy::In<TF>>& a, const std::string& name)
    {
        if (!a.has_value())
            return Array_2d<TF>();

        return Numpy::to_device_2d<TF>(*a, name);
    }

    // A null-collision block count the caller did not give. Blocks a quarter of the
    // grid across are a compromise: coarser and the tracer takes null collisions it
    // did not have to, finer and it crosses block faces it did not have to.
    int default_kn(const int n, const int given)
    {
        return given > 0 ? given : Kokkos::max(1, n/4);
    }
}


void Raytracer::init_python_bindings(py::module_& m)
{
    m.def("trace_rays",
        [](const Numpy::In<TF>& tau_gas, const Numpy::In<TF>& ssa_gas,
           const Numpy::In<TF>& sfc_alb,
           const int nx, const int ny, const int nz,
           const TF dx, const TF dy, const TF dz,
           const TF mu0, const TF azi, const TF inc_dir, const TF inc_dif,
           const int photons_per_pixel,
           const std::optional<Numpy::In<TF>>& tau_cld,
           const std::optional<Numpy::In<TF>>& ssa_cld,
           const std::optional<Numpy::In<TF>>& asy_cld,
           const bool top_at_1, const bool independent_column,
           const int kn_x, const int kn_y, const int kn_z) -> py::dict
        {
            Runtime::get();

            Grid grid;
            grid.nx = nx; grid.ny = ny; grid.nz = nz;
            grid.dx = dx; grid.dy = dy; grid.dz = dz;
            grid.kn_x = default_kn(nx, kn_x);
            grid.kn_y = default_kn(ny, kn_y);
            grid.kn_z = default_kn(nz, kn_z);

            auto tau_gas_d = Numpy::to_device_2d<TF>(tau_gas, "tau_gas");
            auto ssa_gas_d = Numpy::to_device_2d<TF>(ssa_gas, "ssa_gas");
            auto sfc_alb_d = Numpy::to_device_1d<TF>(sfc_alb, "sfc_alb");
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

            const auto fluxes = Fluxes_rt::make(grid);
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
                    Array_map_1d<const TF>(sfc_alb_d.data(), ncol),
                    mu0, azi, inc_dir, inc_dif, fluxes, scratch);
            Kokkos::fence();

            py::dict out;
            out["tod_dn"] = Numpy::from_device(fluxes.tod_dn);
            out["tod_up"] = Numpy::from_device(fluxes.tod_up);
            out["sfc_dir"] = Numpy::from_device(fluxes.sfc_dir);
            out["sfc_dif"] = Numpy::from_device(fluxes.sfc_dif);
            out["sfc_up"] = Numpy::from_device(fluxes.sfc_up);
            out["abs_dir"] = Numpy::from_device(fluxes.abs_dir);
            out["abs_dif"] = Numpy::from_device(fluxes.abs_dif);

            return out;
        },
        py::arg("tau_gas"), py::arg("ssa_gas"), py::arg("sfc_alb"),
        py::arg("nx"), py::arg("ny"), py::arg("nz"),
        py::arg("dx"), py::arg("dy"), py::arg("dz"),
        py::arg("mu0"), py::arg("azi") = TF(0.),
        py::arg("inc_dir") = TF(1.), py::arg("inc_dif") = TF(0.),
        py::arg("photons_per_pixel") = 256,
        py::arg("tau_cld") = py::none(), py::arg("ssa_cld") = py::none(),
        py::arg("asy_cld") = py::none(),
        py::arg("top_at_1") = false, py::arg("independent_column") = false,
        py::arg("kn_x") = 0, py::arg("kn_y") = 0, py::arg("kn_z") = 0,
        "Trace one g-point through a three-dimensional box. The optical properties are "
        "(nlay, ncol) with ncol = nx*ny and the column index i + j*nx; layers from "
        "nz-1 upward are lumped into the top cell. mu0 and azi give the sun's "
        "direction, inc_dir the irradiance entering the top of the domain, already "
        "multiplied by mu0. Returns a dict with the surface and top-of-domain fluxes, "
        "(ncol) each, and the absorbed flux per unit height, (nz, ncol).");
}
