#include <pybind11/stl.h>

#include "gas_optics.h"
#include "runtime.h"


namespace
{
    template<typename T>
    Numpy::In<T> get(const py::dict& d, const char* key)
    {
        if (!d.contains(key))
            throw std::invalid_argument(std::string("Coefficient file is missing '") + key + "'");

        return d[key].cast<Numpy::In<T>>();
    }

    std::vector<std::string> get_names(const py::dict& d, const char* key)
    {
        if (!d.contains(key))
            throw std::invalid_argument(std::string("Coefficient file is missing '") + key + "'");

        return d[key].cast<std::vector<std::string>>();
    }
}


void Gas_optics::init_load_python_bindings(py::module_& m)
{
    m.def("load_kdist",
        [](const py::dict& f, const Gas_concs& gas_concs) -> py::dict
        {
            Runtime::get();

            Kdist_file file;
            file.gas_names = get_names(f, "gas_names");
            file.gas_minor = get_names(f, "gas_minor");
            file.identifier_minor = get_names(f, "identifier_minor");
            file.minor_gases_lower = get_names(f, "minor_gases_lower");
            file.minor_gases_upper = get_names(f, "minor_gases_upper");
            file.scaling_gas_lower = get_names(f, "scaling_gas_lower");
            file.scaling_gas_upper = get_names(f, "scaling_gas_upper");

            file.key_species = Numpy::to_host_3d<int>(get<int>(f, "key_species"), "key_species");
            file.band2gpt = Numpy::to_host_2d<int>(get<int>(f, "band2gpt"), "band2gpt");
            file.press_ref = Numpy::to_host_1d<TF>(get<TF>(f, "press_ref"), "press_ref");
            file.temp_ref = Numpy::to_host_1d<TF>(get<TF>(f, "temp_ref"), "temp_ref");
            file.press_ref_trop = f["press_ref_trop"].cast<TF>();
            file.temp_ref_p = f["temp_ref_p"].cast<TF>();
            file.temp_ref_t = f["temp_ref_t"].cast<TF>();
            file.vmr_ref = Numpy::to_host_3d<TF>(get<TF>(f, "vmr_ref"), "vmr_ref");
            file.kmajor = Numpy::to_host_4d<TF>(get<TF>(f, "kmajor"), "kmajor");

            file.kminor_lower = Numpy::to_host_3d<TF>(get<TF>(f, "kminor_lower"), "kminor_lower");
            file.kminor_upper = Numpy::to_host_3d<TF>(get<TF>(f, "kminor_upper"), "kminor_upper");
            file.minor_limits_gpt_lower = Numpy::to_host_2d<int>(
                    get<int>(f, "minor_limits_gpt_lower"), "minor_limits_gpt_lower");
            file.minor_limits_gpt_upper = Numpy::to_host_2d<int>(
                    get<int>(f, "minor_limits_gpt_upper"), "minor_limits_gpt_upper");
            file.minor_scales_with_density_lower = Numpy::to_host_1d<Bool>(
                    get<Bool>(f, "minor_scales_with_density_lower"), "msd_lower");
            file.minor_scales_with_density_upper = Numpy::to_host_1d<Bool>(
                    get<Bool>(f, "minor_scales_with_density_upper"), "msd_upper");
            file.scale_by_complement_lower = Numpy::to_host_1d<Bool>(
                    get<Bool>(f, "scale_by_complement_lower"), "sbc_lower");
            file.scale_by_complement_upper = Numpy::to_host_1d<Bool>(
                    get<Bool>(f, "scale_by_complement_upper"), "sbc_upper");
            file.kminor_start_lower = Numpy::to_host_1d<int>(
                    get<int>(f, "kminor_start_lower"), "kminor_start_lower");
            file.kminor_start_upper = Numpy::to_host_1d<int>(
                    get<int>(f, "kminor_start_upper"), "kminor_start_upper");

            if (f.contains("totplnk"))
            {
                file.totplnk = Numpy::to_host_2d<TF>(get<TF>(f, "totplnk"), "totplnk");
                file.planck_frac = Numpy::to_host_4d<TF>(get<TF>(f, "planck_frac"), "planck_frac");
            }
            if (f.contains("rayl"))
                file.rayl = Numpy::to_host_4d<TF>(get<TF>(f, "rayl"), "rayl");

            const Kdist_gas k = Gas_optics::load(file, gas_concs);

            py::dict out;
            out["gas_names"] = k.gas_names;
            out["flavor"] = Numpy::from_device(k.flavor);
            out["gpoint_flavor"] = Numpy::from_device(k.gpoint_flavor);
            out["vmr_ref"] = Numpy::from_device(k.vmr_ref);
            out["band_lims_gpt"] = Numpy::from_device(k.band_lims_gpt);
            out["gpt_band"] = Numpy::from_device(k.gpt_band);
            out["band_gpt_start"] = Numpy::from_device(k.band_gpt_start);
            out["idx_h2o"] = k.idx_h2o;

            for (const auto& [prefix, m_] : {std::pair{"lower_", &k.lower}, std::pair{"upper_", &k.upper}})
            {
                out[(std::string(prefix) + "kminor").c_str()] = Numpy::from_device(m_->kminor);
                out[(std::string(prefix) + "minor_limits_gpt").c_str()] =
                        Numpy::from_device(m_->minor_limits_gpt);
                out[(std::string(prefix) + "scales_with_density").c_str()] =
                        Numpy::from_device(m_->scales_with_density);
                out[(std::string(prefix) + "scale_by_complement").c_str()] =
                        Numpy::from_device(m_->scale_by_complement);
                out[(std::string(prefix) + "idx_minor").c_str()] = Numpy::from_device(m_->idx_minor);
                out[(std::string(prefix) + "idx_minor_scaling").c_str()] =
                        Numpy::from_device(m_->idx_minor_scaling);
                out[(std::string(prefix) + "kminor_start").c_str()] =
                        Numpy::from_device(m_->kminor_start);
            }

            return out;
        },
        py::arg("file"), py::arg("gas_concs"),
        "Reduce a coefficient file to the gases in gas_concs and build every derived "
        "index array. Input indices are 1-based, as in the file; everything returned "
        "is 0-based.");
}
