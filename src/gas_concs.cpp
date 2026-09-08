#include <algorithm>
#include <stdexcept>

#include "gas_concs.h"


std::string Gas_concs::normalise(const std::string& name)
{
    std::string out = name;
    std::transform(out.begin(), out.end(), out.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return out;
}


void Gas_concs::set_vmr(const std::string& name, const TF value)
{
    Entry e;
    e.kind = Kind::Scalar;
    e.value = value;

    gases[normalise(name)] = e;
}


void Gas_concs::set_vmr(const std::string& name, const Array_1d<const TF>& vmr)
{
    Entry e;
    e.kind = Kind::Profile;
    e.profile = Array_1d<TF>(Kokkos::view_alloc("vmr_profile", Kokkos::WithoutInitializing),
                             vmr.extent(0));
    Kokkos::deep_copy(e.profile, vmr);

    gases[normalise(name)] = e;
}


void Gas_concs::set_vmr(const std::string& name, const Array_2d<const TF>& vmr)
{
    Entry e;
    e.kind = Kind::Field;
    e.field = Array_2d<TF>(Kokkos::view_alloc("vmr_field", Kokkos::WithoutInitializing),
                           vmr.extent(0), vmr.extent(1));
    Kokkos::deep_copy(e.field, vmr);

    gases[normalise(name)] = e;
}


bool Gas_concs::contains(const std::string& name) const
{
    return gases.find(normalise(name)) != gases.end();
}


std::vector<std::string> Gas_concs::names() const
{
    std::vector<std::string> out;
    out.reserve(gases.size());

    for (const auto& [name, entry] : gases)
        out.push_back(name);

    return out;
}


void Gas_concs::get_vmr(const std::string& name, const Array_2d<TF>& out) const
{
    const auto it = gases.find(normalise(name));
    if (it == gases.end())
        throw std::invalid_argument("Gas '" + name + "' is not in the concentrations.");

    const Entry& e = it->second;

    const int nlay = static_cast<int>(out.extent(0));
    const int ncol = static_cast<int>(out.extent(1));

    if (e.kind == Kind::Scalar)
    {
        Kokkos::deep_copy(out, e.value);
        return;
    }

    if (e.kind == Kind::Profile)
    {
        if (static_cast<int>(e.profile.extent(0)) != nlay)
            throw std::invalid_argument("Profile for gas '" + name + "' has the wrong number of layers.");

        const auto profile = e.profile;
        parallel_for_2d("get_vmr_profile", {0, 0}, {nlay, ncol},
            KOKKOS_LAMBDA(const int ilay, const int icol) { out(ilay, icol) = profile(ilay); });

        return;
    }

    if (static_cast<int>(e.field.extent(0)) != nlay || static_cast<int>(e.field.extent(1)) != ncol)
        throw std::invalid_argument("Field for gas '" + name + "' has the wrong shape.");

    Kokkos::deep_copy(out, e.field);
}
