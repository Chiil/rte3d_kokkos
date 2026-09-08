#pragma once

#include <map>
#include <string>
#include <vector>

#include "types.h"


// Volume mixing ratios by gas name, the counterpart of the reference's ty_gas_concs.
//
// Strings live here and in the k-distribution loader only; they never reach a kernel.
// The container exists so a host model can say "co2 is 348 ppm" without knowing
// RRTMGP's internal gas ordering, and so the k-distribution can be reduced to the
// gases actually supplied.
//
// Names are compared case-insensitively, as the reference does.
class Gas_concs
{
    public:
        // A single value for the whole domain, a profile varying only with height, or
        // a full field.
        void set_vmr(const std::string& name, const TF value);
        void set_vmr(const std::string& name, const Array_1d<const TF>& vmr);  // (nlay)
        void set_vmr(const std::string& name, const Array_2d<const TF>& vmr);  // (nlay, ncol)

        bool contains(const std::string& name) const;

        // Sorted, lower-case.
        std::vector<std::string> names() const;

        // Fill out(nlay, ncol) with this gas, broadcasting a scalar or profile.
        // Throws if the gas is absent.
        void get_vmr(const std::string& name, const Array_2d<TF>& out) const;

        static std::string normalise(const std::string& name);

        static void init_python_bindings(py::module_& m);

    private:
        enum class Kind { Scalar, Profile, Field };

        struct Entry
        {
            Kind kind = Kind::Scalar;
            TF value = TF(0.);
            Array_1d<TF> profile;
            Array_2d<TF> field;
        };

        std::map<std::string, Entry> gases;
};
