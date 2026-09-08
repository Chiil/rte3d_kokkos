#include <array>
#include <algorithm>
#include <stdexcept>

#include "gas_optics.h"


// Reduction of a coefficient file to the gases a host model supplies.
//
// This is the only place, along with Gas_concs, where gas names are interpreted. Every
// output is an integer index; nothing downstream ever sees a string.
namespace
{
    // Position of a name in a list, or -1. The reference's string_loc_in_array, which
    // returns a 1-based index; here 0-based, with the same -1 for absent.
    int index_of(const std::string& name, const std::vector<std::string>& list)
    {
        const std::string key = Gas_concs::normalise(name);

        for (std::size_t i=0; i<list.size(); ++i)
            if (Gas_concs::normalise(list[i]) == key)
                return static_cast<int>(i);

        return -1;
    }


    // Index into the gas dimension of col_gas, where 0 is dry air, so a gas at
    // position p of the reduced list sits at p+1. Absent gases give -1, which every
    // consumer tests for with > 0.
    int col_gas_index(const std::string& name, const std::vector<std::string>& gas_names)
    {
        const int p = index_of(name, gas_names);
        return p < 0 ? -1 : p + 1;
    }


    template<typename View>
    auto to_device(const View& host)
    {
        using T = typename View::non_const_value_type;

        if constexpr (View::rank == 1)
        {
            Array_1d<T> d(Kokkos::view_alloc(host.label(), Kokkos::WithoutInitializing), host.extent(0));
            Kokkos::deep_copy(d, host);
            return d;
        }
        else if constexpr (View::rank == 2)
        {
            Array_2d<T> d(Kokkos::view_alloc(host.label(), Kokkos::WithoutInitializing),
                          host.extent(0), host.extent(1));
            Kokkos::deep_copy(d, host);
            return d;
        }
        else if constexpr (View::rank == 3)
        {
            Array_3d<T> d(Kokkos::view_alloc(host.label(), Kokkos::WithoutInitializing),
                          host.extent(0), host.extent(1), host.extent(2));
            Kokkos::deep_copy(d, host);
            return d;
        }
        else
        {
            Array_4d<T> d(Kokkos::view_alloc(host.label(), Kokkos::WithoutInitializing),
                          host.extent(0), host.extent(1), host.extent(2), host.extent(3));
            Kokkos::deep_copy(d, host);
            return d;
        }
    }


    // Reduce one set of minor absorbers to those whose gas the host supplies, and
    // renumber their blocks in kminor so the survivors tile the reduced table exactly.
    // Reference: reduce_minor_arrays.
    Minor_absorbers reduce_minor(
            const Kdist_file& file,
            const Gas_concs& available_gases,
            const std::vector<std::string>& gas_names_red,
            const std::vector<std::string>& minor_gases,
            const std::vector<std::string>& scaling_gas,
            const Array_3d_h<TF>& kminor,
            const Array_2d_h<int>& limits_gpt,
            const Array_1d_h<Bool>& scales_with_density,
            const Array_1d_h<Bool>& scale_by_complement,
            const Array_1d_h<int>& kminor_start)
    {
        const int nminor = static_cast<int>(minor_gases.size());

        // An absorber survives if the gas behind its identifier is available. The
        // identifier (say h2o_slf) maps through gas_minor to the gas itself (h2o).
        std::vector<int> keep;
        int tot_g = 0;

        for (int i=0; i<nminor; ++i)
        {
            const int idx_mnr = index_of(minor_gases[i], file.identifier_minor);
            if (idx_mnr < 0)
                throw std::invalid_argument(
                        "Minor absorber '" + minor_gases[i] + "' is not in identifier_minor.");

            if (available_gases.contains(file.gas_minor[idx_mnr]))
            {
                keep.push_back(i);
                tot_g += limits_gpt(i, 1) - limits_gpt(i, 0) + 1;
            }
        }

        const int red_nm = static_cast<int>(keep.size());
        const int neta = static_cast<int>(kminor.extent(1));
        const int ntemp = static_cast<int>(kminor.extent(2));

        Array_3d_h<TF> kminor_red("kminor_red", tot_g, neta, ntemp);
        Array_2d_h<int> limits_red("minor_limits_gpt", red_nm, 2);
        Array_1d_h<Bool> density_red("scales_with_density", red_nm);
        Array_1d_h<Bool> complement_red("scale_by_complement", red_nm);
        Array_1d_h<int> start_red("kminor_start", red_nm);
        Array_1d_h<int> idx_minor("idx_minor", red_nm);
        Array_1d_h<int> idx_scaling("idx_minor_scaling", red_nm);

        // n_elim tracks how many g-point blocks have been dropped, so the survivors'
        // starts close the gaps left behind.
        int n_elim = 0;
        int icnt = 0;

        for (int i=0; i<nminor; ++i)
        {
            const int ng = limits_gpt(i, 1) - limits_gpt(i, 0) + 1;

            if (icnt < red_nm && keep[icnt] == i)
            {
                // File indices are 1-based; ours are 0-based.
                limits_red(icnt, 0) = limits_gpt(i, 0) - 1;
                limits_red(icnt, 1) = limits_gpt(i, 1) - 1;
                density_red(icnt) = scales_with_density(i);
                complement_red(icnt) = scale_by_complement(i);

                const int start_in = kminor_start(i) - 1;
                const int start_out = start_in - n_elim;
                start_red(icnt) = start_out;

                for (int j=0; j<ng; ++j)
                    for (int ieta=0; ieta<neta; ++ieta)
                        for (int itemp=0; itemp<ntemp; ++itemp)
                            kminor_red(start_out + j, ieta, itemp) = kminor(start_in + j, ieta, itemp);

                // The gas this absorber belongs to, and the gas that scales it.
                const int idx_mnr = index_of(minor_gases[i], file.identifier_minor);
                idx_minor(icnt) = col_gas_index(file.gas_minor[idx_mnr], gas_names_red);
                idx_scaling(icnt) = col_gas_index(scaling_gas[i], gas_names_red);

                ++icnt;
            }
            else
                n_elim += ng;
        }

        Minor_absorbers m;
        m.kminor = to_device(kminor_red);
        m.minor_limits_gpt = to_device(limits_red);
        m.scales_with_density = to_device(density_red);
        m.scale_by_complement = to_device(complement_red);
        m.idx_minor = to_device(idx_minor);
        m.idx_minor_scaling = to_device(idx_scaling);
        m.kminor_start = to_device(start_red);

        return m;
    }


    // The (0,0) pair means "no key species"; the reference rewrites it to (2,2).
    std::array<int, 2> rewrite_pair(const std::array<int, 2>& pair)
    {
        if (pair[0] == 0 && pair[1] == 0)
            return {2, 2};

        return pair;
    }
}


Kdist_gas Gas_optics::load(const Kdist_file& file, const Gas_concs& available_gases)
{
    Kdist_gas k;

    // ---- which of the file's gases does the host supply? ---------------------
    std::vector<int> file_pos;
    for (std::size_t i=0; i<file.gas_names.size(); ++i)
        if (available_gases.contains(file.gas_names[i]))
        {
            k.gas_names.push_back(Gas_concs::normalise(file.gas_names[i]));
            file_pos.push_back(static_cast<int>(i));
        }

    const int ngas = static_cast<int>(k.gas_names.size());
    if (ngas == 0)
        throw std::invalid_argument("None of the k-distribution's gases were supplied.");

    // ---- reference volume mixing ratios, reduced ------------------------------
    {
        const int ntemp = static_cast<int>(file.vmr_ref.extent(0));
        Array_3d_h<TF> vmr_ref("vmr_ref", ntemp, ngas + 1, 2);

        for (int it=0; it<ntemp; ++it)
            for (int j=0; j<2; ++j)
            {
                // Entry 0 is used by the single-key-species method and is set to the
                // file's first entry, which is dry air.
                vmr_ref(it, 0, j) = file.vmr_ref(it, 0, j);

                for (int i=0; i<ngas; ++i)
                    vmr_ref(it, i + 1, j) = file.vmr_ref(it, file_pos[i] + 1, j);
            }

        k.vmr_ref = to_device(vmr_ref);
    }

    // ---- bands and g-points ---------------------------------------------------
    const int nbnd = static_cast<int>(file.band2gpt.extent(0));
    const int ngpt = static_cast<int>(file.kmajor.extent(0));

    Array_2d_h<int> band_lims("band_lims_gpt", nbnd, 2);
    Array_1d_h<int> band_start("band_gpt_start", nbnd);
    Array_1d_h<int> gpt_band("gpt_band", ngpt);

    for (int ibnd=0; ibnd<nbnd; ++ibnd)
    {
        band_lims(ibnd, 0) = file.band2gpt(ibnd, 0) - 1;
        band_lims(ibnd, 1) = file.band2gpt(ibnd, 1) - 1;
        band_start(ibnd) = band_lims(ibnd, 0);

        for (int igpt=band_lims(ibnd, 0); igpt<=band_lims(ibnd, 1); ++igpt)
            gpt_band(igpt) = ibnd;
    }

    // ---- key species, flavours and the g-point to flavour map -----------------
    // key_species holds 1-based positions in the file's gas list; remap them onto the
    // reduced list, keeping 0 as "no species". Reference: create_key_species_reduce.
    Array_3d_h<int> key_red("key_species_red", nbnd, 2, 2);

    for (int ibnd=0; ibnd<nbnd; ++ibnd)
        for (int iatm=0; iatm<2; ++iatm)
            for (int j=0; j<2; ++j)
            {
                const int ks = file.key_species(ibnd, iatm, j);

                if (ks == 0)
                    key_red(ibnd, iatm, j) = 0;
                else
                {
                    const int p = index_of(file.gas_names[ks - 1], k.gas_names);
                    if (p < 0)
                        throw std::invalid_argument(
                                "Key species '" + file.gas_names[ks - 1] + "' must be supplied.");

                    key_red(ibnd, iatm, j) = p + 1;
                }
            }

    // Unique key-species pairs, in first-seen order. Reference: create_flavor.
    std::vector<std::array<int, 2>> flavors;
    for (int ibnd=0; ibnd<nbnd; ++ibnd)
        for (int iatm=0; iatm<2; ++iatm)
        {
            const auto pair = rewrite_pair({key_red(ibnd, iatm, 0), key_red(ibnd, iatm, 1)});

            if (std::find(flavors.begin(), flavors.end(), pair) == flavors.end())
                flavors.push_back(pair);
        }

    const int nflav = static_cast<int>(flavors.size());

    Array_2d_h<int> flavor("flavor", nflav, 2);
    for (int i=0; i<nflav; ++i)
    {
        flavor(i, 0) = flavors[i][0];
        flavor(i, 1) = flavors[i][1];
    }

    // Reference: create_gpoint_flavor. Stored 0-based.
    Array_2d_h<int> gpoint_flavor("gpoint_flavor", ngpt, 2);
    for (int igpt=0; igpt<ngpt; ++igpt)
        for (int iatm=0; iatm<2; ++iatm)
        {
            const int ibnd = gpt_band(igpt);
            const auto pair = rewrite_pair({key_red(ibnd, iatm, 0), key_red(ibnd, iatm, 1)});

            gpoint_flavor(igpt, iatm) =
                    static_cast<int>(std::find(flavors.begin(), flavors.end(), pair) - flavors.begin());
        }

    k.flavor = to_device(flavor);
    k.gpoint_flavor = to_device(gpoint_flavor);
    k.band_lims_gpt = to_device(band_lims);
    k.band_gpt_start = to_device(band_start);
    k.gpt_band = to_device(gpt_band);
    k.gpt_band_h = gpt_band;
    k.kmajor = to_device(file.kmajor);

    // ---- interpolation grid and the scalars derived from it -------------------
    // The reference assumes temperature runs low to high and pressure high to low.
    {
        const int npres = static_cast<int>(file.press_ref.extent(0));
        const int ntemp = static_cast<int>(file.temp_ref.extent(0));

        Array_1d_h<TF> press_ref_log("press_ref_log", npres);
        for (int i=0; i<npres; ++i)
            press_ref_log(i) = Kokkos::log(file.press_ref(i));

        k.press_ref_log = to_device(press_ref_log);
        k.temp_ref = to_device(file.temp_ref);

        k.temp_ref_min = file.temp_ref(0);
        k.temp_ref_max = file.temp_ref(ntemp - 1);
        k.temp_ref_delta = (k.temp_ref_max - k.temp_ref_min) / static_cast<TF>(ntemp - 1);

        k.press_ref_log_delta =
                (press_ref_log(npres - 1) - press_ref_log(0)) / static_cast<TF>(npres - 1);
        k.press_ref_trop_log = Kokkos::log(file.press_ref_trop);

        k.neta = static_cast<int>(file.kmajor.extent(2));
    }

    // ---- minor absorbers ------------------------------------------------------
    k.lower = reduce_minor(file, available_gases, k.gas_names,
                           file.minor_gases_lower, file.scaling_gas_lower,
                           file.kminor_lower, file.minor_limits_gpt_lower,
                           file.minor_scales_with_density_lower,
                           file.scale_by_complement_lower, file.kminor_start_lower);

    k.upper = reduce_minor(file, available_gases, k.gas_names,
                           file.minor_gases_upper, file.scaling_gas_upper,
                           file.kminor_upper, file.minor_limits_gpt_upper,
                           file.minor_scales_with_density_upper,
                           file.scale_by_complement_upper, file.kminor_start_upper);

    k.lower.build_map(k.gpoint_flavor, ngpt, 0);
    k.upper.build_map(k.gpoint_flavor, ngpt, 1);

    k.idx_h2o = col_gas_index("h2o", k.gas_names);

    // ---- optional tables ------------------------------------------------------
    if (file.totplnk.size() > 0)
    {
        k.totplnk = to_device(file.totplnk);
        k.pfracin = to_device(file.planck_frac);
        // The Planck table spans the same temperature range as the k-distribution's
        // own grid, not the reference P/T of the absorption coefficients.
        k.totplnk_delta = (k.temp_ref_max - k.temp_ref_min)
                        / static_cast<TF>(file.totplnk.extent(1) - 1);
    }

    if (file.rayl.size() > 0)
    {
        k.krayl = to_device(file.rayl);

        // Solar source for the default facular and sunspot indices. Reference:
        // set_solar_variability. The absolute scale is arbitrary here: callers
        // renormalise to their own total solar irradiance, which is what the RFMIP
        // driver does, so the reference's subsequent scaling to tsi_default cancels.
        constexpr TF a_offset = TF(0.1495954);
        constexpr TF b_offset = TF(0.00066696);

        const int ngpt_sw = static_cast<int>(file.solar_source_quiet.extent(0));
        Array_1d_h<TF> solar("solar_source", ngpt_sw);

        for (int igpt=0; igpt<ngpt_sw; ++igpt)
            solar(igpt) = file.solar_source_quiet(igpt)
                        + (file.mg_default - a_offset) * file.solar_source_facular(igpt)
                        + (file.sb_default - b_offset) * file.solar_source_sunspot(igpt);

        k.solar_source = to_device(solar);
    }

    return k;
}
