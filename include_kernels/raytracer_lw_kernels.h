#pragma once

#include "random.h"
#include "raytracer_common.h"
#include "raytracer_lw.h"
#include "types.h"


// The photon walk of the longwave ray tracer.
//
// Ported from raytracer_kernels_lw.cu in rte-rrtmgp-cpp. The transport is the same
// null-collision walk as the shortwave tracer's, and everything geometric comes from
// raytracer_common.h; what is here is the emission sampling and the net-flux
// bookkeeping.
//
// One thread walks photons_to_shoot photons in sequence, reusing the same registers,
// which is why the whole walk is one function rather than a kernel per event.
namespace Rt_lw_kernels
{
    using namespace Rt_common;


    // Which of the three kinds of source emitted a photon. The slot it was drawn from
    // says which, and the walk has to remember it: a terminating photon gives its
    // deposit back to whatever emitted it, and that is a different array in each case.
    enum class Source { Atmosphere, Surface, Top };

    struct Photon
    {
        Vector<TF> position;
        Vector<TF> direction;

        // Where the photon came from, and how much weight it has parted with since.
        Source source;
        int source_idx;         // the column, or the cell, depending on the source
        TF emitted;
    };


    // Everything the walk reads and writes, gathered so that a launch captures one
    // object. Views are handles, so this is cheap to copy to the device.
    struct Scene
    {
        Array_map_2d<const Optics_cell> optics;   // (nz, ncol)
        Array_map_3d<const TF> k_null;            // (kn_z, kn_y, kn_x) [1/m]
        Array_map_1d<const TF> sfc_emis;          // (ncol)

        // The cumulative emitted power over the (nz+2)*ncol source slots, and how many
        // of them there are. Its last element is the total.
        Array_map_1d<const double> cdf;
        int nslot = 0;

        // Photon counts. Written with atomics, since photons from any thread may land
        // in any cell.
        Array_map_1d<TF> tod_dn, tod_up, sfc_dn, sfc_up;   // (ncol)
        Array_map_2d<TF> atmos;                            // (nz, ncol)

        Vector<int> grid_cells;
        Vector<TF> grid_d;
        Vector<TF> grid_d_inv;
        Vector<TF> grid_size;
        Vector<int> kn_grid;
        Vector<TF> kn_grid_d;
        Vector<TF> kn_grid_d_inv;

        KOKKOS_INLINE_FUNCTION
        int column(const int i, const int j) const { return i + j*grid_cells.x; }
    };


    // The first slot whose running total is above x. Slot i owns an interval of the
    // line exactly as wide as its own emitted power, so a uniform x picks it with the
    // probability it should.
    //
    // Used once per thread, to find where that thread's slice of the distribution
    // starts. Per photon the walk uses advance_cdf below instead, which is the whole
    // point: this search jumps around an array of millions of doubles, and every
    // thread of a warp jumps somewhere different, so one warp-wide load fans out into
    // as many sectors as there are threads. Measured on a 192x192x128 field, the
    // per-photon form of this search was 14 percent of the walk's load instructions
    // and 71 percent of its L1 sector traffic, in a kernel that is L1-throttle bound.
    RTE3D_DEVICE_FUNCTION
    int lower_bound_cdf(const Array_map_1d<const double>& cdf, const int n, const double x)
    {
        int lo = 0;
        int hi = n - 1;
        while (lo < hi)
        {
            const int mid = (lo + hi)/2;
            if (cdf(mid) > x)
                hi = mid;
            else
                lo = mid + 1;
        }

        return lo;
    }


    // The same answer, reached by walking forward from where the last photon left off.
    //
    // A thread's photons are stratified: photon i of the launch takes its draw from
    // [i, i+1)/photons_total, so within a thread the draw only ever increases and the
    // slot it lands in only ever moves forward. The cursor therefore crosses each slot
    // at most once over the thread's whole run -- nslot/nthread of them in total,
    // about nine on a domain of this size against a hundred and more photons -- and it
    // crosses them in order, so the loads are sequential and hit the line the last one
    // brought in.
    //
    // Stratifying is not only cheaper but less noisy: one photon per stratum instead
    // of a Poisson scatter of them, which is the same variance reduction the shortwave
    // tracer gets from drawing its launch pixel out of a low-discrepancy sequence.
    RTE3D_DEVICE_FUNCTION
    int advance_cdf(const Array_map_1d<const double>& cdf, const int n, int& cursor,
                    const double x)
    {
        while (cursor < n - 1 && cdf(cursor) <= x)
            ++cursor;

        return cursor;
    }


    // Give a terminating photon's deposit back to whatever emitted it.
    //
    // What is written is the weight the photon actually parted with, not the unit
    // weight it set out with. The two are equal in the mean -- Russian roulette is
    // unbiased -- but taking the actual deposit makes the cancellation exact photon by
    // photon: a photon reabsorbed in the cell that emitted it leaves that cell's net
    // flux untouched, where scoring the emission separately would leave two large
    // numbers to cancel and all of the noise that implies. Reference: write_emission.
    RTE3D_DEVICE_FUNCTION
    void write_emission(const Scene& s, const Photon& photon)
    {
        if (photon.source == Source::Atmosphere)
        {
            const int ncol = s.grid_cells.x*s.grid_cells.y;
            Kokkos::atomic_add(&s.atmos(photon.source_idx/ncol, photon.source_idx%ncol),
                               -photon.emitted);
        }
        else if (photon.source == Source::Surface)
            Kokkos::atomic_add(&s.sfc_up(photon.source_idx), photon.emitted);
        else
            Kokkos::atomic_add(&s.tod_dn(photon.source_idx), photon.emitted);
    }


    // Emit a new photon from a source drawn in proportion to its emitted power, at a
    // random point within it.
    //
    // The slot order is the reference's: 0 the surface, 1 to nz the cells of the box
    // from the bottom up, nz+1 the top boundary. The surface radiates up and the top
    // boundary down, both by the cosine law; a cell radiates isotropically.
    RTE3D_DEVICE_FUNCTION
    void reset_photon(
            Photon& photon, TF& weight, int& photons_shot, const int photons_to_shoot,
            const Scene& s, const TF s_min, int& cursor,
            const double u0, const double du, Rand::Rng& rng)
    {
        ++photons_shot;
        if (photons_shot >= photons_to_shoot)
            return;

        const int ncol = s.grid_cells.x*s.grid_cells.y;

        // This photon's stratum of the emitted power, and a point within it.
        const double u = u0 + (photons_shot + rng.uniform_double())*du;
        const int slot = advance_cdf(s.cdf, s.nslot, cursor, u*s.cdf(s.nslot - 1));

        const int ij = slot%ncol;
        const int k = slot/ncol - 1;

        const int i = ij%s.grid_cells.x;
        const int j = ij/s.grid_cells.x;

        photon.position.x = (i + rng())*s.grid_d.x;
        photon.position.y = (j + rng())*s.grid_d.y;

        if (k < 0)
        {
            photon.position.z = TF(0.);
            photon.direction = cosine_direction(TF(1.), rng);
            photon.source = Source::Surface;
            photon.source_idx = ij;
        }
        else if (k >= s.grid_cells.z)
        {
            // Just inside the domain: eps() is relative, and grid_size.z minus a
            // relative epsilon of itself rounds straight back to grid_size.z.
            photon.position.z = s.grid_size.z - s_min;
            photon.direction = cosine_direction(TF(-1.), rng);
            photon.source = Source::Top;
            photon.source_idx = ij;
        }
        else
        {
            photon.position.z = (k + rng())*s.grid_d.z;

            // Isotropic, so the cosine is uniform rather than square-rooted.
            const TF mu = TF(2.)*rng() - TF(1.);
            const TF azimuth = TF(2.*M_PI)*rng();
            const TF sin_theta = Kokkos::sqrt(TF(1.) - mu*mu + eps());

            photon.direction.x = sin_theta*Kokkos::sin(azimuth);
            photon.direction.y = sin_theta*Kokkos::cos(azimuth);
            photon.direction.z = mu;

            photon.source = Source::Atmosphere;
            photon.source_idx = k*ncol + ij;
        }

        photon.emitted = TF(0.);
        weight = TF(1.);
    }


    // Write out the absorbed weight a photon has piled up in one fine cell.
    //
    // The walk deposits at every collision, null ones included, and a photon takes
    // several of those in the same cell -- so the deposit is held in a register and
    // written with one atomic when the photon leaves the cell, rather than with an
    // atomic per collision. A cell that absorbs nothing asks for no atomic at all.
    RTE3D_DEVICE_FUNCTION
    void flush_absorbed(const Scene& s, TF& absorbed, const int k, const int ij)
    {
        if (absorbed > TF(0.))
        {
            Kokkos::atomic_add(&s.atmos(k, ij), absorbed);
            absorbed = TF(0.);
        }
    }


    // Walk photons_to_shoot photons through the scene, from wherever they were emitted
    // until they leave it or their weight runs out.
    template<bool independent_column>
    RTE3D_DEVICE_FUNCTION
    void trace_photons(
            const Scene& s, const int photons_to_shoot, const unsigned int rng_seed,
            const double u0, const double du)
    {
        Rand::Rng rng(rng_seed);

        // Where this thread's slice of the emitted power begins. The one binary search
        // of the whole run; every photon after this walks forward from it.
        int cursor = lower_bound_cdf(s.cdf, s.nslot, u0*s.cdf(s.nslot - 1));

        // The nudge that carries a photon past a cell face it has just landed on.
        const TF s_min = Kokkos::max(s.grid_size.z,
                Kokkos::max(s.grid_size.y, s.grid_size.x)) * eps();

        Photon photon;
        TF weight = TF(0.);
        int photons_shot = -1;

        reset_photon(photon, weight, photons_shot, photons_to_shoot, s, s_min,
                     cursor, u0, du, rng);

        TF tau = TF(0.);
        TF d_max = TF(0.);
        TF k_ext_null = TF(0.);
        TF k_ext_null_inv = TF(0.);
        bool transition = false;
        int i_n = 0, j_n = 0, k_n = 0;

        // The fine cell the last collision fell in, its optical properties, and the
        // weight absorbed there so far. Consecutive collisions land in the same cell
        // often enough -- null collisions above all -- that both the loads and the
        // atomic are worth holding back for.
        int c_k = -1, c_ij = -1;
        Optics_cell c_optics{};
        TF absorbed = TF(0.);

        while (photons_shot < photons_to_shoot)
        {
            // Entering a new block of the null-collision grid: find it, and how far
            // the photon may go before it has to be looked up again.
            if (d_max == TF(0.))
            {
                i_n = coord_to_index(photon.position.x, s.kn_grid_d_inv.x, s.kn_grid.x);
                j_n = coord_to_index(photon.position.y, s.kn_grid_d_inv.y, s.kn_grid.y);
                k_n = coord_to_index(photon.position.z, s.kn_grid_d_inv.z, s.kn_grid.z);

                const TF sx = Kokkos::abs((photon.direction.x > 0
                        ? (i_n+1)*s.kn_grid_d.x - photon.position.x
                        : i_n*s.kn_grid_d.x - photon.position.x) / photon.direction.x);
                const TF sy = Kokkos::abs((photon.direction.y > 0
                        ? (j_n+1)*s.kn_grid_d.y - photon.position.y
                        : j_n*s.kn_grid_d.y - photon.position.y) / photon.direction.y);
                const TF sz = Kokkos::abs((photon.direction.z > 0
                        ? (k_n+1)*s.kn_grid_d.z - photon.position.z
                        : k_n*s.kn_grid_d.z - photon.position.z) / photon.direction.z);

                d_max = independent_column ? sz : Kokkos::min(sx, Kokkos::min(sy, sz));
                k_ext_null = s.k_null(k_n, j_n, i_n);
                k_ext_null_inv = TF(1.)/k_ext_null;
            }

            // A photon that crossed a block face keeps the optical depth it had left.
            if (!transition)
                tau = sample_tau(rng());
            transition = false;

            const TF dn = Kokkos::max(eps(), tau*k_ext_null_inv);

            if (dn >= d_max)
            {
                // The collision is beyond this block: move to its face instead.
                if (!independent_column)
                {
                    photon.position.x += photon.direction.x*(s_min + d_max);
                    photon.position.y += photon.direction.y*(s_min + d_max);
                }
                photon.position.z += photon.direction.z*(s_min + d_max);

                if (photon.position.z < eps())
                {
                    // The surface: score the incoming flux, let the emissivity absorb
                    // its share, and send the rest back up isotropically.
                    photon.position.z = eps();
                    d_max = TF(0.);

                    const int i = coord_to_index(photon.position.x, s.grid_d_inv.x, s.grid_cells.x);
                    const int j = coord_to_index(photon.position.y, s.grid_d_inv.y, s.grid_cells.y);
                    const int ij = s.column(i, j);

                    Kokkos::atomic_add(&s.sfc_dn(ij), weight);

                    const TF albedo = TF(1.) - s.sfc_emis(ij);
                    photon.emitted += (TF(1.) - albedo)*weight;

                    weight *= albedo;
                    Kokkos::atomic_add(&s.sfc_up(ij), weight);

                    if (weight < w_thres())
                        weight = (rng() > weight) ? TF(0.) : TF(1.);

                    if (weight > TF(0.))
                        photon.direction = cosine_direction(TF(1.), rng);
                    else
                    {
                        write_emission(s, photon);
                        reset_photon(photon, weight, photons_shot, photons_to_shoot, s, s_min,
                     cursor, u0, du, rng);
                    }
                }
                else if (photon.position.z >= s.grid_size.z)
                {
                    // Out of the top: the weight that leaves is weight the photon has
                    // parted with, as surely as weight an absorption took.
                    d_max = TF(0.);

                    const int i = coord_to_index(photon.position.x, s.grid_d_inv.x, s.grid_cells.x);
                    const int j = coord_to_index(photon.position.y, s.grid_d_inv.y, s.grid_cells.y);
                    Kokkos::atomic_add(&s.tod_up(s.column(i, j)), weight);

                    photon.emitted += weight;

                    write_emission(s, photon);
                    reset_photon(photon, weight, photons_shot, photons_to_shoot, s, s_min,
                     cursor, u0, du, rng);
                }
                else
                {
                    // A block face inside the domain: nudge across it, wrap around the
                    // sides, and keep the optical depth that is left.
                    photon.position.z += photon.direction.z > 0 ? s_min : -s_min;

                    if (!independent_column)
                    {
                        photon.position.x += photon.direction.x > 0 ? s_min : -s_min;
                        photon.position.y += photon.direction.y > 0 ? s_min : -s_min;

                        // A step ends on the face of the block it started in, so a
                        // photon leaves the domain by at most one nudge and the wrap
                        // is one subtraction or one addition, not a modulo.
                        if (photon.position.x >= s.grid_size.x)
                            photon.position.x -= s.grid_size.x;
                        else if (photon.position.x < TF(0.))
                            photon.position.x += s.grid_size.x;

                        if (photon.position.y >= s.grid_size.y)
                            photon.position.y -= s.grid_size.y;
                        else if (photon.position.y < TF(0.))
                            photon.position.y += s.grid_size.y;
                    }

                    tau -= d_max*k_ext_null;
                    d_max = TF(0.);
                    transition = true;
                }
            }
            else
            {
                // A collision inside this block. Move there, staying inside it.
                const TF dz = photon.direction.z*dn;
                photon.position.z = dz > 0
                        ? Kokkos::min(photon.position.z + dz, (k_n+1)*s.kn_grid_d.z - s_min)
                        : Kokkos::max(photon.position.z + dz, k_n*s.kn_grid_d.z + s_min);

                if (!independent_column)
                {
                    const TF dx = photon.direction.x*dn;
                    const TF dy = photon.direction.y*dn;

                    photon.position.x = dx > 0
                            ? Kokkos::min(photon.position.x + dx, (i_n+1)*s.kn_grid_d.x - s_min)
                            : Kokkos::max(photon.position.x + dx, i_n*s.kn_grid_d.x + s_min);
                    photon.position.y = dy > 0
                            ? Kokkos::min(photon.position.y + dy, (j_n+1)*s.kn_grid_d.y - s_min)
                            : Kokkos::max(photon.position.y + dy, j_n*s.kn_grid_d.y + s_min);
                }

                const int i = coord_to_index(photon.position.x, s.grid_d_inv.x, s.grid_cells.x);
                const int j = coord_to_index(photon.position.y, s.grid_d_inv.y, s.grid_cells.y);
                const int k = coord_to_index(photon.position.z, s.grid_d_inv.z, s.grid_cells.z);
                const int ij = s.column(i, j);

                // A new cell ends the accumulation the last one was collecting, and
                // the usefulness of its cached optics.
                if (k != c_k || ij != c_ij)
                {
                    flush_absorbed(s, absorbed, c_k, c_ij);

                    c_optics = s.optics(k, ij);
                    c_k = k;
                    c_ij = ij;
                }

                const Optics_cell optics = c_optics;
                const TF k_ext = optics.k_ext;
                const TF k_sca_tot = optics.k_sca_gas + optics.k_sca_cld;

                // Absorption is taken out of the weight rather than sampled, which is
                // the variance reduction of Iwabuchi (2006). The null part of the
                // extinction absorbs nothing, so of the extinction the transport
                // marches on, only this cell's absorption takes anything out of the
                // weight.
                //
                // The fraction absorbed is written as k_abs/k_ext_null rather than
                // the algebraically equal (k_ext_null - k_abs)/k_ext_null, so that a
                // conservative cell, where k_abs is exactly zero, leaves the weight
                // exactly alone: the second form would multiply k_ext_null by its own
                // rounded reciprocal and come back a rounding short of one, and that
                // shortfall accumulates over a scattering photon's many collisions.
                const TF k_abs = k_ext - k_sca_tot;
                const TF f_abs = k_abs*k_ext_null_inv;
                const TF f_no_abs = TF(1.) - f_abs;

                // What the extinction leaves once the absorption is out of it. Never
                // negative, k_ext_null being the largest extinction in the block.
                const TF k_ext_no_abs = k_ext_null - k_abs;

                absorbed += weight*f_abs;
                photon.emitted += weight*f_abs;

                weight *= f_no_abs;
                if (weight < w_thres())
                    weight = (rng() > weight) ? TF(0.) : TF(1.);

                if (weight > TF(0.))
                {
                    // Null collision, or a real one: of the extinction that did
                    // not absorb, the scattering part is what deflects the photon.
                    // Written as a product rather than the ratio it came from, which
                    // spares the division and, in a cell that is the block's own
                    // maximum, the cancellation that ratio suffers.
                    if (rng()*k_ext_no_abs >= k_sca_tot)
                    {
                        d_max -= dn;
                    }
                    else
                    {
                        d_max = TF(0.);
                        photon.direction = scatter_direction(
                                photon.direction, sample_cos_scat(optics, k_sca_tot, rng), rng);
                    }
                }
                else
                {
                    d_max = TF(0.);

                    write_emission(s, photon);
                    reset_photon(photon, weight, photons_shot, photons_to_shoot, s, s_min,
                     cursor, u0, du, rng);
                }
            }
        }

        // What the last cell collected. A reset needs no flush of its own: the key is
        // the cell, not the photon, so a deposit left standing is only ever added to by
        // a photon that would have written to the same place.
        flush_absorbed(s, absorbed, c_k, c_ij);
    }
}
