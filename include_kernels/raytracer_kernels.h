#pragma once

#include <limits>

#include "random.h"
#include "raytracer.h"
#include "types.h"


// The photon walk of the shortwave ray tracer, and the sampling it needs.
//
// Ported from raytracer_kernels_sw.cu in rte-rrtmgp-cpp, which follows Iwabuchi (2006)
// for the weighted absorption and the Russian roulette. The transport is
// null-collision (Woodcock tracking): within a block of the coarse kn grid every cell
// is treated as if it had that block's largest extinction, and the excess is undone by
// a collision that does nothing. That turns an inhomogeneous medium into a homogeneous
// one, so the free path is a single logarithm rather than a march through cells.
//
// One thread walks photons_to_shoot photons in sequence, reusing the same registers,
// which is why the whole walk is one function rather than a kernel per event.
namespace Rt_kernels
{
    using Raytracer::Optics_scat;
    using Raytracer::Vector;

    KOKKOS_INLINE_FUNCTION constexpr TF eps() { return std::numeric_limits<TF>::epsilon(); }

    // Below this the photon weight is put through Russian roulette rather than
    // followed further. Iwabuchi (2006), and the reference's w_thres.
    KOKKOS_INLINE_FUNCTION constexpr TF w_thres() { return TF(0.5); }


    template<typename T> KOKKOS_INLINE_FUNCTION
    Vector<T> operator+(const Vector<T> a, const Vector<T> b)
    { return Vector<T>{a.x + b.x, a.y + b.y, a.z + b.z}; }

    template<typename T> KOKKOS_INLINE_FUNCTION
    Vector<T> operator-(const Vector<T> a, const Vector<T> b)
    { return Vector<T>{a.x - b.x, a.y - b.y, a.z - b.z}; }

    template<typename T> KOKKOS_INLINE_FUNCTION
    Vector<T> operator*(const T s, const Vector<T> v)
    { return Vector<T>{s*v.x, s*v.y, s*v.z}; }

    template<typename T> KOKKOS_INLINE_FUNCTION
    T dot(const Vector<T>& a, const Vector<T>& b)
    { return a.x*b.x + a.y*b.y + a.z*b.z; }

    template<typename T> KOKKOS_INLINE_FUNCTION
    Vector<T> cross(const Vector<T>& a, const Vector<T>& b)
    {
        return Vector<T>{a.y*b.z - a.z*b.y,
                         a.z*b.x - a.x*b.z,
                         a.x*b.y - a.y*b.x};
    }

    template<typename T> KOKKOS_INLINE_FUNCTION
    Vector<T> normalize(const Vector<T> v)
    {
        const T len = Kokkos::sqrt(dot(v, v));
        return Vector<T>{v.x/len, v.y/len, v.z/len};
    }


    // Cosine of the scattering angle for Rayleigh scattering, sampled by the cubic
    // solution of the phase function's cumulative distribution.
    KOKKOS_INLINE_FUNCTION
    TF rayleigh(const TF r)
    {
        const TF q = TF(4.)*r - TF(2.);
        const TF d = TF(1.) + q*q;
        const TF u = Kokkos::pow(-q + Kokkos::sqrt(d), TF(1./3.));
        return u - TF(1.)/u;
    }


    // The same for the Henyey-Greenstein phase function of asymmetry g, which stands
    // in for the cloud droplets' Mie phase function.
    KOKKOS_INLINE_FUNCTION
    TF henyey(const TF g, const TF r)
    {
        const TF a = (TF(1.) - g*g)*(TF(1.) - g*g);
        const TF b = TF(2.)*g*(TF(2.)*r*g + TF(1.) - g)*(TF(2.)*r*g + TF(1.) - g);
        const TF c = -g/TF(2.) - TF(1.)/(TF(2.)*g);
        return TF(-1.)*(a/b) - c;
    }


    // Optical depth to the next collision, from an exponential distribution.
    KOKKOS_INLINE_FUNCTION
    TF sample_tau(const TF r)
    {
        // The epsilon keeps the logarithm off zero.
        return TF(-1.)*Kokkos::log(-r + TF(1.) + eps());
    }


    // Which cell a coordinate falls in, clamped to the last one. Takes the reciprocal
    // of the cell size rather than the size itself: the walk asks this question at
    // every collision, and a multiplication is what a division would be lowered to
    // anyway, but only where the compiler can see the divisor is loop-invariant.
    // The cast rounds toward zero, which is the rounding this wants.
    KOKKOS_INLINE_FUNCTION
    int coord_to_index(const TF s, const TF ds_inv, const int ntot)
    {
        const int n = static_cast<int>(s*ds_inv);
        return n < ntot ? n : ntot - 1;
    }


    enum class Photon_kind { Direct, Diffuse };

    struct Photon
    {
        Vector<TF> position;
        Vector<TF> direction;
        Photon_kind kind;
    };


    // Everything the walk reads and writes, gathered so that a launch captures one
    // object. Views are handles, so this is cheap to copy to the device.
    struct Scene
    {
        Array_map_2d<const TF> k_ext;             // (nz, ncol) [1/m]
        Array_map_2d<const Optics_scat> scat;     // (nz, ncol)
        Array_map_3d<const TF> k_null;            // (kn_z, kn_y, kn_x) [1/m]
        Array_map_1d<const TF> sfc_alb;           // (ncol)

        // Photon counts. Written with atomics, since photons from any thread may land
        // in any cell.
        Array_map_1d<TF> tod_dn, tod_up, sfc_dir, sfc_dif, sfc_up;   // (ncol)
        Array_map_2d<TF> atmos_dir, atmos_dif;                       // (nz, ncol)

        Vector<int> grid_cells;
        Vector<TF> grid_d;
        Vector<TF> grid_d_inv;
        Vector<TF> grid_size;
        Vector<int> kn_grid;
        Vector<TF> kn_grid_d;
        Vector<TF> kn_grid_d_inv;

        Vector<TF> sun_direction;
        TF inc_dir = TF(0.);
        TF inc_dif = TF(0.);

        // The horizontal extent the quasi-random sequence covers is the next power of
        // two above the grid, since a Sobol pair maps onto a power-of-two lattice.
        // Draws that land outside the grid are thrown away, which is what makes the
        // photon count per pixel come out right for a grid that is not a power of two.
        //
        // Held as the shift that takes a 32-bit draw onto that lattice rather than as
        // the extent itself: the extent is a power of two, so the mapping is a shift,
        // and integer division is the one thing fast math does not help with.
        unsigned int qrng_shift_x = 31, qrng_shift_y = 31;

        Rand::Qrng_vectors qrng;

        KOKKOS_INLINE_FUNCTION
        int column(const int i, const int j) const { return i + j*grid_cells.x; }
    };


    // Put a new photon at the top of the domain, in a pixel drawn from the
    // quasi-random sequence and at a random point within it. photons_shot counts every
    // draw, including the ones thrown away for landing outside the grid.
    RTE3D_DEVICE_FUNCTION
    void reset_photon(
            Photon& photon, TF& weight, int& photons_shot, const int photons_to_shoot,
            const Scene& s, Rand::Qrng_2d& qrng, Rand::Rng& rng)
    {
        unsigned int rx, ry;
        int i, j;

        while (true)
        {
            qrng.next(rx, ry);
            i = static_cast<int>(rx >> s.qrng_shift_x);
            j = static_cast<int>(ry >> s.qrng_shift_y);

            ++photons_shot;
            if (i < s.grid_cells.x && j < s.grid_cells.y)
                break;
        }

        if (photons_shot >= photons_to_shoot)
            return;

        photon.position.x = (i + rng())*s.grid_d.x;
        photon.position.y = (j + rng())*s.grid_d.y;
        photon.position.z = s.grid_size.z;

        const TF diffuse_fraction = s.inc_dif / (s.inc_dir + s.inc_dif);
        if (rng() >= diffuse_fraction)
        {
            photon.direction = s.sun_direction;
            photon.kind = Photon_kind::Direct;
        }
        else
        {
            // Isotropic downward, sampled by the cosine law.
            const TF mu = Kokkos::sqrt(rng());
            const TF azimuth = TF(2.*M_PI)*rng();
            const TF sin_theta = Kokkos::sqrt(TF(1.) - mu*mu + eps());

            photon.direction.x = sin_theta*Kokkos::sin(azimuth);
            photon.direction.y = sin_theta*Kokkos::cos(azimuth);
            photon.direction.z = -mu;
            photon.kind = Photon_kind::Diffuse;
        }

        Kokkos::atomic_add(&s.tod_dn(s.column(i, j)), TF(1.));
        weight = TF(1.);
    }


    // Write out the absorbed weight a photon has piled up in one fine cell.
    //
    // The walk deposits at every collision, null ones included, and a photon takes
    // several of those in the same cell -- so the deposit is held in a register and
    // written with one atomic when the photon leaves the cell or changes kind, rather
    // than with an atomic per collision. Thesis 4.1.7. A cell that absorbs nothing
    // asks for no atomic at all.
    RTE3D_DEVICE_FUNCTION
    void flush_absorbed(
            const Scene& s, TF& absorbed, const int k, const int ij, const Photon_kind kind)
    {
        if (absorbed > TF(0.))
        {
            if (kind == Photon_kind::Direct)
                Kokkos::atomic_add(&s.atmos_dir(k, ij), absorbed);
            else
                Kokkos::atomic_add(&s.atmos_dif(k, ij), absorbed);

            absorbed = TF(0.);
        }
    }


    // Send the photon off in a new direction, cos_scat away from the old one and at a
    // random azimuth about it.
    RTE3D_DEVICE_FUNCTION
    void scatter(Photon& photon, const TF cos_scat, Rand::Rng& rng)
    {
        const TF sin_scat = Kokkos::max(TF(0.), Kokkos::sqrt(TF(1.) - cos_scat*cos_scat + eps()));

        // Any vector not parallel to the direction will do to build the frame; take
        // the axis the direction leans on least.
        Vector<TF> t1{TF(0.), TF(0.), TF(0.)};
        if (Kokkos::abs(photon.direction.x) < Kokkos::abs(photon.direction.y))
        {
            if (Kokkos::abs(photon.direction.x) < Kokkos::abs(photon.direction.z))
                t1.x = TF(1.);
            else
                t1.z = TF(1.);
        }
        else
        {
            if (Kokkos::abs(photon.direction.y) < Kokkos::abs(photon.direction.z))
                t1.y = TF(1.);
            else
                t1.z = TF(1.);
        }

        t1 = normalize(t1 - dot(t1, photon.direction)*photon.direction);
        const Vector<TF> t2 = cross(photon.direction, t1);

        const TF phi = TF(2.*M_PI)*rng();

        photon.direction = cos_scat*photon.direction
                + sin_scat*(Kokkos::sin(phi)*t1 + Kokkos::cos(phi)*t2);
        photon.kind = Photon_kind::Diffuse;
    }


    // Walk photons_to_shoot photons through the scene, from the top of the domain
    // until they leave it or their weight runs out.
    //
    // qrng_offset is the photon's index in the global quasi-random sequence, so that
    // the threads of one launch together cover a contiguous block of it; rng_seed is
    // its counterpart for the pseudo-random stream.
    template<bool independent_column>
    RTE3D_DEVICE_FUNCTION
    void trace_photons(
            const Scene& s, const int photons_to_shoot,
            const unsigned int qrng_offset, const unsigned int rng_seed)
    {
        Rand::Rng rng(rng_seed);
        Rand::Qrng_2d qrng(s.qrng, qrng_offset);

        // The nudge that carries a photon past a cell face it has just landed on.
        const TF s_min = Kokkos::max(s.grid_size.z,
                Kokkos::max(s.grid_size.y, s.grid_size.x)) * eps();

        Photon photon;
        TF weight = TF(0.);
        int photons_shot = -1;

        reset_photon(photon, weight, photons_shot, photons_to_shoot, s, qrng, rng);

        TF tau = TF(0.);
        TF d_max = TF(0.);
        TF k_ext_null = TF(0.);
        TF k_ext_null_inv = TF(0.);
        bool transition = false;
        int i_n = 0, j_n = 0, k_n = 0;

        // The fine cell the last collision fell in, its optical properties, the kind
        // of photon that was in it, and the weight absorbed there so far. Consecutive
        // collisions land in the same cell often enough -- null collisions above all
        // -- that both the loads and the atomic are worth holding back for. Thesis
        // 4.1.6 and 4.1.7. The cell's raw coefficients are cached, not the fractions
        // derived from them, since the fractions also depend on the coarse block and
        // the block can change while the fine cell does not.
        int c_k = -1, c_ij = -1;
        Optics_scat c_scat{};
        TF c_k_ext = TF(0.);
        Photon_kind c_kind = Photon_kind::Direct;
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
                    // The surface: score the incoming flux, reflect what the albedo
                    // keeps, and send it back up isotropically.
                    photon.position.z = eps();
                    d_max = TF(0.);

                    const int i = coord_to_index(photon.position.x, s.grid_d_inv.x, s.grid_cells.x);
                    const int j = coord_to_index(photon.position.y, s.grid_d_inv.y, s.grid_cells.y);
                    const int ij = s.column(i, j);

                    if (photon.kind == Photon_kind::Direct)
                        Kokkos::atomic_add(&s.sfc_dir(ij), weight);
                    else
                        Kokkos::atomic_add(&s.sfc_dif(ij), weight);

                    weight *= s.sfc_alb(ij);
                    Kokkos::atomic_add(&s.sfc_up(ij), weight);

                    if (weight < w_thres())
                        weight = (rng() > weight) ? TF(0.) : TF(1.);

                    if (weight > TF(0.))
                    {
                        const TF mu = Kokkos::sqrt(rng());
                        const TF azimuth = TF(2.*M_PI)*rng();
                        const TF sin_theta = Kokkos::sqrt(TF(1.) - mu*mu + eps());

                        photon.direction.x = sin_theta*Kokkos::sin(azimuth);
                        photon.direction.y = sin_theta*Kokkos::cos(azimuth);
                        photon.direction.z = mu;
                        photon.kind = Photon_kind::Diffuse;
                    }
                    else
                        reset_photon(photon, weight, photons_shot, photons_to_shoot, s, qrng, rng);
                }
                else if (photon.position.z >= s.grid_size.z)
                {
                    // Out of the top: score it and start a new photon.
                    d_max = TF(0.);

                    const int i = coord_to_index(photon.position.x, s.grid_d_inv.x, s.grid_cells.x);
                    const int j = coord_to_index(photon.position.y, s.grid_d_inv.y, s.grid_cells.y);
                    Kokkos::atomic_add(&s.tod_up(s.column(i, j)), weight);

                    reset_photon(photon, weight, photons_shot, photons_to_shoot, s, qrng, rng);
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

                // A new cell, or a new kind of photon in it, ends the accumulation
                // the last one was collecting; a new cell also ends the usefulness of
                // its cached optics.
                if (k != c_k || ij != c_ij)
                {
                    flush_absorbed(s, absorbed, c_k, c_ij, c_kind);

                    c_scat = s.scat(k, ij);
                    c_k_ext = s.k_ext(k, ij);
                    c_k = k;
                    c_ij = ij;
                    c_kind = photon.kind;
                }
                else if (photon.kind != c_kind)
                {
                    flush_absorbed(s, absorbed, c_k, c_ij, c_kind);
                    c_kind = photon.kind;
                }

                const Optics_scat scat = c_scat;
                const TF k_ext = c_k_ext;
                const TF k_sca_tot = scat.k_sca_gas + scat.k_sca_cld;

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

                        const bool by_cloud = rng()*k_sca_tot < scat.k_sca_cld;
                        const TF g = Kokkos::min(TF(1.) - eps(), scat.asy_cld);

                        // Henyey-Greenstein divides by g, so a cloud that happens to
                        // scatter isotropically gets the isotropic law directly.
                        const TF cos_scat = !by_cloud ? rayleigh(rng())
                                : (g > TF(1.e-6) ? henyey(g, rng()) : TF(2.)*rng() - TF(1.));

                        scatter(photon, cos_scat, rng);
                    }
                }
                else
                {
                    d_max = TF(0.);
                    reset_photon(photon, weight, photons_shot, photons_to_shoot, s, qrng, rng);
                }
            }
        }

        // What the last cell collected. A reset needs no flush of its own: the key is
        // the cell and the kind, not the photon, so a deposit left standing is only
        // ever added to by a photon that would have written to the same place.
        flush_absorbed(s, absorbed, c_k, c_ij, c_kind);
    }
}
