#pragma once

#include <limits>

#include "random.h"
#include "raytracer.h"
#include "types.h"


// What the shortwave and longwave photon walks have in common: vector arithmetic, the
// phase functions, the free-path sampling and the grid lookup.
//
// The two walks differ in where photons come from and what they score, not in how they
// move, so everything geometric lives here and both raytracer_kernels.h and
// raytracer_lw_kernels.h build on it. Ported from raytracer_functions.h in
// rte-rrtmgp-cpp, which the two reference kernels share in the same way.
namespace Rt_common
{
    using Raytracer::Optics_cell;
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


    // A direction drawn from the cosine law about the vertical, which is what an
    // isotropically radiating horizontal surface emits and what a Lambertian one
    // reflects. sign is +1 for upward and -1 for downward.
    RTE3D_DEVICE_FUNCTION
    Vector<TF> cosine_direction(const TF sign, Rand::Rng& rng)
    {
        const TF mu = Kokkos::sqrt(rng());
        const TF azimuth = TF(2.*M_PI)*rng();
        const TF sin_theta = Kokkos::sqrt(TF(1.) - mu*mu + eps());

        return Vector<TF>{sin_theta*Kokkos::sin(azimuth),
                          sin_theta*Kokkos::cos(azimuth),
                          sign*mu};
    }


    // A direction cos_scat away from the old one, at a random azimuth about it.
    RTE3D_DEVICE_FUNCTION
    Vector<TF> scatter_direction(const Vector<TF>& direction, const TF cos_scat, Rand::Rng& rng)
    {
        const TF sin_scat = Kokkos::max(TF(0.), Kokkos::sqrt(TF(1.) - cos_scat*cos_scat + eps()));

        // Any vector not parallel to the direction will do to build the frame; take
        // the axis the direction leans on least.
        Vector<TF> t1{TF(0.), TF(0.), TF(0.)};
        if (Kokkos::abs(direction.x) < Kokkos::abs(direction.y))
        {
            if (Kokkos::abs(direction.x) < Kokkos::abs(direction.z))
                t1.x = TF(1.);
            else
                t1.z = TF(1.);
        }
        else
        {
            if (Kokkos::abs(direction.y) < Kokkos::abs(direction.z))
                t1.y = TF(1.);
            else
                t1.z = TF(1.);
        }

        t1 = normalize(t1 - dot(t1, direction)*direction);
        const Vector<TF> t2 = cross(direction, t1);

        const TF phi = TF(2.*M_PI)*rng();

        return cos_scat*direction + sin_scat*(Kokkos::sin(phi)*t1 + Kokkos::cos(phi)*t2);
    }


    // Which of the two scatterers deflected the photon, and by how much. The draw is
    // over the scattering coefficients of the cell; a cloud uses Henyey-Greenstein
    // with the asymmetry the RRTMGP tables give, a gas the Rayleigh law.
    RTE3D_DEVICE_FUNCTION
    TF sample_cos_scat(const Optics_cell& optics, const TF k_sca_tot, Rand::Rng& rng)
    {
        const bool by_cloud = rng()*k_sca_tot < optics.k_sca_cld;
        const TF g = Kokkos::min(TF(1.) - eps(), optics.asy_cld);

        // Henyey-Greenstein divides by g, so a cloud that happens to scatter
        // isotropically gets the isotropic law directly.
        return !by_cloud ? rayleigh(rng())
                : (g > TF(1.e-6) ? henyey(g, rng()) : TF(2.)*rng() - TF(1.));
    }
}
