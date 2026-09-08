#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Kokkos_Core.hpp>

namespace py = pybind11;


#ifdef USE_SINGLE_PRECISION
using TF = float;
#else
using TF = double;
#endif

// Matches the Bool/Int of the reference C API (rte-kernels/api/rte_types.h), used
// only where we cross the ABI boundary in the correctness tests.
using Bool = signed char;
using Int = unsigned long long;


KOKKOS_INLINE_FUNCTION constexpr TF operator""_tf(const long double ld) { return static_cast<TF>(ld); }
KOKKOS_INLINE_FUNCTION constexpr TF operator""_tf(const unsigned long long ll) { return static_cast<TF>(ll); }


// Default execution space: Cuda/HIP for GPU builds, OpenMP or Serial for CPU.
#if defined(USECUDA)
using Default_exec = Kokkos::Cuda;
#elif defined(USEHIP)
using Default_exec = Kokkos::HIP;
#elif defined(KOKKOS_ENABLE_OPENMP)
using Default_exec = Kokkos::OpenMP;
#else
using Default_exec = Kokkos::Serial;
#endif

using Range_1d = Kokkos::MDRangePolicy<Kokkos::Rank<1, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using Range_2d = Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using Range_3d = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using Range_4d = Kokkos::MDRangePolicy<Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;


// Assert that the following loop carries no dependencies, so the compiler may
// vectorize it. Spelled differently by every compiler, and #pragma cannot appear in a
// macro body, hence _Pragma. Note the ordering: clang-based Intel compilers define
// __clang__, and clang itself defines __GNUC__, so the most specific test comes first.
//
// clang's assume_safety is the analogue of GCC's ivdep: vectorize(enable) alone is
// only a hint and does not assert independence.
//
// Because this is an explicit request rather than a hint, clang warns under
// -Wpass-failed when it cannot honour it. That is signal worth keeping: the solver
// kernels currently trip it, because each (igpt, icol) thread runs a whole sequential
// column recurrence and there is nothing left to vectorize across columns.
#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
    #define RTE3D_IVDEP _Pragma("ivdep")
#elif defined(__clang__)
    #define RTE3D_IVDEP _Pragma("clang loop vectorize(assume_safety)")
#elif defined(__GNUC__)
    #define RTE3D_IVDEP _Pragma("GCC ivdep")
#else
    #define RTE3D_IVDEP
#endif


// The parallel_for wrappers below bypass MDRangePolicy on CPU. On Serial/OpenMP,
// MDRangePolicy still adds enough loop-nest overhead to prevent the compiler from
// vectorizing the inner loop the way a hand-rolled loop nest does, so there we run
// the outer dim via RangePolicy and hand-write the inner loops with ivdep. Taken
// from MicroHH, where the win was measured.
//
// Throughout RTE the last (fastest-varying) dimension is the column, so the ivdep
// loop below is always the column loop, matching the vector-over-ncol structure of
// the reference kernels.
template<typename Kernel>
inline void parallel_for_1d(
        const std::string& name,
        const int64_t begin,
        const int64_t end,
        const Kernel& kernel)
{
    Kokkos::parallel_for(name, Kokkos::RangePolicy<Default_exec>(begin, end), kernel);
}


template<typename Kernel>
inline void parallel_for_2d(
        const std::string& name,
        const Kokkos::Array<int64_t, 2>& begin,
        const Kokkos::Array<int64_t, 2>& end,
        const Kernel& kernel)
{
    #ifdef USEGPU
    Kokkos::parallel_for(name, Range_2d(begin, end), kernel);
    #else
    const int istart = begin[1];
    const int iend = end[1];
    Kokkos::parallel_for(name,
        Kokkos::RangePolicy<Default_exec>(begin[0], end[0]),
        KOKKOS_LAMBDA(const int j)
        {
            RTE3D_IVDEP
            for (int i=istart; i<iend; ++i)
                kernel(j, i);
        });
    #endif
}


template<typename Kernel>
inline void parallel_for_3d(
        const std::string& name,
        const Kokkos::Array<int64_t, 3>& begin,
        const Kokkos::Array<int64_t, 3>& end,
        const Kernel& kernel)
{
    #ifdef USEGPU
    Kokkos::parallel_for(name, Range_3d(begin, end), kernel);
    #else
    const int jstart = begin[1];
    const int jend = end[1];
    const int istart = begin[2];
    const int iend = end[2];
    Kokkos::parallel_for(name,
        Kokkos::RangePolicy<Default_exec>(begin[0], end[0]),
        KOKKOS_LAMBDA(const int k)
        {
            for (int j=jstart; j<jend; ++j)
                RTE3D_IVDEP
                for (int i=istart; i<iend; ++i)
                    kernel(k, j, i);
        });
    #endif
}


template<typename Kernel>
inline void parallel_for_4d(
        const std::string& name,
        const Kokkos::Array<int64_t, 4>& begin,
        const Kokkos::Array<int64_t, 4>& end,
        const Kernel& kernel)
{
    #ifdef USEGPU
    Kokkos::parallel_for(name, Range_4d(begin, end), kernel);
    #else
    const int kstart = begin[1];
    const int kend = end[1];
    const int jstart = begin[2];
    const int jend = end[2];
    const int istart = begin[3];
    const int iend = end[3];
    Kokkos::parallel_for(name,
        Kokkos::RangePolicy<Default_exec>(begin[0], end[0]),
        KOKKOS_LAMBDA(const int n)
        {
            for (int k=kstart; k<kend; ++k)
                for (int j=jstart; j<jend; ++j)
                    RTE3D_IVDEP
                    for (int i=istart; i<iend; ++i)
                        kernel(n, k, j, i);
        });
    #endif
}


// Column-wise solver pattern: independent work per (igpt, icol) with a sequential
// recurrence over layers inside the kernel. Used by every RTE solver. Split out
// from parallel_for_2d so the CPU/GPU trade-off for the recurrence can be tuned in
// one place without touching the physics.
template<typename Kernel>
inline void parallel_for_gpt_col(
        const std::string& name,
        const int ngpt,
        const int ncol,
        const Kernel& kernel)
{
    parallel_for_2d(name, {0, 0}, {ngpt, ncol}, kernel);
}


// These are the regular array types used for computation. C / Python style indexing,
// assumed not to overlap in memory with other arrays. Convention throughout RTE:
// the g-point is the slowest-varying dimension and the column the fastest, so
// tau(ngpt, nlay, ncol) has exactly the memory layout of the reference Fortran
// tau(ncol, nlay, ngpt).
template<typename T>
using Array_1d = Kokkos::View<T*, Kokkos::LayoutRight, Default_exec::memory_space, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template<typename T>
using Array_2d = Kokkos::View<T**, Kokkos::LayoutRight, Default_exec::memory_space, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template<typename T>
using Array_3d = Kokkos::View<T***, Kokkos::LayoutRight, Default_exec::memory_space, Kokkos::MemoryTraits<Kokkos::Restrict>>;

// The RRTMGP k-distribution tables need up to six dimensions.
template<typename T>
using Array_4d = Kokkos::View<T****, Kokkos::LayoutRight, Default_exec::memory_space, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template<typename T>
using Array_5d = Kokkos::View<T*****, Kokkos::LayoutRight, Default_exec::memory_space, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template<typename T>
using Array_6d = Kokkos::View<T******, Kokkos::LayoutRight, Default_exec::memory_space, Kokkos::MemoryTraits<Kokkos::Restrict>>;

// These are the array types that map to existing memory. We assume no overlapping maps.
template<typename T>
using Array_map_1d = Kokkos::View<T*, Kokkos::LayoutRight, Default_exec::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;

template<typename T>
using Array_map_2d = Kokkos::View<T**, Kokkos::LayoutRight, Default_exec::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;

template<typename T>
using Array_map_3d = Kokkos::View<T***, Kokkos::LayoutRight, Default_exec::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;

// Pinned on GPU so device <-> host transfers are fast and async-capable.
#if defined(USECUDA)
using Host_pinned_space = Kokkos::CudaHostPinnedSpace;
#elif defined(USEHIP)
using Host_pinned_space = Kokkos::HIPHostPinnedSpace;
#else
using Host_pinned_space = Kokkos::HostSpace;
#endif

// These are the array types used for work always done on the host, like initialization and IO.
template<typename T>
using Array_1d_h = Kokkos::View<T*, Kokkos::LayoutRight, Host_pinned_space, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template<typename T>
using Array_2d_h = Kokkos::View<T**, Kokkos::LayoutRight, Host_pinned_space, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template<typename T>
using Array_3d_h = Kokkos::View<T***, Kokkos::LayoutRight, Host_pinned_space, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template<typename T>
using Array_map_1d_h = Kokkos::View<T*, Kokkos::LayoutRight, Host_pinned_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;

template<typename T>
using Array_map_2d_h = Kokkos::View<T**, Kokkos::LayoutRight, Host_pinned_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;

template<typename T>
using Array_map_3d_h = Kokkos::View<T***, Kokkos::LayoutRight, Host_pinned_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;

// Unmanaged maps onto plain host memory that we do not own, such as a numpy buffer.
// Deliberately Kokkos::HostSpace and not Host_pinned_space: a numpy allocation is
// ordinary malloc'd memory, and declaring it pinned on a GPU build would be a lie
// that breaks the async copy paths.
template<typename T>
using Array_map_1d_host = Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;

template<typename T>
using Array_map_2d_host = Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;

template<typename T>
using Array_map_3d_host = Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;

template<typename T>
using Array_map_4d_host = Kokkos::View<T****, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;


// This helper function creates a new host mirror using pinned memory if GPU is enabled.
template<typename View_type>
inline auto create_pinned_mirror_view(const View_type& view)
{
    #ifdef USEGPU
    return Kokkos::create_mirror_view(Host_pinned_space{}, view);
    #else
    return Kokkos::create_mirror_view(view);
    #endif
}


namespace Numpy
{
    // Numpy array accepted from Python: C-contiguous, and cast to TF if needed so a
    // test can hand us a float64 array in a single-precision build.
    template<typename T>
    using In = py::array_t<T, py::array::c_style | py::array::forcecast>;

    inline void check_rank(const py::buffer_info& info, const int rank, const std::string& name)
    {
        if (info.ndim != rank)
            throw std::invalid_argument(
                    "Argument '" + name + "' must be " + std::to_string(rank) + "-dimensional, got "
                    + std::to_string(info.ndim) + "-dimensional.");
    }

    // Copy a numpy array into a freshly allocated device View. The numpy shape is the
    // View shape one-to-one: no transposes anywhere, because gpt-outer row-major and
    // the reference's col-outer column-major are the same bytes.
    template<typename T>
    Array_1d<T> to_device_1d(const In<T>& a, const std::string& name)
    {
        const py::buffer_info info = a.request();
        check_rank(info, 1, name);

        Array_1d<T> view(Kokkos::view_alloc(name, Kokkos::WithoutInitializing), info.shape[0]);
        Array_map_1d_host<const T> host(static_cast<const T*>(info.ptr), info.shape[0]);
        Kokkos::deep_copy(view, host);

        return view;
    }

    template<typename T>
    Array_2d<T> to_device_2d(const In<T>& a, const std::string& name)
    {
        const py::buffer_info info = a.request();
        check_rank(info, 2, name);

        Array_2d<T> view(Kokkos::view_alloc(name, Kokkos::WithoutInitializing), info.shape[0], info.shape[1]);
        Array_map_2d_host<const T> host(static_cast<const T*>(info.ptr), info.shape[0], info.shape[1]);
        Kokkos::deep_copy(view, host);

        return view;
    }

    template<typename T>
    Array_3d<T> to_device_3d(const In<T>& a, const std::string& name)
    {
        const py::buffer_info info = a.request();
        check_rank(info, 3, name);

        Array_3d<T> view(Kokkos::view_alloc(name, Kokkos::WithoutInitializing), info.shape[0], info.shape[1], info.shape[2]);
        Array_map_3d_host<const T> host(static_cast<const T*>(info.ptr), info.shape[0], info.shape[1], info.shape[2]);
        Kokkos::deep_copy(view, host);

        return view;
    }

    template<typename T>
    Array_4d<T> to_device_4d(const In<T>& a, const std::string& name)
    {
        const py::buffer_info info = a.request();
        check_rank(info, 4, name);

        Array_4d<T> view(Kokkos::view_alloc(name, Kokkos::WithoutInitializing),
                         info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
        Array_map_4d_host<const T> host(static_cast<const T*>(info.ptr),
                                        info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
        Kokkos::deep_copy(view, host);

        return view;
    }

    // Copy a device View back into a new numpy array owning its own memory.
    template<typename View_type>
    py::array_t<typename View_type::non_const_value_type> from_device(const View_type& view)
    {
        using T = typename View_type::non_const_value_type;
        constexpr int N = View_type::rank;

        std::vector<ssize_t> shape(N);
        for (int i=0; i<N; ++i)
            shape[i] = view.extent(i);

        py::array_t<T> a(shape);

        auto mirror = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(mirror, view);

        std::memcpy(a.mutable_data(), mirror.data(), view.size()*sizeof(T));

        return a;
    }
}
