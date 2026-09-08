! Test oracle only. Exposes the reduction inside ty_gas_optics_rrtmgp%load, which the
! reference's C API does not reach, so rte3d's Gas_optics::load can be compared against
! it. Not part of rte3d, and not built by rte3d's CMake.
!
! Python reads the coefficient file and passes the same raw arrays to both
! implementations, so no NetCDF is needed here.
module rte3d_shim
  use, intrinsic :: iso_c_binding
  use mo_rte_kind,           only: wp, wl
  use mo_gas_concentrations, only: ty_gas_concs
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp
  implicit none

  integer, parameter :: name_len = 32

  type(ty_gas_optics_rrtmgp), save :: k_dist
  type(ty_gas_concs),         save :: gas_concs

contains

  ! Unpack a flat buffer of n fixed-width names into a Fortran string array.
  function unpack_names(buf, n) result(names)
    character(kind=c_char), dimension(*), intent(in) :: buf
    integer,                              intent(in) :: n
    character(len=name_len), dimension(n) :: names
    integer :: i, j

    do i = 1, n
      names(i) = ''
      do j = 1, name_len
        names(i)(j:j) = buf((i-1)*name_len + j)
      end do
    end do
  end function unpack_names

  subroutine shim_load(ngas_file, nminor_abs, nminor_lower, nminor_upper, &
                       ngpt, nbnd, npres, ntemp, neta, nplancktemp, nfit, &
                       ncontrib_lower, ncontrib_upper, navail,            &
                       gas_names_buf, gas_minor_buf, identifier_minor_buf,&
                       minor_gases_lower_buf, minor_gases_upper_buf,      &
                       scaling_gas_lower_buf, scaling_gas_upper_buf,      &
                       avail_names_buf,                                   &
                       key_species, band2gpt, band_lims_wavenum,          &
                       press_ref, temp_ref, press_ref_trop, temp_ref_p, temp_ref_t, &
                       vmr_ref, kmajor, kminor_lower, kminor_upper,       &
                       minor_limits_gpt_lower, minor_limits_gpt_upper,    &
                       minor_scales_with_density_lower, minor_scales_with_density_upper, &
                       scale_by_complement_lower, scale_by_complement_upper, &
                       kminor_start_lower, kminor_start_upper,            &
                       totplnk, planck_frac, optimal_angle_fit, status)   &
                       bind(C, name="rte3d_shim_load")
    integer(c_int), intent(in) :: ngas_file, nminor_abs, nminor_lower, nminor_upper
    integer(c_int), intent(in) :: ngpt, nbnd, npres, ntemp, neta, nplancktemp, nfit
    integer(c_int), intent(in) :: ncontrib_lower, ncontrib_upper, navail
    character(kind=c_char), dimension(*), intent(in) :: gas_names_buf, gas_minor_buf
    character(kind=c_char), dimension(*), intent(in) :: identifier_minor_buf
    character(kind=c_char), dimension(*), intent(in) :: minor_gases_lower_buf, minor_gases_upper_buf
    character(kind=c_char), dimension(*), intent(in) :: scaling_gas_lower_buf, scaling_gas_upper_buf
    character(kind=c_char), dimension(*), intent(in) :: avail_names_buf
    integer(c_int), intent(in) :: key_species(2, 2, nbnd), band2gpt(2, nbnd)
    real(c_double), intent(in) :: band_lims_wavenum(2, nbnd)
    real(c_double), intent(in) :: press_ref(npres), temp_ref(ntemp)
    real(c_double), intent(in) :: press_ref_trop, temp_ref_p, temp_ref_t
    real(c_double), intent(in) :: vmr_ref(2, ngas_file+1, ntemp)
    real(c_double), intent(in) :: kmajor(ntemp, neta, npres+1, ngpt)
    ! Note the ordering: load() takes kminor with the contributor axis fastest, but
    ! stores its reduced result with the contributor axis slowest. rte3d uses the
    ! latter throughout, so tests/shim.py transposes on the way in only.
    real(c_double), intent(in) :: kminor_lower(ncontrib_lower, neta, ntemp)
    real(c_double), intent(in) :: kminor_upper(ncontrib_upper, neta, ntemp)
    integer(c_int), intent(in) :: minor_limits_gpt_lower(2, nminor_lower)
    integer(c_int), intent(in) :: minor_limits_gpt_upper(2, nminor_upper)
    integer(c_int), intent(in) :: minor_scales_with_density_lower(nminor_lower)
    integer(c_int), intent(in) :: minor_scales_with_density_upper(nminor_upper)
    integer(c_int), intent(in) :: scale_by_complement_lower(nminor_lower)
    integer(c_int), intent(in) :: scale_by_complement_upper(nminor_upper)
    integer(c_int), intent(in) :: kminor_start_lower(nminor_lower), kminor_start_upper(nminor_upper)
    real(c_double), intent(in) :: totplnk(nplancktemp, nbnd)
    real(c_double), intent(in) :: planck_frac(ntemp, neta, npres+1, ngpt)
    real(c_double), intent(in) :: optimal_angle_fit(nfit, nbnd)
    integer(c_int), intent(out) :: status

    real(wp), dimension(:,:,:), allocatable :: rayl_lower, rayl_upper  ! left unallocated
    character(len=128) :: err
    character(len=name_len), dimension(navail) :: avail_names
    integer :: i

    avail_names = unpack_names(avail_names_buf, navail)

    err = gas_concs%init(avail_names)
    if (len_trim(err) /= 0) then
      status = 1
      return
    end if
    ! Values are irrelevant to the reduction; only the names matter.
    do i = 1, navail
      err = gas_concs%set_vmr(trim(avail_names(i)), 1.0e-6_wp)
      if (len_trim(err) /= 0) then
        status = 2
        return
      end if
    end do

    err = k_dist%load(gas_concs, &
            unpack_names(gas_names_buf, ngas_file), key_species, &
            band2gpt, band_lims_wavenum, &
            press_ref, press_ref_trop, temp_ref, temp_ref_p, temp_ref_t, &
            vmr_ref, kmajor, kminor_lower, kminor_upper, &
            unpack_names(gas_minor_buf, nminor_abs), &
            unpack_names(identifier_minor_buf, nminor_abs), &
            unpack_names(minor_gases_lower_buf, nminor_lower), &
            unpack_names(minor_gases_upper_buf, nminor_upper), &
            minor_limits_gpt_lower, minor_limits_gpt_upper, &
            logical(minor_scales_with_density_lower /= 0, wl), &
            logical(minor_scales_with_density_upper /= 0, wl), &
            unpack_names(scaling_gas_lower_buf, nminor_lower), &
            unpack_names(scaling_gas_upper_buf, nminor_upper), &
            logical(scale_by_complement_lower /= 0, wl), &
            logical(scale_by_complement_upper /= 0, wl), &
            kminor_start_lower, kminor_start_upper, &
            totplnk, planck_frac, rayl_lower, rayl_upper, optimal_angle_fit)

    status = 0
    if (len_trim(err) /= 0) status = 3
  end subroutine shim_load

  subroutine shim_dims(ngas, nflav, nminor_lower, ncontrib_lower, &
                       nminor_upper, ncontrib_upper) bind(C, name="rte3d_shim_dims")
    integer(c_int), intent(out) :: ngas, nflav, nminor_lower, ncontrib_lower
    integer(c_int), intent(out) :: nminor_upper, ncontrib_upper

    ngas           = size(k_dist%gas_names)
    nflav          = size(k_dist%flavor, 2)
    nminor_lower   = size(k_dist%kminor_start_lower)
    ncontrib_lower = size(k_dist%kminor_lower, 3)
    nminor_upper   = size(k_dist%kminor_start_upper)
    ncontrib_upper = size(k_dist%kminor_upper, 3)
  end subroutine shim_dims

  subroutine shim_get_main(flavor, gpoint_flavor, vmr_ref) bind(C, name="rte3d_shim_get_main")
    integer(c_int), intent(out) :: flavor(*), gpoint_flavor(*)
    real(c_double), intent(out) :: vmr_ref(*)

    flavor(1:size(k_dist%flavor))               = reshape(k_dist%flavor, [size(k_dist%flavor)])
    gpoint_flavor(1:size(k_dist%gpoint_flavor)) = reshape(k_dist%gpoint_flavor, [size(k_dist%gpoint_flavor)])
    vmr_ref(1:size(k_dist%vmr_ref))             = reshape(k_dist%vmr_ref, [size(k_dist%vmr_ref)])
  end subroutine shim_get_main

  subroutine shim_get_minor(side, limits, density, complement, &
                            idx_minor, idx_scaling, kminor_start, kminor) &
                            bind(C, name="rte3d_shim_get_minor")
    integer(c_int), intent(in)  :: side   ! 0 lower, 1 upper
    integer(c_int), intent(out) :: limits(*), density(*), complement(*)
    integer(c_int), intent(out) :: idx_minor(*), idx_scaling(*), kminor_start(*)
    real(c_double), intent(out) :: kminor(*)

    if (side == 0) then
      limits(1:size(k_dist%minor_limits_gpt_lower)) = &
        reshape(k_dist%minor_limits_gpt_lower, [size(k_dist%minor_limits_gpt_lower)])
      density(1:size(k_dist%minor_scales_with_density_lower)) = &
        merge(1, 0, k_dist%minor_scales_with_density_lower)
      complement(1:size(k_dist%scale_by_complement_lower)) = &
        merge(1, 0, k_dist%scale_by_complement_lower)
      idx_minor(1:size(k_dist%idx_minor_lower))       = k_dist%idx_minor_lower
      idx_scaling(1:size(k_dist%idx_minor_scaling_lower)) = k_dist%idx_minor_scaling_lower
      kminor_start(1:size(k_dist%kminor_start_lower)) = k_dist%kminor_start_lower
      kminor(1:size(k_dist%kminor_lower))             = reshape(k_dist%kminor_lower, [size(k_dist%kminor_lower)])
    else
      limits(1:size(k_dist%minor_limits_gpt_upper)) = &
        reshape(k_dist%minor_limits_gpt_upper, [size(k_dist%minor_limits_gpt_upper)])
      density(1:size(k_dist%minor_scales_with_density_upper)) = &
        merge(1, 0, k_dist%minor_scales_with_density_upper)
      complement(1:size(k_dist%scale_by_complement_upper)) = &
        merge(1, 0, k_dist%scale_by_complement_upper)
      idx_minor(1:size(k_dist%idx_minor_upper))       = k_dist%idx_minor_upper
      idx_scaling(1:size(k_dist%idx_minor_scaling_upper)) = k_dist%idx_minor_scaling_upper
      kminor_start(1:size(k_dist%kminor_start_upper)) = k_dist%kminor_start_upper
      kminor(1:size(k_dist%kminor_upper))             = reshape(k_dist%kminor_upper, [size(k_dist%kminor_upper)])
    end if
  end subroutine shim_get_minor

end module rte3d_shim
