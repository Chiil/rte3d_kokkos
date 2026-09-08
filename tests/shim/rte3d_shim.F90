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
  use mo_cloud_optics_rrtmgp, only: ty_cloud_optics_rrtmgp
  use mo_optical_props,      only: ty_optical_props_1scl, ty_optical_props_2str
  implicit none

  integer, parameter :: name_len = 32

  type(ty_gas_optics_rrtmgp), save :: k_dist
  type(ty_gas_concs),         save :: gas_concs
  type(ty_cloud_optics_rrtmgp), save :: cloud_optics

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

  ! --- cloud optics -----------------------------------------------------------
  !
  ! ty_cloud_optics_rrtmgp%cloud_optics has no bind(C) entry point either. The tables
  ! come in as arrays rather than a file, so this needs no NetCDF and is unaffected by
  ! the radice/diamice rename in newer coefficient files.

  subroutine shim_cloud_optics(nspec, nsize_liq, nsize_ice, nrghice, icergh, &
                               ncol, nlay, two_stream,                       &
                               band_lims_wvn,                                &
                               radliq_lwr, radliq_upr, radice_lwr, radice_upr, &
                               extliq, ssaliq, asyliq, extice, ssaice, asyice, &
                               clwp, ciwp, reliq, reice,                     &
                               tau, ssa, g, status) bind(C, name="rte3d_shim_cloud_optics")
    integer(c_int), intent(in) :: nspec, nsize_liq, nsize_ice, nrghice, icergh
    integer(c_int), intent(in) :: ncol, nlay, two_stream
    real(c_double), intent(in) :: band_lims_wvn(2, nspec)
    real(c_double), intent(in) :: radliq_lwr, radliq_upr, radice_lwr, radice_upr
    real(c_double), intent(in) :: extliq(nsize_liq, nspec), ssaliq(nsize_liq, nspec)
    real(c_double), intent(in) :: asyliq(nsize_liq, nspec)
    real(c_double), intent(in) :: extice(nsize_ice, nspec, nrghice)
    real(c_double), intent(in) :: ssaice(nsize_ice, nspec, nrghice)
    real(c_double), intent(in) :: asyice(nsize_ice, nspec, nrghice)
    real(c_double), intent(in) :: clwp(ncol, nlay), ciwp(ncol, nlay)
    real(c_double), intent(in) :: reliq(ncol, nlay), reice(ncol, nlay)
    real(c_double), intent(out) :: tau(*), ssa(*), g(*)
    integer(c_int), intent(out) :: status

    type(ty_optical_props_1scl) :: props_1scl
    type(ty_optical_props_2str) :: props_2str
    character(len=128) :: err
    integer :: n

    status = 0
    n = ncol*nlay*nspec

    ! The saved object may already hold tables from a previous call, and load()
    ! allocates unconditionally.
    call cloud_optics%finalize()

    err = cloud_optics%load(band_lims_wvn, radliq_lwr, radliq_upr, radice_lwr, radice_upr, &
                            extliq, ssaliq, asyliq, extice, ssaice, asyice)
    if (len_trim(err) /= 0) then
      status = 1
      return
    end if

    err = cloud_optics%set_ice_roughness(icergh)
    if (len_trim(err) /= 0) then
      status = 2
      return
    end if

    if (two_stream /= 0) then
      err = props_2str%init(band_lims_wvn)
      if (len_trim(err) == 0) err = props_2str%alloc_2str(ncol, nlay)
      if (len_trim(err) == 0) err = cloud_optics%cloud_optics(clwp, ciwp, reliq, reice, props_2str)
      if (len_trim(err) /= 0) then
        status = 3
        return
      end if
      tau(1:n) = reshape(props_2str%tau, [n])
      ssa(1:n) = reshape(props_2str%ssa, [n])
      g  (1:n) = reshape(props_2str%g,   [n])
    else
      err = props_1scl%init(band_lims_wvn)
      if (len_trim(err) == 0) err = props_1scl%alloc_1scl(ncol, nlay)
      if (len_trim(err) == 0) err = cloud_optics%cloud_optics(clwp, ciwp, reliq, reice, props_1scl)
      if (len_trim(err) /= 0) then
        status = 4
        return
      end if
      tau(1:n) = reshape(props_1scl%tau, [n])
    end if
  end subroutine shim_cloud_optics

end module rte3d_shim
