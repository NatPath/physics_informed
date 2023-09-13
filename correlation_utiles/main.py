from utils import projection_coincidence_rate


projection_coincidence_rate = Projection_coincidence_rate(
        waist_pump0=params.waist_pump0,
        signal_wavelength=params.lam_signal,
        crystal_x=params.x,
        crystal_y=params.y,
        temperature=params.Temperature,
        ctype=params.ctype,
        polarization=coincidence_projection_polarization,
        z=coincidence_projection_z,
        projection_basis=coincidence_projection_basis,
        max_mode1=coincidence_projection_max_mode1,
        max_mode2=coincidence_projection_max_mode2,
        waist=coincidence_projection_waist,
        wavelength=coincidence_projection_wavelength,
        tau=tau,
        SMF_waist=SMF_waist,
    )