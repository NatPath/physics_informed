COINCIDENCE_RATE = 'coincidence_rate'
DENSITY_MATRIX = 'density_matrix'
TOMOGRAPHY_MATRIX = 'tomography_matrix'
REAL = 'real'
IMAG = 'imag'
QUBIT = 'qubit'
QUTRIT = 'qutrit'
qubit_projection_n_state2 = 6
qubit_tomography_dimensions = 2
qutrit_projection_n_state2 = 15
qutrit_tomography_dimensions = 3

SFG_idler_wavelength    = lambda lambda_p, lambda_s: lambda_p * lambda_s / (lambda_s - lambda_p)

class Params():
        def __init__(
            self,
            coupling_inefficiencies: bool = False,
            tau: float = 1e-9,
            SMF_waist: float = 2.18e-6,
            pump_basis: str = 'LG',
            pump_max_mode1: int = 5,
            pump_max_mode2: int = 1,
            initial_pump_waist: str = 'waist_pump0',
            crystal_basis: str = 'LG',
            crystal_max_mode1: int = 5,
            crystal_max_mode2: int = 2,
            initial_crystal_waist: str = 'r_scale0',
            lam_pump: float = 405e-9,
            crystal_str: str = 'ktp',
            power_pump: float = 1e-3,
            waist_pump0: float = 40e-6,
            r_scale0: float = 40e-6,
            dx: float = 4e-6,
            dy: float = 4e-6,
            dz: float = 10e-6,
            maxX: float = 120e-6,
            maxY: float = 120e-6,
            maxZ: float = 1e-4,
            R: float = 0.1,
            Temperature: float = 50,
            pump_polarization: str = 'y',
            signal_polarization: str = 'y',
            idler_polarization: str = 'z',
            dk_offset: float = 1.,
            power_signal: float = 1.,
            power_idler: float = 1.,
            coincidence_projection_basis: str = 'LG',
            coincidence_projection_max_mode1: int = 1,
            coincidence_projection_max_mode2: int = 4,
            coincidence_projection_waist: float = None,
            coincidence_projection_wavelength: float = None,
            coincidence_projection_polarization: str = 'y',
            coincidence_projection_z: float = 0.,
            tomography_projection_basis: str = 'LG',
            tomography_projection_max_mode1: int = 1,
            tomography_projection_max_mode2: int = 1,
            tomography_projection_waist: float = None,
            tomography_projection_wavelength: float = None,
            tomography_projection_polarization: str = 'y',
            tomography_projection_z: float = 0.,
            tomography_quantum_state: str = 'qutrit'
        ):
              
            self.coupling_inefficiencies=coupling_inefficiencies
            self.tau=tau
            self.SMF_waist=SMF_waist
            self.pump_basis=pump_basis
            self.pump_max_mode1=pump_max_mode1
            self.pump_max_mode2=pump_max_mode2
            self.initial_pump_waist=initial_pump_waist
            self.crystal_basis=crystal_basis
            self.crystal_max_mode1=crystal_max_mode1
            self.crystal_max_mode2=crystal_max_mode2
            self.initial_crystal_waist=initial_crystal_waist
            self.lam_pump=lam_pump
            self.crystal_str=crystal_str
            self.power_pump=power_pump
            self.waist_pump0=waist_pump0
            self.r_scale0=r_scale0
            self.dx=dx
            self.dy=dy
            self.dz=dz
            self.maxX=maxX
            self.maxY=maxY
            self.maxZ=maxZ
            self.R=R
            self.Temperature=Temperature
            self.pump_polarization=pump_polarization
            self.signal_polarization=signal_polarization
            self.idler_polarization=idler_polarization
            self.dk_offset=dk_offset
            self.power_signal=power_signal
            self.power_idler=power_idler
            self.coincidence_projection_basis=coincidence_projection_basis
            self.coincidence_projection_max_mode1=coincidence_projection_max_mode1
            self.coincidence_projection_max_mode2=coincidence_projection_max_mode2
            self.coincidence_projection_waist=coincidence_projection_waist
            self.coincidence_projection_wavelength=coincidence_projection_wavelength
            self.coincidence_projection_polarization=coincidence_projection_polarization
            self.coincidence_projection_z=coincidence_projection_z
            self.tomography_projection_basis=tomography_projection_basis
            self.tomography_projection_max_mode1=tomography_projection_max_mode1
            self.tomography_projection_max_mode2=tomography_projection_max_mode2
            self.tomography_projection_waist=tomography_projection_waist
            self.tomography_projection_wavelength=tomography_projection_wavelength
            self.tomography_projection_polarization=tomography_projection_polarization
            self.tomography_projection_z=tomography_projection_z
            self.tomography_quantum_state=tomography_quantum_state
            self.lam_signal = 2 * lam_pump
            self.lam_idler = SFG_idler_wavelength(self.lam_pump, self.lam_signal)


"""

    Parameters
    ----------
    pump_basis: Pump's construction basis method
                Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
    pump_max_mode1: Maximum value of first mode of the 2D pump basis
    pump_max_mode2: Maximum value of second mode of the 2D pump basis
    initial_pump_waist: defines the initial values of waists for pump basis function
                        can be: waist_pump0- will be set according to waist_pump0
                                load- will be loaded from np.arrays defined under path: pump_waists_path
                                with name: PumpWaistCoeffs.npy
    crystal_basis: Crystal's construction basis method
                   Can be:
                   None / FT (Fourier-Taylor) / FB (Fourier-Bessel) / LG (Laguerre-Gauss) / HG (Hermite-Gauss)
                   - if None, the crystal will contain NO hologram
    crystal_max_mode1: Maximum value of first mode of the 2D crystal basis
    crystal_max_mode2: Maximum value of second mode of the 2D crystal basis
    initial_crystal_waist: defines the initial values of waists for crystal basis function
                           can be: r_scale0- will be set according to r_scale0
                                   load- will be loaded from np.arrays defined under path: crystal_waists_path
                                         with name: CrystalWaistCoeffs.npy
    lam_pump: Pump wavelength
    crystal_str: Crystal type. Can be: KTP or MgCLN
    power_pump: Pump power [watt]
    waist_pump0: waists of the pump basis functions.
                 -- If None, waist_pump0 = sqrt(maxZ / self.pump_k)
    r_scale0: effective waists of the crystal basis functions.
              -- If None, r_scale0 = waist_pump0
    dx: transverse resolution in x [m]
    dy: transverse resolution in y [m]
    dz: longitudinal resolution in z [m]
    maxX: Transverse cross-sectional size from the center of the crystal in x [m]
    maxY: Transverse cross-sectional size from the center of the crystal in y [m]
    maxZ: Crystal's length in z [m]
    R: distance to far-field screen [m]
    Temperature: crystal's temperature [Celsius Degrees]
    pump_polarization: Polarization of the pump beam
    signal_polarization: Polarization of the signal beam
    idler_polarization: Polarization of the idler beam
    dk_offset: delta_k offset
    power_signal: Signal power [watt]
    power_idler: Idler power [watt]

    coincidence_projection_basis: represents the projective basis for calculating the coincidence rate observable
                                  of the interaction. Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
    coincidence_projection_max_mode1: Maximum value of first mode of the 2D projection basis for coincidence rate
    coincidence_projection_max_mode2: Maximum value of second mode of the 2D projection basis for coincidence rate
    coincidence_projection_waist: waists of the projection basis functions of coincidence rate.
                                  if None, np.sqrt(2) * waist_pump0 is used
    coincidence_projection_wavelength: wavelength for generating projection basis of coincidence rate.
                                       if None, the signal wavelength is used
    coincidence_projection_polarization: polarization for calculating effective refractive index
    coincidence_projection_z: projection longitudinal position
    tomography_projection_basis: represents the projective basis for calculating the tomography matrix & density matrix
                                    observables of the interaction. Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
    tomography_projection_max_mode1: Maximum value of first mode of the 2D projection basis for tomography matrix &
                                        density matrix
    tomography_projection_max_mode2: Maximum value of second mode of the 2D projection basis for tomography matrix &
                                        density matrix
    tomography_projection_waist: waists of the projection basis functions of tomography matrix & density matrix
                                  if None, np.sqrt(2) * waist_pump0 is used
    tomography_projection_wavelength: wavelength for generating projection basis of tomography matrix & density matrix.
                                       if None, the signal wavelength is used
    tomography_projection_polarization: polarization for calculating effective refractive index
    tomography_projection_z: projection longitudinal position
    tomography_relative_phase: The relative phase between the mutually unbiased bases (MUBs) states
   tomography_quantum_state: the current quantum state we calculate it tomography matrix.
                               currently we support: qubit/qutrit
    tau: coincidence window [nano sec]
    SMF_waist: signal/idler beam radius at single mode fibre
    -------
 """
        