from utils_class import *
from utils_function import n_KTP_Kato
from defaults import *
from corr_calc import corr_calc

# get fields from spdc-pino-net
fields = None
pump, signal, idler = fields


shape = Shape()

projection_coincidence_rate = Projection_coincidence_rate(
        waist_pump0= params.waist_pump0,
        signal_wavelength= (2 * params.lam_pump),
        crystal_x=shape.x,
        crystal_y=shape.y,
        temperature = params.Temperature,
        ctype = n_KTP_Kato,
        polarization = params.coincidence_projection_polarization,
        z = params.coincidence_projection_z,
        projection_basis = params.coincidence_projection_basis,
        max_mode1 = params.coincidence_projection_max_mode1,
        max_mode2 = params.coincidence_projection_max_mode2,
        waist = params.coincidence_projection_waist,
        wavelength = params.coincidence_projection_wavelength,
        tau = params.tau,
        SMF_waist = params.SMF_waist,
    )

projection_tomography_matrix = Projection_tomography_matrix(
        waist_pump0= params.waist_pump0,
        signal_wavelength= (2 * params.lam_pump),
        crystal_x=shape.x,
        crystal_y=shape.y,
        temperature = params.Temperature,
        ctype = n_KTP_Kato,
        polarization = params.coincidence_projection_polarization,
        z = params.coincidence_projection_z,
        projection_basis = params.coincidence_projection_basis,
        max_mode1 = params.coincidence_projection_max_mode1,
        max_mode2 = params.coincidence_projection_max_mode2,
        waist = params.coincidence_projection_waist,
        wavelength = params.coincidence_projection_wavelength,
        tau = params.tau,
        # relative_phase: List[Union[Union[int, float], Any]] = None,
        # tomography_quantum_state: str = None,

)



corr_calc = corr_calc(
            pump = pump,
            signal = signal,
            idler = idler,
            projection_coincidence_rate = projection_coincidence_rate,
            projection_tomography_matrix = projection_tomography_matrix,
            coincidence_rate_observable = True,
            density_matrix_observable = True,
            tomography_matrix_observable = False,
            coupling_inefficiencies = False,
    
)

corr_calc.get_observables()