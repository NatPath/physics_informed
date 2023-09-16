from correlation_utiles.utils_class import *
from correlation_utiles.utils_function import n_KTP_Kato,c
from correlation_utiles.defaults import *
from correlation_utiles.corr_calc import corr_calc
from correlation_utiles.draw_corr_utils import save_results
import pickle
import numpy as onp 

# get fields from spdc-pino-net

calc_tomography = False
datapath = "/home/dor-hay.sha/project/data/spdc/fixed_pump_N-100_seed-1701.bin"
# datapath = '/home/dor-hay.sha/project/data/spdc/single_mode-(0,3)_N-100_seed-123.bin'
with open(file=datapath,mode="rb") as file:
    data = pickle.load(file)

# dict = ["pump","signal_vac", "idler_vac", "signal_out", "idler_out"] # Order of the fields
idler_vac = data["fields"][:,-3,:,:,-1]
signal_out= data["fields"][:,-2,:,:,-1]
idler_out = data["fields"][:,-1,:,:,-1]
signal_k =  data["k_signal"].item() 
idler_k =  data["k_idler"].item() 

shape = Shape()
params = Params()

projection_coincidence_rate = Projection_coincidence_rate(
        waist_pump0= params.waist_pump0,
        signal_wavelength= params.lam_signal,
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
        signal_wavelength= params.lam_signal,
        crystal_x=shape.x,
        crystal_y=shape.y,
        temperature = params.Temperature,
        ctype = n_KTP_Kato,
        polarization = params.tomography_projection_polarization,
        z = params.tomography_projection_z,
        projection_basis = params.tomography_projection_basis,
        max_mode1 = params.tomography_projection_max_mode1,
        max_mode2 = params.tomography_projection_max_mode2,
        waist = params.tomography_projection_waist,
        wavelength = params.tomography_projection_wavelength,
        tau = params.tau,
        relative_phase = [0, onp.pi, 3 * (onp.pi / 2), onp.pi / 2],
        tomography_quantum_state =  'qutrit',

)



corr_calc = Corr_calc(
            idler_out = idler_out,
            idler_vac = idler_vac,
            signal_out = signal_out,
            signal_k = signal_k,
            idler_k = idler_k,
            shape  = shape,
            projection_coincidence_rate = projection_coincidence_rate,
            projection_tomography_matrix = projection_tomography_matrix,
            coincidence_rate_observable = True,
            density_matrix_observable = True,
            tomography_matrix_observable = calc_tomography,
            coupling_inefficiencies = False,
    
)

observables = corr_calc.get_observables()

save_results(
        run_name = "single_mode-(0,3)_N-100_seed-123",
        observable_vec = {COINCIDENCE_RATE: True, DENSITY_MATRIX: True, TOMOGRAPHY_MATRIX: calc_tomography},
        observables = observables,
        projection_coincidence_rate = projection_coincidence_rate,
        projection_tomography_matrix = projection_tomography_matrix,
        signal_w = 2 * np.pi * c / (2 * params.lam_signal),
        idler_w = 2 * np.pi * c / (2 * params.lam_idler)
)