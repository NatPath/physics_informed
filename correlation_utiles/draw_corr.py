from correlation_utiles.utils_class import *
from correlation_utiles.utils_function import n_KTP_Kato,c
from correlation_utiles.defaults import *
from correlation_utiles.corr_calc import Corr_calc
from correlation_utiles.draw_corr_utils import save_results
import pickle
import numpy as onp 


def draw_corr(
        signal_out,
        idler_out,
        idler_vac, 
        save_location = ".",
        calc_tomography = False, 
        coincidence_projection_max_mode1=1,
        coincidence_projection_max_mode2=4
        ):
        """
           print and save the coincidence corralation matrix, desity matrix and tomograph matrix.
           Args:
                signal_out - signal profile at the end of the crystal     
                idler_out - idler profile at the end of the crystal     
                idler_vac - idler vacum states profile at the end of the crystal  
                save_location - location where the resultes will be saved   
                calc_tomograph - bool, flag if to calc and print tomograph results
                coincidence_projection_max_mode1 - Highest modes of the matrices that will be printed
                coincidence_projection_max_mode2 - Highest modes of the matrices that will be printed
        """

        shape = Shape()
        params = Params()

        signal = Beam(lam=2*params.lam_pump, polarization=params.signal_polarization, T=params.Temperature, power=params.power_signal)
        idler = Beam(lam=SFG_idler_wavelength(params.lam_pump,signal.lam), polarization=params.idler_polarization, T=params.Temperature, power=params.power_idler)



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
                max_mode1 = coincidence_projection_max_mode1,
                max_mode2 = coincidence_projection_max_mode2,
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
                signal_k = signal.k,
                idler_k = idler.k,
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
                run_name = save_location,
                observable_vec = {COINCIDENCE_RATE: True, DENSITY_MATRIX: True, TOMOGRAPHY_MATRIX: calc_tomography},
                observables = observables,
                projection_coincidence_rate = projection_coincidence_rate,
                projection_tomography_matrix = projection_tomography_matrix,
                signal_w = signal.w,
                idler_w = idler.w
        )