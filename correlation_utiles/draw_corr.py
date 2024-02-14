from correlation_utiles.utils_class import *
from correlation_utiles.utils_function import n_KTP_Kato,c
from correlation_utiles.defaults import *
from correlation_utiles.corr_calc import Corr_calc
from correlation_utiles.draw_corr_utils import save_results, trace_distace, total_variation_distance
import pickle
import numpy as onp 
import matplotlib.pyplot as plt


def get_observables(
        signal_out,
        idler_out,
        idler_vac, 
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

        return corr_calc.get_observables() # coincidence_rate, density_matrix, tomography_matrix

def draw_observables(
                observables,
                save_location='.',
                max_mode2 = 4
                ):
        '''
        observables - array of dim (2,3), where the first index indicate if the observable is prediction (0) or ground truth (1) and the second is which observable according to the order coincidence_rate, density_matrix, tomography_matrix
        save_location - location where the resultes will be saved   
        '''
        fig, ax = plt.subplots(3,2,dpi=200,figsize=[10,10])
        src = ["Prediction", "Ground truth"]

        # plotting 
        for i in [1,0]:
                # plotting correlation
                plt.subplot(3,2,i+1)
                ax[0][i].set_title(src[i])
                coincidence_rate = observables[i][0][0]
                coincidence_rate = coincidence_rate / np.sum(np.abs(coincidence_rate))
                coincidence_rate = coincidence_rate.reshape(2*max_mode2+1, 2*max_mode2+1)
                observables[i][0] = coincidence_rate
                loc = range(coincidence_rate.shape[0])
                ticks = [str(t)  for t in range(-max_mode2,max_mode2+1)]
                im = plt.imshow(coincidence_rate,vmin=0,vmax=np.max(observables[1][0])) # probability
                plt.xlabel(r'signal mode i')
                plt.ylabel(r'idle mode j')
                plt.xticks(loc,ticks)
                plt.yticks(loc,ticks)
                plt.colorbar(im)

                # plotting density matrix
                plt.subplot(3,2,i+3)
                density_matrix = observables[i][1]
                # density_matrix = density_matrix / np.trace(np.real(density_matrix)) # why only real?
                density_matrix = density_matrix / np.trace(density_matrix) # Might shoud be real
                observables[i][1] = density_matrix
                density_matrix_real = onp.real(density_matrix)
                density_matrix_imag = onp.imag(density_matrix)

                n = density_matrix.shape[0] // 2
                loc = range(density_matrix.shape[0])
                ticks = [r"$\left|-1,-1\right\rangle$",r"$\left|-1,\ \ 0\right\rangle$",r"$\left|-1,\ \ 1\right\rangle$",r"$\left|\ \ 0,-1\right\rangle$",r"$\left|\ \ 0,\ \ 0\right\rangle$",r"$\left|\ \ 0,\ \ 1\right\rangle$",r"$\left|\ \ 1,-1\right\rangle$",r"$\left|\ \ 1,\ \ 0\right\rangle$",r"$\left|\ \ 1,\ \ 1\right\rangle$"]
                # ticks = [str(t)  for t in range(-n,n+1)]

                max_I = np.max(np.abs(observables[1][1]))
                im = plt.imshow(density_matrix_real, vmin=-max_I,vmax=max_I)
                # plt.xlabel(r'signal mode i')
                # plt.ylabel(r'idle mode j')
                plt.xticks(loc,ticks, fontsize=3)
                plt.yticks(loc,ticks, fontsize=5) 
                plt.colorbar(im)

                plt.subplot(3,2,i+5)
                im = plt.imshow(density_matrix_imag, vmin=-max_I,vmax=max_I)
                # plt.xlabel(r'signal mode i')
                # plt.ylabel(r'idle mode j')
                plt.xticks(loc,ticks, fontsize=4)
                plt.yticks(loc,ticks, fontsize=5)
                plt.colorbar(im)

        ax[0][0].text(-3.5,1,"(a)",fontsize=14)
        ax[1][0].text(-3.5,1,"(b)",fontsize=14)
        ax[2][0].text(-3.5,1,"(c)",fontsize=14)

        plt.figure(fig)
        plt.savefig(f"{save_location}/observables.jpg")
        plt.close('all')

        # calculting errors
        with open(str(f"{save_location}/observables_err.txt"),"w") as file:
                coincidence_rate_pred = observables[0][0]
                coincidence_rate_grt = observables[1][0]
                mse_err = ((coincidence_rate_pred-coincidence_rate_grt)**2).mean()
                file.write(f"Coincidence rate mse error:{mse_err}\n")
                
                tvd_err = total_variation_distance(coincidence_rate_pred,coincidence_rate_grt)
                file.write(f"Coincidence rate Total variation distance:{tvd_err}\n")


                file.write(f"--------\n")
                density_matrix_pred = observables[0][1]
                density_matrix_grt = observables[1][1]
                mse_err = (np.abs((density_matrix_pred-density_matrix_grt)**2)).mean()
                file.write(f"Density matrix mse error:{mse_err}\n")
                td_err = trace_distace(density_matrix_pred,density_matrix_grt)
                file.write(f"Density matrix trace distance:{td_err}\n\n")
                
                mse_err = (np.abs((np.real(density_matrix_pred)-np.real(density_matrix_grt))**2)).mean()
                file.write(f"Density matrix real part mse error:{mse_err}\n\n")

                mse_err = (np.abs((np.imag(density_matrix_pred)-np.imag(density_matrix_grt))**2)).mean()
                file.write(f"Density matrix imag part mse error:{mse_err}\n\n")

        
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