from abc import ABC
import jax.numpy as np

from defaults import QUBIT
from correlation_utiles.utils_function import DensMat
from correlation_utiles.utils_function import (
    projection_matrix_calc, projection_matrices_calc,
    decompose, fix_power, get_qubit_density_matrix, get_qutrit_density_matrix,
    coupling_inefficiency_calc_G2, coupling_inefficiency_calc_tomo,
)


class corr_calc(ABC):


    def __init__(
            self,
            pump,
            signal,
            idler,
            projection_coincidence_rate,
            projection_tomography_matrix,
            coincidence_rate_observable: bool = True,
            density_matrix_observable: bool = True,
            tomography_matrix_observable: bool = False,
            coupling_inefficiencies: bool = False,
    ):

        self.pump = pump
        self.signal = signal
        self.idler = idler
        self.projection_coincidence_rate = projection_coincidence_rate
        self.projection_tomography_matrix = projection_tomography_matrix
        self.coincidence_rate_observable = coincidence_rate_observable
        self.density_matrix_observable = density_matrix_observable
        self.tomography_matrix_observable = tomography_matrix_observable
        self.coupling_inefficiencies = coupling_inefficiencies

        self.N = None
        self.N_device = None



    

    def get_1st_order_projections(
            self,
            signal_out,
            idler_out,
            idler_vac,
            signal_out_back_prop,
            idler_out_back_prop,
            idler_vac_back_prop,
    ):
        """
        the function calculates first order correlation functions.
            According to  https://doi.org/10.1002/lpor.201900321

        Parameters
        ----------
        signal_out: the signal at the end of interaction
        idler_out: the idler at the end of interaction
        idler_vac: the idler vacuum state at the end of interaction

        Returns: first order correlation functions according to  https://doi.org/10.1002/lpor.201900321
        -------

        """

        coincidence_rate_projections, tomography_matrix_projections = None, None
        if self.coincidence_rate_observable:
            coincidence_rate_projections = self.decompose_and_get_projections(
                signal_out,
                idler_out,
                idler_vac,
                signal_out_back_prop,
                idler_out_back_prop,
                idler_vac_back_prop,
                self.projection_coincidence_rate.basis_arr,
                self.projection_coincidence_rate.projection_n_modes1,
                self.projection_coincidence_rate.projection_n_modes2
            )

        if self.tomography_matrix_observable or self.density_matrix_observable:
            tomography_matrix_projections = self.decompose_and_get_projections(
                signal_out,
                idler_out,
                idler_vac,
                signal_out_back_prop,
                idler_out_back_prop,
                idler_vac_back_prop,
                self.projection_tomography_matrix.basis_arr,
                self.projection_tomography_matrix.projection_n_state1,
                self.projection_tomography_matrix.projection_n_state2
            )

        return coincidence_rate_projections, tomography_matrix_projections

    def decompose_and_get_projections(
            self,
            signal_out,
            idler_out,
            idler_vac,
            signal_out_back_prop,
            idler_out_back_prop,
            idler_vac_back_prop,
            basis_arr,
            projection_n_1,
            projection_n_2
    ):
        """
        The function decompose the interacting fields onto selected basis array, and calculates first order
            correlation functions according to  https://doi.org/10.1002/lpor.201900321

        Parameters
        ----------
        signal_out
        idler_out
        idler_vac
        signal_out_back_prop
        idler_out_back_prop
        idler_vac_back_prop
        basis_arr
        projection_n_1
        projection_n_2

        Returns
        -------

        """

        signal_beam_decompose, idler_beam_decompose, idler_vac_decompose = \
            self.decompose(
                signal_out,
                idler_out,
                idler_vac,
                signal_out_back_prop,
                idler_out_back_prop,
                idler_vac_back_prop,
                basis_arr,
                projection_n_1,
                projection_n_2
            )

        G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger = projection_matrices_calc(
            signal_beam_decompose,
            idler_beam_decompose,
            idler_vac_decompose,
            self.N
        )

        return G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger

    def decompose(
            self,
            signal_out,
            idler_out,
            idler_vac,
            signal_out_back_prop,
            idler_out_back_prop,
            idler_vac_back_prop,
            basis_arr,
            projection_n_1,
            projection_n_2
    ):

        signal_beam_decompose = decompose(
            signal_out_back_prop,
            basis_arr
        ).reshape(
            self.N_device,
            projection_n_1,
            projection_n_2)

        idler_beam_decompose = decompose(
            idler_out_back_prop,
            basis_arr
        ).reshape(
            self.N_device,
            projection_n_1,
            projection_n_2)

        idler_vac_decompose = decompose(
            idler_vac_back_prop,
            basis_arr
        ).reshape(
            self.N_device,
            projection_n_1,
            projection_n_2)

        # say there are no higher modes by normalizing the power
        signal_beam_decompose = fix_power(signal_beam_decompose, signal_out)
        idler_beam_decompose = fix_power(idler_beam_decompose, idler_out)
        idler_vac_decompose = fix_power(idler_vac_decompose, idler_vac)

        return signal_beam_decompose, idler_beam_decompose, idler_vac_decompose

    def get_observables(
            self,
    ):
        coincidence_rate_projections, tomography_matrix_projections = \
            self.get_1st_order_projections(
                signal_out,
                idler_out,
                idler_vac,
                signal_out_back_prop,
                idler_out_back_prop,
                idler_vac_back_prop,


            )
        coincidence_rate, density_matrix, tomography_matrix = None, None, None

        if self.coincidence_rate_observable:
            coincidence_rate = projection_matrix_calc(
                *coincidence_rate_projections
            ).reshape(
                self.projection_coincidence_rate.projection_n_modes1 ** 2,
                self.projection_coincidence_rate.projection_n_modes2 ** 2
            )
            ## coupling inefficiences
            if self.coupling_inefficiencies:
                assert self.projection_coincidence_rate.projection_basis.lower() == 'lg', \
                    f'Only implemented for Laguerre-Gauss bases. ' \
                    f'We received {self.projection_coincidence_rate.projection_basis}'
                coincidence_rate = np.multiply(coupling_inefficiency_calc_G2(
                    self.signal.lam,
                    self.projection_coincidence_rate.SMF_waist,
                ), coincidence_rate
                )

        if self.tomography_matrix_observable or self.density_matrix_observable:
            tomography_matrix = projection_matrix_calc(
                *tomography_matrix_projections
            ).reshape(
                self.projection_tomography_matrix.projection_n_state1 ** 2,
                self.projection_tomography_matrix.projection_n_state2 ** 2)
            if self.coupling_inefficiencies:
                if self.projection_tomography_matrix.tomography_quantum_state is not QUBIT:
                    # in the case of qubit tomography, inefficiency factor is same for all modes
                    # in the tomography matrix
                    tomography_matrix = np.multiply(coupling_inefficiency_calc_tomo(
                        self.signal.lam,
                        self.projection_tomography_matrix.SMF_waist,
                    ), tomography_matrix
                    )

            if self.density_matrix_observable:
                densmat = DensMat(
                    self.projection_tomography_matrix.projection_n_state2,
                    self.projection_tomography_matrix.tomography_dimensions
                )

                if self.projection_tomography_matrix.tomography_quantum_state is QUBIT:
                    density_matrix = get_qubit_density_matrix(tomography_matrix,
                                                              densmat.masks,
                                                              densmat.rotation_mats,
                                                              ).reshape(
                        self.projection_tomography_matrix.tomography_dimensions ** 2,
                        self.projection_tomography_matrix.tomography_dimensions ** 2)
                else:
                    density_matrix = get_qutrit_density_matrix(tomography_matrix,
                                                               densmat.masks,
                                                               densmat.rotation_mats,
                                                               ).reshape(
                        self.projection_tomography_matrix.tomography_dimensions ** 2,
                        self.projection_tomography_matrix.tomography_dimensions ** 2)

        return coincidence_rate, density_matrix, tomography_matrix

