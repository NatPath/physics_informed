import sys
from abc import ABC
import jax.numpy as np
from jax import lax
from jax import jit
from defaults import qubit_projection_n_state2, \
    qubit_tomography_dimensions, qutrit_projection_n_state2, qutrit_tomography_dimensions
import math


from typing import Tuple, Dict, Any, List, Union
from spdc_inv.utils.utils import (
    HermiteBank, LaguerreBank, TomographyBankLG, TomographyBankHG
)
from spdc_inv.utils.defaults import QUBIT, QUTRIT
from spdc_inv.utils.defaults import qubit_projection_n_state2, \
    qubit_tomography_dimensions, qutrit_projection_n_state2, qutrit_tomography_dimensions

@jit
def project(projection_basis, beam_profile):
    """
    The function projects some state beam_profile onto given projection_basis
    Parameters
    ----------
    projection_basis: array of basis function
    beam_profile: beam profile (2d)

    Returns
    -------

    """
    Nxx2           = beam_profile.shape[1] ** 2
    N              = beam_profile.shape[0]
    Nh             = projection_basis.shape[0]
    projection     = (np.conj(projection_basis) * beam_profile).reshape(Nh, N, Nxx2).sum(2)
    normalization1 = np.abs(beam_profile ** 2).reshape(N, Nxx2).sum(1)
    normalization2 = np.abs(projection_basis ** 2).reshape(Nh, Nxx2).sum(1)
    projection     = projection / np.sqrt(normalization1[None, :] * normalization2[:, None])
    return projection


@jit
def decompose(beam_profile, projection_basis_arr):
    """
    Decompose a given beam profile into modes defined in the dictionary
    Parameters
    ----------
    beam_profile: beam profile (2d)
    projection_basis_arr: array of basis function

    Returns: beam profile as a decomposition of basis functions
    -------

    """
    projection = project(projection_basis_arr[:, None], beam_profile)
    return np.transpose(projection)


@jit
def fix_power(decomposed_profile, beam_profile):
    """
    Normalize power and ignore higher modes
    Parameters
    ----------
    decomposed_profile: the decomposed beam profile
    beam_profile: the original beam profile

    Returns a normalized decomposed profile
    -------

    """
    scale = np.sqrt(
        np.sum(beam_profile * np.conj(beam_profile), (1, 2))) / np.sqrt(
        np.sum(decomposed_profile * np.conj(decomposed_profile), (1, 2)))

    return decomposed_profile * scale[:, None, None]


@jit
def kron(a, b, multiple_devices: bool = False):
    """
    Calculates the kronecker product between two 2d tensors
    Parameters
    ----------
    a, b: 2d tensors
    multiple_devices: (True/False) whether multiple devices are used

    Returns the kronecker product
    -------

    """
    if multiple_devices:
        return lax.psum((a[:, :, None, :, None] * b[:, None, :, None, :]).sum(0), 'device')

    else:
        return (a[:, :, None, :, None] * b[:, None, :, None, :]).sum(0)


@jit
def projection_matrices_calc(a, b, c, N):
    """

    Parameters
    ----------
    a, b, c: the interacting fields
    N: Total number of interacting vacuum state elements

    Returns the projective matrices
    -------

    """
    G1_ss        = kron(np.conj(a), a) / N
    G1_ii        = kron(np.conj(b), b) / N
    G1_si        = kron(np.conj(b), a) / N
    G1_si_dagger = kron(np.conj(a), b) / N
    Q_si         = kron(c, a) / N
    Q_si_dagger  = kron(np.conj(a), np.conj(c)) / N

    return G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger


@jit
def projection_matrix_calc(G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger):
    """

    Parameters
    ----------
    G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger: the projective matrices
    Returns the 2nd order projective matrix
    -------

    """
    return (lax.psum(G1_ii, 'device') *
            lax.psum(G1_ss, 'device') +
            lax.psum(Q_si_dagger, 'device') *
            lax.psum(Q_si, 'device') +
            lax.psum(G1_si_dagger, 'device') *
            lax.psum(G1_si, 'device')
            ).real


# for coupling inefficiencies
@jit
def coupling_inefficiency_calc_G2(
        lam,
        SMF_waist,
        max_mode_l: int = 4,
        focal_length: float = 4.6e-3,
        SMF_mode_diam: float = 2.5e-6,
):
    waist = 46.07 * SMF_waist
    a_0 = np.sqrt(2) * lam * focal_length / (np.pi * waist)
    A = 2 / (1 + (SMF_mode_diam ** 2 / a_0 ** 2))
    B = 2 / (1 + (a_0 ** 2 / SMF_mode_diam ** 2))
    inef_coeff = np.zeros([2 * max_mode_l + 1, 2 * max_mode_l + 1])

    for l_i in range(-max_mode_l, max_mode_l + 1):
        inef_coeff_i = (math.factorial(abs(l_i)) ** 2) * (A ** (2 * abs(l_i) + 1) * B) / (math.factorial(2 * abs(l_i)))
        for l_s in range(-max_mode_l, max_mode_l + 1):
            inef_coeff_s = (math.factorial(abs(l_s)) ** 2) * (A ** (2 * abs(l_s) + 1) * B) / (
                math.factorial(2 * abs(l_s)))
            inef_coeff = inef_coeff.at[l_i + max_mode_l, l_s + max_mode_l].set((inef_coeff_i + inef_coeff_s))

    return inef_coeff.reshape(1, (2 * max_mode_l + 1) ** 2)


@jit
def coupling_inefficiency_calc_tomo(
        lam,
        SMF_waist,
        focal_length: float = 4.6e-3,
        SMF_mode_diam: float = 2.5e-6,
):
    waist = 46.07 * SMF_waist
    a_0 = np.sqrt(2) * lam * focal_length / (np.pi * waist)
    A = 2 / (1 + (SMF_mode_diam ** 2 / a_0 ** 2))
    B = 2 / (1 + (a_0 ** 2 / SMF_mode_diam ** 2))
    inef_coeff = np.zeros([qutrit_projection_n_state2, qutrit_projection_n_state2])

    for base_1 in range(qutrit_projection_n_state2):
        # azimuthal modes l = {-1, 0, 1} defined according to order of MUBs
        if base_1 == 0 or base_1 == 2:
            l_1 = 1
            inef_coeff_i = (math.factorial(abs(l_1)) ** 2) * (A ** (2 * abs(l_1) + 1) * B) / (
                math.factorial(2 * abs(l_1)))
        elif base_1 == 1:
            l_1 = 0
            inef_coeff_i = (math.factorial(abs(l_1)) ** 2) * (A ** (2 * abs(l_1) + 1) * B) / (
                math.factorial(2 * abs(l_1)))

        else:
            if base_1 < 7 or base_1 > 10:
                l_1, l_2 = 1, 0
            else:
                l_1, l_2 = 1, 1
            inef_coeff_1 = (math.factorial(abs(l_1)) ** 2) * (A ** (2 * abs(l_1) + 1) * B) / (
                math.factorial(2 * abs(l_1)))
            inef_coeff_2 = (math.factorial(abs(l_2)) ** 2) * (A ** (2 * abs(l_2) + 1) * B) / (
                math.factorial(2 * abs(l_2)))
            inef_coeff_i = 0.5 * (inef_coeff_1 + inef_coeff_2)

        for base_2 in range(qutrit_projection_n_state2):
            if base_2 == 0 or base_2 == 2:
                l_1 = 1
                inef_coeff_s = (math.factorial(abs(l_1)) ** 2) * (A ** (2 * abs(l_1) + 1) * B) / (
                    math.factorial(2 * abs(l_1)))
            elif base_2 == 1:
                l_1 = 0
                inef_coeff_s = (math.factorial(abs(l_1)) ** 2) * (A ** (2 * abs(l_1) + 1) * B) / (
                    math.factorial(2 * abs(l_1)))

            else:
                if base_2 < 7 or base_2 > 10:
                    l_1, l_2 = 1, 0
                else:
                    l_1, l_2 = 1, 1
                inef_coeff_1 = (math.factorial(abs(l_1)) ** 2) * (A ** (2 * abs(l_1) + 1) * B) / (
                    math.factorial(2 * abs(l_1)))
                inef_coeff_2 = (math.factorial(abs(l_2)) ** 2) * (A ** (2 * abs(l_2) + 1) * B) / (
                    math.factorial(2 * abs(l_2)))
                inef_coeff_s = 0.5 * (inef_coeff_1 + inef_coeff_2)

            inef_coeff = inef_coeff.at[base_1, base_2].set((inef_coeff_i + inef_coeff_s))

    return inef_coeff.reshape(1, qutrit_projection_n_state2 ** 2)


@jit
def get_qubit_density_matrix(
        tomography_matrix,
        masks,
        rotation_mats
):

    tomography_matrix = tomography_matrix.reshape(qubit_projection_n_state2, qubit_projection_n_state2)

    dens_mat = (1 / (qubit_tomography_dimensions ** 2)) * (tomography_matrix * masks).sum(1).sum(1).reshape(
        qubit_tomography_dimensions ** 4, 1, 1)
    dens_mat = (dens_mat * rotation_mats)
    dens_mat = dens_mat.sum(0)

    return dens_mat


@jit
def get_qutrit_density_matrix(
        tomography_matrix,
        masks,
        rotation_mats
):

    tomography_matrix = tomography_matrix.reshape(qutrit_projection_n_state2, qutrit_projection_n_state2)

    dens_mat = (1 / (qutrit_tomography_dimensions ** 2)) * (tomography_matrix * masks).sum(1).sum(1).reshape(
        qutrit_tomography_dimensions ** 4, 1, 1)
    dens_mat = (dens_mat * rotation_mats)
    dens_mat = dens_mat.sum(0)

    return dens_mat


class DensMat(ABC):
    """
    A class that holds tomography dimensions and
    tensors used for calculating qubit and qutrit tomography
    """

    def __init__(
            self,
            projection_n_state2,
            tomography_dimension
    ):
        assert tomography_dimension in [2, 3], "tomography_dimension must be 2 or 3, " \
                                               f"got {tomography_dimension}"

        self.projection_n_state2 = projection_n_state2
        self.tomography_dimension = tomography_dimension
        self.rotation_mats, self.masks = self.dens_mat_tensors()

    def dens_mat_tensors(
            self
    ):
        rot_mats_tensor = np.zeros([self.tomography_dimension ** 4,
                                    self.tomography_dimension ** 2,
                                    self.tomography_dimension ** 2],
                                   dtype='complex64')

        masks_tensor = np.zeros([self.tomography_dimension ** 4,
                                 self.projection_n_state2,
                                 self.projection_n_state2],
                                dtype='complex64')

        if self.tomography_dimension == 2:
            mats = (
                np.eye(2, dtype='complex64'),
                np.array([[0, 1], [1, 0]]),
                np.array([[0, -1j], [1j, 0]]),
                np.array([[1, 0], [0, -1]])
            )

            vecs = (
                np.array([1, 1, 0, 0, 0, 0]),
                np.array([0, 0, 1, -1, 0, 0]),
                np.array([0, 0, 0, 0, 1, -1]),
                np.array([1, -1, 0, 0, 0, 0])
            )

        else:  # tomography_dimension == 3
            mats = (
                np.eye(3, dtype='complex64'),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
                np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
                np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]),
                (1 / np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
            )

            vecs = (
                np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]),
                (np.sqrt(3) / 3) * np.array([1, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            )

        counter = 0

        for m in range(self.tomography_dimension ** 2):
            for n in range(self.tomography_dimension ** 2):
                norm1 = np.trace(mats[m] @ mats[m])
                norm2 = np.trace(mats[n] @ mats[n])
                mat1 = mats[m] / norm1
                mat2 = mats[n] / norm2
                rot_mats_tensor = index_add(rot_mats_tensor, index[counter, :, :], np.kron(mat1, mat2))
                mask = np.dot(vecs[m].reshape(self.projection_n_state2, 1),
                              np.transpose((vecs[n]).reshape(self.projection_n_state2, 1)))
                masks_tensor = index_add(masks_tensor, index[counter, :, :], mask)
                counter = counter + 1

        return rot_mats_tensor, masks_tensor




def LaguerreBank(
        lam,
        refractive_index,
        W0,
        max_mode_p,
        max_mode_l,
        x,
        y,
        z=0,
        get_dict: bool = False,
):
    """
    generates a dictionary of Laguerre Gauss basis functions

    Parameters
    ----------
    lam; wavelength
    refractive_index: refractive index
    W0: beam waist
    max_mode_p: maximum projection mode 1st axis
    max_mode_l: maximum projection mode 2nd axis
    x: transverse points, x axis
    y: transverse points, y axis
    z: projection longitudinal position
    get_dict: (True/False) if True, the function will return a dictionary,
              else the dictionary is splitted to basis functions np.array and list of dictionary keys.

    Returns
    -------
    dictionary of Laguerre Gauss basis functions
    """
    Laguerre_dict = {}
    for p in range(max_mode_p):
        for l in range(-max_mode_l, max_mode_l + 1):
            Laguerre_dict[f'|LG{p}{l}>'] = Laguerre_gauss(lam, refractive_index, W0, l, p, z, x, y)
    if get_dict:
        return Laguerre_dict

    return np.array(list(Laguerre_dict.values())), [*Laguerre_dict]


def TomographyBankLG(
        lam,
        refractive_index,
        W0,
        max_mode_p,
        max_mode_l,
        x,
        y,
        z=0,
        relative_phase: List[Union[Union[int, float], Any]] = None,
        tomography_quantum_state: str = None,
):
    """
    generates a dictionary of basis function with projections into two orthogonal LG bases and mutually unbiased
    bases (MUBs). The MUBs are constructed from superpositions of the two orthogonal LG bases.
    according to: https://doi.org/10.1364/AOP.11.000067

    Parameters
    ----------
    lam; wavelength
    refractive_index: refractive index
    W0: beam waist
    max_mode_p: maximum projection mode 1st axis
    max_mode_l: maximum projection mode 2nd axis
    x: transverse points, x axis
    y: transverse points, y axis
    z: projection longitudinal position
    relative_phase: The relative phase between the mutually unbiased bases (MUBs) states
    tomography_quantum_state: the current quantum state we calculate it tomography matrix.
                              currently we support: qubit/qutrit

    Returns
    -------
    dictionary of bases functions used for constructing the tomography matrix
    """

    TOMO_dict = \
        LaguerreBank(
            lam,
            refractive_index,
            W0,
            max_mode_p,
            max_mode_l,
            x, y, z,
            get_dict=True)

    if tomography_quantum_state is QUBIT:
        del TOMO_dict['|LG00>']

    LG_modes, LG_string = np.array(list(TOMO_dict.values())), [*TOMO_dict]

    for m in range(len(TOMO_dict) - 1, -1, -1):
        for n in range(m - 1, -1, -1):
            for k in range(len(relative_phase)):
                TOMO_dict[f'{LG_string[m]}+e^j{str(relative_phase[k]/np.pi)}π{LG_string[n]}'] = \
                    (1 / np.sqrt(2)) * (LG_modes[m] + np.exp(1j * relative_phase[k]) * LG_modes[n])

    return np.array(list(TOMO_dict.values())), [*TOMO_dict]

def LaguerreP(p, l, x):
    """
    Generalized Laguerre polynomial of rank p,l L_p^|l|(x)

    Parameters
    ----------
    l, p: order of the LG beam
    x: matrix of x

    Returns
    -------
    Generalized Laguerre polynomial
    """
    if p == 0:
        return 1
    elif p == 1:
        return 1 + np.abs(l)-x
    else:
        return ((2*p-1+np.abs(l)-x)*LaguerreP(p-1, l, x) - (p-1+np.abs(l))*LaguerreP(p-2, l, x))/p


def Laguerre_gauss(lam, refractive_index, W0, l, p, z, x, y, coef=None):
    """
    Laguerre Gauss in 2D

    Parameters
    ----------
    lam: wavelength
    refractive_index: refractive index
    W0: beam waists
    l, p: order of the LG beam
    z: the place in z to calculate for
    x,y: matrices of x and y
    coef

    Returns
    -------
    Laguerre-Gaussian beam of order l,p in 2D
    """
    k = 2 * np.pi * refractive_index / lam
    z0 = np.pi * W0 ** 2 * refractive_index / lam  # Rayleigh range
    Wz = W0 * np.sqrt(1 + (z / z0) ** 2)  # w(z), the variation of the spot size
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    invR = z / ((z ** 2) + (z0 ** 2))  # radius of curvature
    gouy = (np.abs(l)+2*p+1)*np.arctan(z/z0)
    if coef is None:
        coef = np.sqrt(2*math.factorial(p)/(np.pi * math.factorial(p + np.abs(l))))

    U = coef * \
        (W0/Wz)*(r*np.sqrt(2)/Wz)**(np.abs(l)) * \
        np.exp(-r**2 / Wz**2) * \
        LaguerreP(p, l, 2 * r**2 / Wz**2) * \
        np.exp(-1j * (k * r**2 / 2) * invR) * \
        np.exp(-1j * l * phi) * \
        np.exp(1j * gouy)
    return U



class Projection_coincidence_rate(ABC):
    """
    A class that represents the projective basis for
    calculating the coincidence rate observable of the interaction.
    """

    def __init__(
            self,
            waist_pump0: float,
            signal_wavelength: float,
            crystal_x: np.array,
            crystal_y: np.array,
            temperature: float,
            ctype,
            polarization: str,
            z: float = 0.,
            projection_basis: str = 'LG',
            max_mode1: int = 1,
            max_mode2: int = 4,
            waist: float = None,
            wavelength:  float = None,
            tau: float = 1e-9,
            SMF_waist: float = None,

    ):
        """

        Parameters
        ----------
        waist_pump0: pump waists at the center of the crystal (initial-before training)
        signal_wavelength: signal wavelength at spdc interaction
        crystal_x: x axis linspace array (transverse)
        crystal_y: y axis linspace array (transverse)
        temperature: interaction temperature
        ctype: refractive index function
        polarization: polarization for calculating effective refractive index
        z: projection longitudinal position
        projection_basis: type of projection basis
                          Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
        max_mode1: Maximum value of first mode of the 2D projection basis
        max_mode2: Maximum value of second mode of the 2D projection basis
        waist: waists of the projection basis functions
        wavelength: wavelength for generating projection basis
        tau: coincidence window [nano sec]
        SMF_waist: signal/idler beam radius at single mode fibre
        """

        self.tau = tau

        if waist is None:
            self.waist = np.sqrt(2) * waist_pump0
        else:
            self.waist = waist

        if wavelength is None:
            wavelength = signal_wavelength

        assert projection_basis.lower() in ['lg', 'hg'], 'The projection basis is LG or HG ' \
                                                         'basis functions only'

        self.projection_basis = projection_basis
        self.max_mode1 = max_mode1
        self.max_mode2 = max_mode2

        # number of modes for projection basis
        if projection_basis.lower() == 'lg':
            self.projection_n_modes1 = max_mode1
            self.projection_n_modes2 = 2 * max_mode2 + 1
        else:
            self.projection_n_modes1 = max_mode1
            self.projection_n_modes2 = max_mode2

        # Total number of projection modes
        self.projection_n_modes = self.projection_n_modes1 * self.projection_n_modes2

        refractive_index = ctype(wavelength * 1e6, temperature, polarization)
        [x, y] = np.meshgrid(crystal_x, crystal_y)

        self.SMF_waist = SMF_waist

        if True:
            if projection_basis.lower() == 'lg':
                self.basis_arr, self.basis_str = \
                    LaguerreBank(
                        wavelength,
                        refractive_index,
                        self.waist,
                        self.max_mode1,
                        self.max_mode2,
                        x, y, z)
            else:
                self.basis_arr, self.basis_str = \
                    HermiteBank(
                        wavelength,
                        refractive_index,
                        self.waist,
                        self.max_mode1,
                        self.max_mode2,
                        x, y, z)


class Projection_tomography_matrix(ABC):
    """
    A class that represents the projective basis for
    calculating the tomography matrix & density matrix observable of the interaction.
    """

    def __init__(
            self,
            waist_pump0: float,
            signal_wavelength: float,
            crystal_x: np.array,
            crystal_y: np.array,
            temperature: float,
            ctype,
            polarization: str,
            z: float = 0.,
            projection_basis: str = 'LG',
            max_mode1: int = 1,
            max_mode2: int = 1,
            waist: float = None,
            wavelength:  float = None,
            tau: float = 1e-9,
            relative_phase: List[Union[Union[int, float], Any]] = None,
            tomography_quantum_state: str = None,

    ):
        """

        Parameters
        ----------
        waist_pump0: pump waists at the center of the crystal (initial-before training)
        signal_wavelength: signal wavelength at spdc interaction
        crystal_x: x axis linspace array (transverse)
        crystal_y: y axis linspace array (transverse)
        temperature: interaction temperature
        ctype: refractive index function
        polarization: polarization for calculating effective refractive index
        z: projection longitudinal position
        projection_basis: type of projection basis
                          Can be: LG (Laguerre-Gauss)
        max_mode1: Maximum value of first mode of the 2D projection basis
        max_mode2: Maximum value of second mode of the 2D projection basis
        waist: waists of the projection basis functions
        wavelength: wavelength for generating projection basis
        tau: coincidence window [nano sec]
        relative_phase: The relative phase between the mutually unbiased bases (MUBs) states
        tomography_quantum_state: the current quantum state we calculate it tomography matrix.
                                  currently we support: qubit/qutrit
        """

        self.tau = tau

        if waist is None:
            self.waist = np.sqrt(2) * waist_pump0
        else:
            self.waist = waist

        if wavelength is None:
            wavelength = signal_wavelength

        assert projection_basis.lower() in ['lg', 'hg'], 'The projection basis is LG or HG' \
                                                         'basis functions only'

        assert max_mode1 == 1, 'for Tomography projections, max_mode1 must be 1'
        assert max_mode2 == 1, 'for Tomography projections, max_mode2 must be 1'

        self.projection_basis = projection_basis
        self.max_mode1 = max_mode1
        self.max_mode2 = max_mode2

        assert tomography_quantum_state in [QUBIT, QUTRIT], f'quantum state must be {QUBIT} or {QUTRIT}, ' \
                                                            'but received {tomography_quantum_state}'
        self.tomography_quantum_state = tomography_quantum_state
        self.relative_phase = relative_phase

        self.projection_n_state1 = 1
        if self.tomography_quantum_state is QUBIT:
            self.projection_n_state2 = qubit_projection_n_state2
            self.tomography_dimensions = qubit_tomography_dimensions
        else:
            self.projection_n_state2 = qutrit_projection_n_state2
            self.tomography_dimensions = qutrit_tomography_dimensions

        refractive_index = ctype(wavelength * 1e6, temperature, polarization)
        [x, y] = np.meshgrid(crystal_x, crystal_y)
        if True:
            if self.projection_basis == 'lg':
                self.basis_arr, self.basis_str = \
                    TomographyBankLG(
                        wavelength,
                        refractive_index,
                        self.waist,
                        self.max_mode1,
                        self.max_mode2,
                        x, y, z,
                        self.relative_phase,
                        self.tomography_quantum_state
                    )
            else:
                self.basis_arr, self.basis_str = \
                    TomographyBankHG(
                        wavelength,
                        refractive_index,
                        self.waist,
                        self.max_mode1,
                        self.max_mode2,
                        x, y, z,
                        self.relative_phase,
                        self.tomography_quantum_state
                    )
