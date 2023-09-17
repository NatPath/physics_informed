import sys
import jax.numpy as np
from jax import lax
from jax import jit
from correlation_utiles.defaults import qubit_projection_n_state2, \
    qubit_tomography_dimensions, qutrit_projection_n_state2, qutrit_tomography_dimensions, QUBIT, QUTRIT
import math
from typing import List, Union, Any





# Constants:
pi      = np.pi
c       = 2.99792458e8  # speed of light [meter/sec]
eps0    = 8.854187817e-12  # vacuum permittivity [Farad/meter]
h_bar   = 1.054571800e-34  # [m^2 kg / s], taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck

G1_Normalization        = lambda w: h_bar * w / (2 * eps0 * c)
SFG_idler_wavelength    = lambda lambda_p, lambda_s: lambda_p * lambda_s / (lambda_s - lambda_p)

def n_KTP_Kato(
        lam: float,
        T: float,
        ax: str,
):
    """
    Refractive index for KTP, based on K. Kato

    Parameters
    ----------
    lam: wavelength (lambda) [um]
    T: Temperature [Celsius Degrees]
    ax: polarization

    Returns
    -------
    n: Refractive index

    """
    assert ax in ['z', 'y'], 'polarization must be either z or y'
    dT = (T - 20)
    if ax == "z":
        n_no_T_dep = np.sqrt(4.59423 + 0.06206 / (lam ** 2 - 0.04763) + 110.80672 / (lam ** 2 - 86.12171))
        dn         = (0.9221 / lam ** 3 - 2.9220 / lam ** 2 + 3.6677 / lam - 0.1897) * 1e-5 * dT
    if ax == "y":
        n_no_T_dep = np.sqrt(3.45018 + 0.04341 / (lam ** 2 - 0.04597) + 16.98825 / (lam ** 2 - 39.43799))
        dn         = (0.1997 / lam ** 3 - 0.4063 / lam ** 2 + 0.5154 / lam + 0.5425) * 1e-5 * dT
    n           = n_no_T_dep + dn
    return n


def propagate(A, x, y, k, dz):
    """
    Free Space propagation using the free space transfer function,
    (two  dimensional), according to Saleh
    Using CGS, or MKS, Boyd 2nd eddition

    Parameters
    ----------
    A: electromagnetic beam profile
    x,y: spatial vectors
    k: wave vector
    dz: The distance to propagate

    Returns the propagated field
    -------

    """
    dx      = np.abs(x[1] - x[0])
    dy      = np.abs(y[1] - y[0])

    # define the fourier vectors
    X, Y    = np.meshgrid(x, y, indexing='ij')
    KX      = 2 * np.pi * (X / dx) / (np.size(X, 1) * dx)
    KY      = 2 * np.pi * (Y / dy) / (np.size(Y, 1) * dy)

    # The Free space transfer function of propagation, using the Fresnel approximation
    # (from "Engineering optics with matlab"/ing-ChungPoon&TaegeunKim):
    H_w = np.exp(-1j * dz * (np.square(KX) + np.square(KY)) / (2 * k))
    H_w = np.fft.ifftshift(H_w)

    # Fourier Transform: move to k-space
    G = np.fft.fft2(A)  # The two-dimensional discrete Fourier transform (DFT) of A.

    # propoagte in the fourier space
    F = np.multiply(G, H_w)

    # inverse Fourier Transform: go back to real space
    Eout = np.fft.ifft2(F)  # [in real space]. E1 is the two-dimensional INVERSE discrete Fourier transform (DFT) of F1

    return Eout



#@jit
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


#@jit
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


#@jit
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


#@jit
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


#@jit
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


#@jit
def projection_matrix_calc(G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger):
    """

    Parameters
    ----------
    G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger: the projective matrices
    Returns the 2nd order projective matrix
    -------

    """
    # return (lax.psum(G1_ii, 'device') *
    #         lax.psum(G1_ss, 'device') +
    #         lax.psum(Q_si_dagger, 'device') *
    #         lax.psum(Q_si, 'device') +
    #         lax.psum(G1_si_dagger, 'device') *
    #         lax.psum(G1_si, 'device')
    #         ).real

    return (G1_ii *
            G1_ss +
            Q_si_dagger *
            Q_si +
            G1_si_dagger *
            G1_si
            ).real

# for coupling inefficiencies
#@jit
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


#@jit
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


#@jit
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


#@jit
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



