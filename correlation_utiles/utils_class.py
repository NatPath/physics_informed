from abc import ABC
import jax.numpy as np
from jax import lax
from jax import jit
from jax.ops import  index_add, index
from jax.numpy.ndarray import  at
from typing import List, Union, Any
import math
from defaults import qubit_projection_n_state2, \
    qubit_tomography_dimensions, qutrit_projection_n_state2, qutrit_tomography_dimensions, QUBIT, QUTRIT
from utils_function import LaguerreBank

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


class Beam(ABC):
    """
    A class that holds everything to do with a beam
    """
    def __init__(self,
                 lam: float,
                 polarization: str,
                 T: float,
                 power: float = 0):

        """

        Parameters
        ----------
        lam: beam's wavelength
        ctype: function that holds crystal type fo calculating refractive index
        polarization: Polarization of the beam
        T: crystal's temperature [Celsius Degrees]
        power: beam power [watt]
        """
        ctype = self.ctype = n_KTP_Kato
        self.lam          = lam
        self.n            = ctype(lam * 1e6, T, polarization)  # refractive index
        self.w            = 2 * np.pi * c / lam  # frequency
        self.k            = 2 * np.pi * ctype(lam * 1e6, T, polarization) / lam  # wave vector
        self.power        = power  # beam power


class Field(ABC):
    """
    A class that holds everything to do with the interaction values of a given beam
    vac   - corresponding vacuum state coefficient
    kappa - coupling constant
    k     - wave vector
    """
    def __init__(
            self,
            beam,
            dx,
            dy,
            maxZ
    ):
        """

        Parameters
        ----------
        beam: A class that holds everything to do with a beam
        dx: transverse resolution in x [m]
        dy: transverse resolution in y [m]
        maxZ: Crystal's length in z [m]
        """

        self.beam = beam
        self.vac   = np.sqrt(h_bar * beam.w / (2 * eps0 * beam.n ** 2 * dx * dy * maxZ))
        self.kappa = 2 * 1j * beam.w ** 2 / (beam.k * c ** 2)
        self.k     = beam.k

class Shape():
    """
    A class that holds everything to do with the dimensions
    """
    def __init__(
            self,
            dx: float = 2e-6, # [1e-6,2e-6,3e-6,4e-6,5e-6,6e-6]
            dy: float = 2e-6, # [1e-6,2e-6,3e-6,4e-6,5e-6,6e-6]
            dz: float = 10e-6,  # [2e-6,5e-6,10e-6,20e-6]
            maxX: float = 120e-6, # [80e-6,120e-6,160e-6,200e-6,240e-6]
            maxY: float = 120e-6, # [80e-6,120e-6,160e-6,200e-6,240e-6]
            maxZ: float = 1e-4, # [1e-4,2e-4,3e-4,4e-4,5e-4]
    ):


        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x = np.arange(-maxX, maxX, dx)  # x axis, length 2*MaxX (transverse)
        self.y = np.arange(-maxY, maxY, dy)  # y axis, length 2*MaxY  (transverse)
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.z = np.arange(-maxZ / 2, maxZ / 2, dz)  # z axis, length MaxZ (propagation)
        self.Nz = len(self.z)
        self.maxX = maxX
        self.maxY = maxY
        self.maxZ = maxZ
