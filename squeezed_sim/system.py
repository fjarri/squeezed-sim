import enum

import numpy
from scipy.linalg import schur, sqrtm


class Representation(enum.Enum):
    POSITIVE_P = 'posp'
    WIGNER = 'wigner'
    Q = 'q'


def ordering(representation):
    if representation == Representation.POSITIVE_P:
        return 0
    elif representation == Representation.WIGNER:
        return 0.5
    elif representation == Representation.Q:
        return 1
    else:
        raise NotImplementedError(representation)


class System:

    def __init__(
            self,
            unitary=None,
            transmission=1,
            decoherence=0,
            squeezing=0,
            thermal_noise=0,
            inputs=None):

        complex_dtype = numpy.complex128
        real_dtype = numpy.float64

        assert unitary.ndim == 2 and unitary.shape[0] == unitary.shape[1]
        self.unitary = unitary.astype(complex_dtype)

        self.modes = unitary.shape[0]

        if inputs is not None:
            assert inputs <= self.modes
            self.inputs = inputs
        else:
            self.inputs = self.modes

        if isinstance(decoherence, numpy.ndarray):
            assert decoherence.shape == (self.inputs,)
        else:
            decoherence = numpy.ones(self.inputs) * decoherence
        self.decoherence = decoherence.astype(real_dtype)

        if isinstance(transmission, numpy.ndarray):
            assert transmission.shape == (self.inputs,)
        else:
            transmission = numpy.ones(self.inputs) * transmission
        self.transmission = transmission.astype(real_dtype)

        if isinstance(squeezing, numpy.ndarray):
            assert squeezing.shape == (self.inputs,)
        else:
            squeezing = numpy.ones(self.inputs) * squeezing
        self.squeezing = squeezing.astype(real_dtype)

        if isinstance(thermal_noise, numpy.ndarray):
            assert thermal_noise.shape == (self.inputs,)
        else:
            thermal_noise = numpy.ones(self.inputs) * thermal_noise
        self.thermal_noise = thermal_noise.astype(real_dtype)

        # A two-step process, since the noise matrix is expensive to calculate,
        # and is only needed for non-positive-P representations,
        # which we rarely use.
        self._needs_noise_matrix = None
        self._diff_matrix = None
        self._noise_matrix = None

    def needs_noise_matrix(self):
        if self._needs_noise_matrix is None:
            diff = numpy.eye(self.modes) - self.unitary @ self.unitary.transpose().conj()
            self._needs_noise_matrix = numpy.allclose(numpy.linalg.norm(diff), 0)
            # so that we don't have to re-calculate it
            if self._needs_noise_matrix:
                self._diff_matrix = diff

        return self._needs_noise_matrix

    def noise_matrix(self):
        if self._noise_matrix is None:
            if not self.needs_noise_matrix():
                return None

            U, T = schur(self._diff_matrix)
            self._noise_matrix = U @ sqrtm(T) * U.transpose().conj()

        return self._noise_matrix


class State:

    def __init__(self, system, representation, alpha, beta):
        self.system = system
        self.representation = representation
        self.alpha = alpha
        self.beta = beta
        self.samples = alpha.shape[0]
