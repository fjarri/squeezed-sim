import numpy
import enum


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
            input_transmission=1,
            output_transmission=1,
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

        if isinstance(input_transmission, numpy.ndarray):
            assert input_transmission.shape == (self.modes,)
        else:
            input_transmission = numpy.ones(self.modes) * input_transmission
        self.input_transmission = input_transmission.astype(real_dtype)

        if isinstance(output_transmission, numpy.ndarray):
            assert output_transmission.shape == (self.modes,)
        else:
            output_transmission = numpy.ones(self.modes) * output_transmission
        self.output_transmission = output_transmission.astype(real_dtype)

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


class State:

    def __init__(self, system, representation, alpha, beta):
        self.system = system
        self.representation = representation
        self.alpha = alpha
        self.beta = beta
        self.samples = alpha.shape[0]
