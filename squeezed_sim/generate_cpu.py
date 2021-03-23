import numpy

from .system import Representation, State, ordering


def generate_input_state(system, representation, samples, seed):

    rng = numpy.random.RandomState(seed)

    nts = (system.thermal_noise / 2)**0.5

    if representation == Representation.POSITIVE_P:

        w1 = rng.normal(size=(samples, system.inputs))
        w2 = rng.normal(size=(samples, system.inputs))
        w3 = (
            rng.normal(size=(samples, system.inputs))
            + 1j * rng.normal(size=(samples, system.inputs)))

        sr = numpy.sinh(system.squeezing)
        cr = numpy.cosh(system.squeezing)
        a = (sr * (cr + 1) / 2)**0.5
        b = (sr * (cr - 1) / 2)**0.5

        alpha = numpy.zeros((samples, system.modes), numpy.complex128)
        beta = numpy.zeros((samples, system.modes), numpy.complex128)

        alpha[:,:system.inputs] = a * w1 + b * w2 + nts * w3
        beta[:,:system.inputs] = b * w1 + a * w2 + nts * w3.conj()

    elif representation == Representation.WIGNER:

        w1 = rng.normal(size=(samples, system.modes))
        w2 = rng.normal(size=(samples, system.modes))
        w3 = (
            rng.normal(size=(samples, system.inputs))
            + 1j * rng.normal(size=(samples, system.inputs)))

        er = numpy.exp(system.squeezing)

        a = numpy.concatenate([0.5 * er, numpy.ones(system.modes - system.inputs) * 0.5])
        b = numpy.concatenate([0.5 / er, numpy.ones(system.modes - system.inputs) * 0.5])
        alpha = a * w1 + 1j * b * w2
        alpha[:,:system.inputs] += nts * w3
        beta = alpha.conj()

    elif representation == Representation.Q:

        w1 = rng.normal(size=(samples, system.modes))
        w2 = rng.normal(size=(samples, system.modes))
        w3 = (
            rng.normal(size=(samples, system.inputs))
            + 1j * rng.normal(size=(samples, system.inputs)))

        e2r = numpy.exp(2 * system.squeezing)

        a = numpy.concatenate([0.5 * (e2r + 1)**0.5, numpy.ones(system.modes - system.inputs) * 0.5**0.5])
        b = numpy.concatenate([0.5 * (1 / e2r + 1)**0.5, numpy.ones(system.modes - system.inputs) * 0.5**0.5])

        alpha = a * w1 + 1j * b * w2
        alpha[:,:system.inputs] += nts * w3
        beta = alpha.conj()

    else:
        raise NotImplementedError(representation)

    s = ordering(representation)

    alpha_i = alpha * system.input_transmission
    beta_i = beta * system.input_transmission

    if (system.input_transmission != 1).any() and s != 0:
        w = (
            rng.normal(size=(samples, system.modes))
            + 1j * rng.normal(size=(samples, system.modes)))
        alpha_i += w * ((1 - system.input_transmission**2) * s / 2)**0.5
        beta_i = alpha_i.conj()

    return State(system, representation, alpha_i, beta_i)


def apply_matrix(input_state, seed):

    rng = numpy.random.RandomState(seed)

    system = input_state.system
    representation = input_state.representation
    s = ordering(representation)

    alpha_i = input_state.alpha
    beta_i = input_state.beta

    # Transposing U to perform batch matrix multiplication
    alpha = (alpha_i @ system.unitary.transpose()) * system.output_transmission
    beta = (beta_i @ system.unitary.transpose().conj()) * system.output_transmission

    if (system.output_transmission != 1).any() and s != 0:
        w = (
            rng.normal(size=(input_state.samples, system.modes))
            + 1j * rng.normal(size=(input_state.samples, system.modes)))
        alpha += w * ((1 - system.output_transmission**2) * s / 2)**0.5
        beta = alpha.conj()

    return State(system, representation, alpha, beta)
