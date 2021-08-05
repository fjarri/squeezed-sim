import numpy

from .system import Representation, State, ordering


def generate_input_state(system, representation, samples, seed):

    rng = numpy.random.RandomState(seed)

    nts = (system.thermal_noise / 2)**0.5

    s = ordering(representation)

    alpha = numpy.zeros((samples, system.modes), numpy.complex128)
    beta = numpy.zeros((samples, system.modes), numpy.complex128)

    n = numpy.sinh(system.squeezing)**2
    m = (1 - system.decoherence) * numpy.cosh(system.squeezing) * numpy.sinh(system.squeezing)
    nmax = system.inputs

    if representation != Representation.POSITIVE_P:
        nmax = system.modes
        n = numpy.concatenate([n, numpy.zeros(system.modes - system.inputs)])
        m = numpy.concatenate([m, numpy.zeros(system.modes - system.inputs)])

    # Converting to complex numbers because `n-m+s` can sometimes be < 0
    x = (0.5*(n + m + s).astype(numpy.complex128))**0.5 * rng.normal(size=(samples, nmax))
    y = (0.5*(n - m + s).astype(numpy.complex128))**0.5 * rng.normal(size=(samples, nmax))
    alpha[:,:nmax] = x + 1j * y
    beta[:,:nmax] = x - 1j * y

    return State(system, representation, alpha, beta)


def apply_matrix(input_state, seed):

    rng = numpy.random.RandomState(seed)

    system = input_state.system
    representation = input_state.representation
    s = ordering(representation)

    alpha_i = input_state.alpha
    beta_i = input_state.beta

    # Transposing U to perform batched matrix-vector multiplication
    alpha = (alpha_i @ system.unitary.transpose())
    beta = (beta_i @ system.unitary.transpose().conj())

    E = numpy.eye(system.modes) - system.unitary @ system.unitary.transpose().conj()
    if representation != Representation.POSITIVE_P and system.needs_noise_matrix():
        noise_matrix = system.noise_matrix()
        w = (
            numpy.random.normal(size=(input_state.samples, system.modes))
            + 1j * numpy.random.normal(size=(input_state.samples, system.modes)))
        # transposing the noise matrix to perform batched matrix-vector multiplication
        alpha += w @ noise_matrix.transpose()
        beta = alpha.conj()

    return State(system, representation, alpha, beta)
