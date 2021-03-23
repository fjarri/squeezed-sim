import numpy

from reikna import helpers
import reikna.cluda.functions as functions
from reikna.core import Computation, Parameter, Type, Annotation, Transformation
from reikna.cbrng import CBRNG
from reikna.cbrng.bijections import philox
from reikna.cbrng.tools import KeyGenerator
from reikna.cbrng.samplers import normal_bm
from reikna.linalg import MatrixMul

from .system import State, Representation, ordering


TEMPLATE = helpers.template_for(__file__)


class GenerateInputState(Computation):

    def __init__(self, system, representation, samples):
        state = Type(numpy.complex128, (samples, system.modes))
        Computation.__init__(
            self,
            [
                Parameter('alpha', Annotation(state, 'o')),
                Parameter('beta', Annotation(state, 'o')),
                Parameter('seed', Annotation(numpy.int32)),
            ])

        self._system = system
        self._representation = representation

    def _build_plan(self, plan_factory, device_params, alpha, beta, seed):
        plan = plan_factory()

        bijection = philox(64, 2)

        # Keeping the kernel the same so it can be cached.
        # The seed will be passed as the computation parameter instead.
        keygen = KeyGenerator.create(bijection, seed=numpy.int32(0))

        sampler = normal_bm(bijection, numpy.float64)

        squeezing = plan.persistent_array(self._system.squeezing)
        decoherence = plan.persistent_array(self._system.decoherence)
        transmission = plan.persistent_array(self._system.transmission)

        plan.kernel_call(
            TEMPLATE.get_def("generate_input_state"),
            [alpha, beta, squeezing, decoherence, transmission, seed],
            kernel_name="generate",
            global_size=alpha.shape,
            render_kwds=dict(
                system=self._system,
                representation=self._representation,
                Representation=Representation,
                bijection=bijection,
                keygen=keygen,
                sampler=sampler,
                ordering=ordering,
                exp=functions.exp(numpy.float64),
                mul_cr=functions.mul(numpy.complex128, numpy.float64),
                add_cc=functions.add(numpy.complex128, numpy.complex128),
                ))

        return plan


def prepare_generate_input_state(thread, system, representation, samples):

    comp = GenerateInputState(system, representation, samples)
    ccomp = comp.compile(thread)

    alpha = thread.empty_like(ccomp.parameter.alpha)
    beta = thread.empty_like(ccomp.parameter.alpha)

    def generate_input_state(seed):
        ccomp(alpha, beta, numpy.int32(seed))
        return State(system, representation, alpha, beta)

    return generate_input_state


class ApplyMatrix(Computation):

    def __init__(self, system, representation, samples):
        state = Type(numpy.complex128, (samples, system.modes))
        Computation.__init__(
            self,
            [
                Parameter('alpha', Annotation(state, 'o')),
                Parameter('beta', Annotation(state, 'o')),
                Parameter('alpha_i', Annotation(state, 'i')),
                Parameter('beta_i', Annotation(state, 'i')),
                Parameter('seed', Annotation(numpy.int32)),
            ])

        self._system = system
        self._representation = representation

    def _make_trf_conj(self):
        return Transformation([
            Parameter('output', Annotation(self._system.unitary, 'o')),
            Parameter('input', Annotation(self._system.unitary, 'i'))
            ],
            """
            ${output.store_same}(${conj}(${input.load_same}));
            """,
            render_kwds=dict(conj=functions.conj(self._system.unitary.dtype)))

    def _build_plan(self, plan_factory, device_params, alpha, beta, alpha_i, beta_i, seed):
        plan = plan_factory()

        system = self._system
        representation = self._representation

        unitary = plan.persistent_array(self._system.unitary)

        needs_noise_matrix = representation != Representation.POSITIVE_P and system.needs_noise_matrix()

        mmul = MatrixMul(alpha, unitary, transposed_b=True)

        if not needs_noise_matrix:

            # TODO: this could be sped up for repr != POSITIVE_P,
            # since in that case alpha == conj(beta), and we don't need to do two multuplications.

            mmul_beta = MatrixMul(beta, unitary, transposed_b=True)
            trf_conj = self._make_trf_conj()
            mmul_beta.parameter.matrix_b.connect(trf_conj, trf_conj.output, matrix_b_p=trf_conj.input)

            plan.computation_call(mmul, alpha, alpha_i, unitary)
            plan.computation_call(mmul_beta, beta, beta_i, unitary)

        else:

            noise_matrix = system.noise_matrix()
            noise_matrix_dev = plan.persistent_array(noise_matrix)

            # If we're here, it's not positive-P, and alpha == conj(beta).
            # This means we can just calculate alpha, and then build beta from it.

            w = plan.temp_array_like(alpha)
            temp_alpha = plan.temp_array_like(alpha)

            plan.computation_call(mmul, temp_alpha, alpha_i, unitary)

            bijection = philox(64, 2)

            # Keeping the kernel the same so it can be cached.
            # The seed will be passed as the computation parameter instead.
            keygen = KeyGenerator.create(bijection, seed=numpy.int32(0))

            sampler = normal_bm(bijection, numpy.float64)

            plan.kernel_call(
                TEMPLATE.get_def("generate_apply_matrix_noise"),
                [w, seed],
                kernel_name="generate_apply_matrix_noise",
                global_size=alpha.shape,
                render_kwds=dict(
                    bijection=bijection,
                    keygen=keygen,
                    sampler=sampler,
                    mul_cr=functions.mul(numpy.complex128, numpy.float64),
                    add_cc=functions.add(numpy.complex128, numpy.complex128),
                    ))

            noise = plan.temp_array_like(alpha)
            plan.computation_call(mmul, noise, w, noise_matrix_dev)

            plan.kernel_call(
                TEMPLATE.get_def("add_noise"),
                [alpha, beta, temp_alpha, noise],
                kernel_name="add_noise",
                global_size=alpha.shape,
                render_kwds=dict(
                    add=functions.add(numpy.complex128, numpy.complex128),
                    conj=functions.conj(numpy.complex128)
                    ))

        return plan


def prepare_apply_matrix(thread, system, representation, samples):

    comp = ApplyMatrix(system, representation, samples)
    ccomp = comp.compile(thread)

    alpha = thread.empty_like(ccomp.parameter.alpha)
    beta = thread.empty_like(ccomp.parameter.alpha)

    def apply_matrix(input_state, seed):
        ccomp(alpha, beta, input_state.alpha, input_state.beta, numpy.int32(seed))
        return State(system, representation, alpha, beta)

    return apply_matrix
