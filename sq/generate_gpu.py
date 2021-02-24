import numpy

from reikna import helpers
import reikna.cluda.functions as functions
from reikna.core import Computation, Parameter, Type, Annotation, Transformation
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
        thermal_noise = plan.persistent_array(self._system.thermal_noise)
        input_transmission = plan.persistent_array(self._system.input_transmission)

        plan.kernel_call(
            TEMPLATE.get_def("generate_input_state"),
            [alpha, beta, squeezing, thermal_noise, input_transmission, seed],
            kernel_name="generate",
            global_size=alpha.shape,
            render_kwds=dict(
                system=self._system,
                representation=self._representation,
                Representation=Representation,
                bijection=bijection,
                keygen=keygen,
                sampler=sampler,
                apply_transmission_coeffs=(self._system.input_transmission != 1).any(),
                ordering=ordering,
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

    def _make_trf_transmission(self, state, conj_noise=False):

        bijection = philox(64, 2)

        # Keeping the kernel the same so it can be cached.
        # The seed will be passed as the computation parameter instead.
        keygen = KeyGenerator.create(bijection, seed=numpy.int32(0))

        sampler = normal_bm(bijection, numpy.float64)

        return Transformation([
            Parameter('output', Annotation(state, 'o')),
            Parameter('input', Annotation(state, 'i')),
            Parameter('transmission', Annotation(self._system.output_transmission, 'i')),
            Parameter('seed', Annotation(numpy.int32)),
            ],
            """
            ${input.ctype} val = ${input.load_same};

            %if apply_transmission_coeffs:

                <%
                    real = dtypes.ctype(dtypes.real_for(input.dtype))
                    comp = input.ctype

                    s = ordering(representation)
                %>

                VSIZE_T sample_idx = ${idxs[0]};
                VSIZE_T mode_idx = ${idxs[1]};
                ${transmission.ctype} otr = ${transmission.load_idx}(mode_idx);
                val = ${mul_cr}(val, otr);

                %if s != 0:

                    VSIZE_T flat_idx = ${idxs[1]} + ${idxs[0]} * ${input.shape[1]};
                    ${bijection.module}Key key = ${keygen.module}key_from_int(flat_idx);
                    ${bijection.module}Counter ctr = ${bijection.module}make_counter_from_int(${seed});
                    ${bijection.module}State st = ${bijection.module}make_state(key, ctr);

                    ${sampler.module}Result w = ${sampler.module}sample(&st);

                    ${real} coeff = sqrt((1 - otr * otr) * ${s / 2});
                    ${real} w_re = w.v[0] * coeff;
                    ${real} w_im = w.v[1] * coeff;

                    %if conj_noise:
                    w_im = -w_im;
                    %endif

                    val = ${add_cc}(val, COMPLEX_CTR(${comp})(w_re, w_im));
                %endif

            %endif

            ${output.store_same}(val);
            """,
            #TEMPLATE.get_def("output_transmission_transformation"),
            render_kwds=dict(
                system=self._system,
                representation=self._representation,
                Representation=Representation,
                bijection=bijection,
                keygen=keygen,
                sampler=sampler,
                apply_transmission_coeffs=(self._system.output_transmission != 1).any(),
                ordering=ordering,
                mul_cr=functions.mul(numpy.complex128, numpy.float64),
                add_cc=functions.add(numpy.complex128, numpy.complex128),
                conj_noise=conj_noise,
                ))

    def _build_plan(self, plan_factory, device_params, alpha, beta, alpha_i, beta_i, seed):
        plan = plan_factory()

        output_transmission = plan.persistent_array(self._system.output_transmission)
        unitary = plan.persistent_array(self._system.unitary)

        mmul_alpha = MatrixMul(alpha, unitary, transposed_b=True)

        # Apply output transmission (possibly with noise)
        trf_transmission = self._make_trf_transmission(alpha, conj_noise=False)
        mmul_alpha.parameter.output.connect(
            trf_transmission, trf_transmission.input,
            output_p=trf_transmission.output,
            transmission=trf_transmission.transmission,
            seed=trf_transmission.seed)

        mmul_beta = MatrixMul(alpha, unitary, transposed_b=True)

        # Conjugate the unitary before multiplication
        trf_conj = self._make_trf_conj()
        mmul_beta.parameter.matrix_b.connect(trf_conj, trf_conj.output, matrix_b_p=trf_conj.input)

        # Apply output transmission (possibly with noise)
        trf_transmission = self._make_trf_transmission(beta, conj_noise=True)
        mmul_beta.parameter.output.connect(
            trf_transmission, trf_transmission.input,
            output_p=trf_transmission.output,
            transmission=trf_transmission.transmission,
            seed=trf_transmission.seed)

        plan.computation_call(mmul_alpha, alpha, output_transmission, seed, alpha_i, unitary)
        plan.computation_call(mmul_beta, beta, output_transmission, seed, beta_i, unitary)

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
