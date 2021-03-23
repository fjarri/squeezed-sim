import numpy

from reikna import helpers
from reikna.cluda import OutOfResourcesError
import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.core import Computation, Parameter, Type, Annotation, Transformation
from reikna.algorithms import Reduce, predicate_sum
from reikna.fft import FFT

from .system import ordering, Representation


TEMPLATE = helpers.template_for(__file__)


class PopulationMeter(Computation):

    def __init__(self, meter, system, representation, samples):

        self._system = system
        self._representation = representation

        state = Type(numpy.complex128, (samples, system.modes))
        output = Type(numpy.float64, (system.modes,))
        Computation.__init__(
            self,
            [
                Parameter('output', Annotation(output, 'o')),
                Parameter('alpha', Annotation(state, 'i')),
                Parameter('beta', Annotation(state, 'i')),
            ])

    def _build_plan(self, plan_factory, device_params, output, alpha, beta):

        plan = plan_factory()

        for_reduction = Type(numpy.float64, alpha.shape)

        meter_trf = Transformation([
            Parameter('output', Annotation(for_reduction, 'o')),
            Parameter('alpha', Annotation(alpha, 'i')),
            Parameter('beta', Annotation(beta, 'i')),
            ],
            """
                ${alpha.ctype} alpha = ${alpha.load_same};
                ${beta.ctype} beta = ${beta.load_same};
                ${alpha.ctype} t = ${mul_cc}(alpha, beta);
                ${output.store_same}(t.x - ${ordering});
                """,
            render_kwds=dict(
                mul_cc=functions.mul(alpha.dtype, alpha.dtype),
                ordering=ordering(self._representation),
                ))

        reduction = Reduce(for_reduction, predicate_sum(output.dtype), axes=(0,))
        reduction.parameter.input.connect(
            meter_trf, meter_trf.output, alpha_p=meter_trf.alpha, beta_p=meter_trf.beta)

        plan.computation_call(reduction, output, alpha, beta)

        return plan


class MomentsMeter(Computation):

    def __init__(self, moments_meter, system, representation, samples):

        self._system = system
        self._representation = representation
        self._max_moment = moments_meter.max_moment

        state = Type(numpy.complex128, (samples, system.modes))
        output = Type(numpy.float64, (self._max_moment,))
        Computation.__init__(
            self,
            [
                Parameter('output', Annotation(output, 'o')),
                Parameter('alpha', Annotation(state, 'i')),
                Parameter('beta', Annotation(state, 'i')),
            ])

    def _build_plan(self, plan_factory, device_params, output, alpha, beta):

        plan = plan_factory()

        for_reduction = Type(numpy.float64, (alpha.shape[0], self._max_moment))

        meter_trf = Transformation([
            Parameter('output', Annotation(for_reduction, 'o')),
            Parameter('alpha', Annotation(alpha, 'i')),
            Parameter('beta', Annotation(beta, 'i')),
            ],
            """
                VSIZE_T sample_idx = ${idxs[0]};
                VSIZE_T order = ${idxs[1]};

                ${alpha.ctype} result = COMPLEX_CTR(${alpha.ctype})(1, 0);
                for (VSIZE_T i = 0; i <= order; i++) {
                    ${alpha.ctype} alpha = ${alpha.load_idx}(sample_idx, i);
                    ${beta.ctype} beta = ${beta.load_idx}(sample_idx, i);
                    ${alpha.ctype} t = ${mul_cc}(alpha, beta);
                    t.x -= ${ordering};
                    result = ${mul_cc}(result, t);
                }
                ${output.store_same}(result.x);
                """,
            render_kwds=dict(
                mul_cc=functions.mul(alpha.dtype, alpha.dtype),
                ordering=ordering(self._representation),
                ))

        reduction = Reduce(for_reduction, predicate_sum(output.dtype), axes=(0,))
        reduction.parameter.input.connect(
            meter_trf, meter_trf.output, alpha_p=meter_trf.alpha, beta_p=meter_trf.beta)

        plan.computation_call(reduction, output, alpha, beta)

        return plan


class ClickProbabilityMeter(Computation):

    def __init__(self, click_probability_meter, system, representation, samples):

        assert representation == Representation.POSITIVE_P
        self._system = system

        state = Type(numpy.complex128, (samples, system.modes))
        output = Type(numpy.float64, (system.modes,))
        Computation.__init__(
            self,
            [
                Parameter('output', Annotation(output, 'o')),
                Parameter('alpha', Annotation(state, 'i')),
                Parameter('beta', Annotation(state, 'i')),
            ])

    def _build_plan(self, plan_factory, device_params, output, alpha, beta):

        plan = plan_factory()

        for_reduction = Type(numpy.float64, alpha.shape)

        meter_trf = Transformation([
            Parameter('output', Annotation(for_reduction, 'o')),
            Parameter('alpha', Annotation(alpha, 'i')),
            Parameter('beta', Annotation(beta, 'i')),
            ],
            """
                ${alpha.ctype} alpha = ${alpha.load_same};
                ${beta.ctype} beta = ${beta.load_same};
                ${alpha.ctype} t = ${mul_cc}(alpha, beta);
                ${alpha.ctype} np = ${exp_c}(COMPLEX_CTR(${alpha.ctype})(-t.x, -t.y));
                ${alpha.ctype} cp = COMPLEX_CTR(${alpha.ctype})(1 - np.x, -np.y);
                ${output.store_same}(cp.x);
                """,
            render_kwds=dict(
                mul_cc=functions.mul(alpha.dtype, alpha.dtype),
                exp_c=functions.exp(alpha.dtype),
                ))

        reduction = Reduce(for_reduction, predicate_sum(output.dtype), axes=(0,))
        reduction.parameter.input.connect(
            meter_trf, meter_trf.output, alpha_p=meter_trf.alpha, beta_p=meter_trf.beta)

        plan.computation_call(reduction, output, alpha, beta)

        return plan


class ClicksMeter(Computation):

    def __init__(self, clicks_meter, system, representation, samples):

        assert representation == Representation.POSITIVE_P
        self._system = system
        self._max_click_order = clicks_meter.max_click_order

        state = Type(numpy.complex128, (samples, system.modes))
        output = Type(numpy.float64, (self._max_click_order,))
        Computation.__init__(
            self,
            [
                Parameter('output', Annotation(output, 'o')),
                Parameter('alpha', Annotation(state, 'i')),
                Parameter('beta', Annotation(state, 'i')),
            ])

    def _build_plan(self, plan_factory, device_params, output, alpha, beta):

        plan = plan_factory()

        for_reduction = Type(numpy.float64, (alpha.shape[0], self._max_click_order))

        meter_trf = Transformation([
            Parameter('output', Annotation(for_reduction, 'o')),
            Parameter('alpha', Annotation(alpha, 'i')),
            Parameter('beta', Annotation(beta, 'i')),
            ],
            """
                VSIZE_T sample_idx = ${idxs[0]};
                VSIZE_T order = ${idxs[1]} + 1;

                ${alpha.ctype} result = COMPLEX_CTR(${alpha.ctype})(1, 0);
                for (VSIZE_T i = 0; i < order; i++) {
                    ${alpha.ctype} alpha = ${alpha.load_idx}(sample_idx, i);
                    ${beta.ctype} beta = ${beta.load_idx}(sample_idx, i);
                    ${alpha.ctype} t = ${mul_cc}(alpha, beta);
                    ${alpha.ctype} np = ${exp_c}(COMPLEX_CTR(${alpha.ctype})(-t.x, -t.y));
                    ${alpha.ctype} cp = COMPLEX_CTR(${alpha.ctype})(1 - np.x, -np.y);
                    result = ${mul_cc}(result, cp);
                }

                ${output.store_same}(result.x);
                """,
            render_kwds=dict(
                mul_cc=functions.mul(alpha.dtype, alpha.dtype),
                exp_c=functions.exp(alpha.dtype),
                ))

        reduction = Reduce(for_reduction, predicate_sum(output.dtype), axes=(0,))
        reduction.parameter.input.connect(
            meter_trf, meter_trf.output, alpha_p=meter_trf.alpha, beta_p=meter_trf.beta)

        plan.computation_call(reduction, output, alpha, beta)

        return plan


class ZeroClicksMeter(Computation):

    def __init__(self, zero_clicks_meter, system, representation, samples):

        assert representation == Representation.POSITIVE_P
        self._system = system
        self._max_click_order = zero_clicks_meter.max_click_order

        state = Type(numpy.complex128, (samples, system.modes))
        output = Type(numpy.float64, (self._max_click_order,))
        Computation.__init__(
            self,
            [
                Parameter('output', Annotation(output, 'o')),
                Parameter('alpha', Annotation(state, 'i')),
                Parameter('beta', Annotation(state, 'i')),
            ])

    def _build_plan(self, plan_factory, device_params, output, alpha, beta):

        plan = plan_factory()

        for_reduction = Type(numpy.float64, (alpha.shape[0], self._max_click_order))

        meter_trf = Transformation([
            Parameter('output', Annotation(for_reduction, 'o')),
            Parameter('alpha', Annotation(alpha, 'i')),
            Parameter('beta', Annotation(beta, 'i')),
            ],
            """
                VSIZE_T sample_idx = ${idxs[0]};
                VSIZE_T order = ${idxs[1]} + 1;

                ${alpha.ctype} result = COMPLEX_CTR(${alpha.ctype})(1, 0);
                for (VSIZE_T i = 0; i < ${modes}; i++) {
                    ${alpha.ctype} alpha = ${alpha.load_idx}(sample_idx, i);
                    ${beta.ctype} beta = ${beta.load_idx}(sample_idx, i);
                    ${alpha.ctype} t = ${mul_cc}(alpha, beta);
                    ${alpha.ctype} np = ${exp_c}(COMPLEX_CTR(${alpha.ctype})(-t.x, -t.y));

                    if (i >= order) {
                        result = ${mul_cc}(result, np);
                    }
                    else {
                        ${alpha.ctype} cp = COMPLEX_CTR(${alpha.ctype})(1 - np.x, -np.y);
                        result = ${mul_cc}(result, cp);
                    }
                }

                ${output.store_same}(result.x);
                """,
            render_kwds=dict(
                mul_cc=functions.mul(alpha.dtype, alpha.dtype),
                exp_c=functions.exp(alpha.dtype),
                modes=self._system.modes,
                ))

        reduction = Reduce(for_reduction, predicate_sum(output.dtype), axes=(0,))
        reduction.parameter.input.connect(
            meter_trf, meter_trf.output, alpha_p=meter_trf.alpha, beta_p=meter_trf.beta)

        plan.computation_call(reduction, output, alpha, beta)

        return plan


class CompoundClickProbabilityMeter(Computation):

    def __init__(self, compound_click_probability_meter, system, representation, samples):

        assert representation == Representation.POSITIVE_P
        self._system = system
        self._max_total_clicks = compound_click_probability_meter.max_total_clicks

        state = Type(numpy.complex128, (samples, system.modes))
        output = Type(numpy.float64, (self._max_total_clicks + 1,))
        Computation.__init__(
            self,
            [
                Parameter('output', Annotation(output, 'o')),
                Parameter('alpha', Annotation(state, 'i')),
                Parameter('beta', Annotation(state, 'i')),
            ])

    def _build_plan(self, plan_factory, device_params, output, alpha, beta):

        plan = plan_factory()

        samples, modes = alpha.shape

        for_reduction = Type(alpha.dtype, (samples, self._max_total_clicks + 1))

        prepared_state = plan.temp_array_like(alpha)

        plan.kernel_call(
            TEMPLATE.get_def("compound_click_probability_prepare"),
            [prepared_state, alpha, beta],
            kernel_name="compound_click_probability_prepare",
            global_size=alpha.shape,
            render_kwds=dict(
                mul_cc=functions.mul(alpha.dtype, alpha.dtype),
                exp_c=functions.exp(alpha.dtype),
                ))

        # Block size is limited by the amount of available local memory.
        # In some OpenCL implementations the number reported cannot actually be fully used
        # (because it's used by kernel arguments), so we're padding it a little.
        local_mem_size = device_params.local_mem_size
        max_elems = (local_mem_size - 256) // alpha.dtype.itemsize
        block_size = 2**helpers.log2(max_elems)

        # No reason to have block size larger than the number of modes
        block_size = min(block_size, helpers.bounding_power_of_2(modes))

        products_gsize = (samples, helpers.min_blocks(self._max_total_clicks + 1, block_size) * block_size)
        products = plan.temp_array_like(for_reduction)

        read_size = min(block_size, device_params.max_work_group_size)

        while read_size > 1:

            full_steps = modes // block_size
            remainder_size = modes % block_size

            try:
                plan.kernel_call(
                    TEMPLATE.get_def("compound_click_probability_aggregate"),
                    [products, prepared_state],
                    kernel_name="compound_click_probability_aggregate",
                    global_size=products_gsize,
                    local_size=(1, read_size,),
                    render_kwds=dict(
                        block_size=block_size,
                        read_size=read_size,
                        full_steps=full_steps,
                        remainder_size=remainder_size,
                        output_size=self._max_total_clicks + 1,
                        mul_cc=functions.mul(alpha.dtype, alpha.dtype),
                        add_cc=functions.add(alpha.dtype, alpha.dtype),
                        polar_unit=functions.polar_unit(dtypes.real_for(alpha.dtype)),
                        modes=self._system.modes,
                        max_total_clicks=self._max_total_clicks,
                        ))

            except OutOfResourcesError:
                read_size //= 2

            break

        reduction = Reduce(for_reduction, predicate_sum(alpha.dtype), axes=(0,))

        temp = plan.temp_array_like(reduction.parameter.output)

        plan.computation_call(reduction, temp, products)

        fft = FFT(temp)
        real_trf = Transformation([
            Parameter('output', Annotation(output, 'o')),
            Parameter('input', Annotation(temp, 'i')),
            ],
            """
                ${input.ctype} val = ${input.load_same};
                ${output.store_same}(val.x);
                """)
        fft.parameter.output.connect(real_trf, real_trf.input, output_p=real_trf.output)

        plan.computation_call(fft, output, temp, True)

        return plan


def _prepare_1d(thread, comp_cls, meter, system, representation, samples):

    comp = comp_cls(meter, system, representation, samples)
    ccomp = comp.compile(thread)
    result_dev = thread.empty_like(ccomp.parameter.output)

    def measure(state):
        ccomp(result_dev, state.alpha, state.beta)
        result = result_dev.get()
        return result.reshape(1, result.size) / samples

    return measure


def prepare_population(thread, meter, system, representation, samples):
    return _prepare_1d(thread, PopulationMeter, meter, system, representation, samples)


def prepare_moments(thread, meter, system, representation, samples):
    return _prepare_1d(thread, MomentsMeter, meter, system, representation, samples)


def prepare_click_probability(thread, meter, system, representation, samples):
    return _prepare_1d(thread, ClickProbabilityMeter, meter, system, representation, samples)


def prepare_clicks(thread, meter, system, representation, samples):
    return _prepare_1d(thread, ClicksMeter, meter, system, representation, samples)


def prepare_zero_clicks(thread, meter, system, representation, samples):
    return _prepare_1d(thread, ZeroClicksMeter, meter, system, representation, samples)


def prepare_compound_click_probability(thread, meter, system, representation, samples):
    return _prepare_1d(thread, CompoundClickProbabilityMeter, meter, system, representation, samples)
