import numpy

from . import meters_gpu
from .system import ordering, Representation


class Result:

    @classmethod
    def from_samples(cls, x_values, values, x_label=None, y_label=None):

        assert x_values.ndim == 1
        assert values.shape[1:] == x_values.shape
        means = values.mean(0)
        # TODO: we could save (values**2).mean() here along with the number of samples
        # to be able to join chunks with different numbers of samples.

        return cls(x_values=x_values, values=means, x_label=x_label, y_label=y_label)

    def __init__(
            self, x_values, values, errors=None, reference=None,
            x_label=None, y_label=None):
        self.x_values = x_values
        self.values = values
        self.errors = errors
        self.reference = reference
        self.x_label = x_label
        self.y_label = y_label

    @staticmethod
    def merge(results):

        if results[0] is None:
            return None

        values = numpy.stack([m.values for m in results])
        means = values.mean(0)
        errors = values.std(0) / values.shape[0]**0.5

        return Result(
            x_values=results[0].x_values,
            values=means,
            errors=errors,
            reference=results[0].reference,
            x_label=results[0].x_label,
            y_label=results[0].y_label)

    def with_reference(self, reference):
        assert reference.shape == self.x_values.shape
        return Result(
            x_values=self.x_values,
            values=self.values,
            errors=self.errors,
            reference=reference,
            x_label=self.x_label,
            y_label=self.y_label)


class PrepareGPUMixin:

    def prepare_gpu(self, thread, system, representation, samples):

        meter = self._gpu_meter()(thread, self, system, representation, samples)

        def measure(state):
            raw_result = meter(state)

            # FIXME: temporary measure until all meters are implemented
            if isinstance(raw_result, Result):
                return raw_result

            return self._make_result(raw_result)

        return measure


class Population(PrepareGPUMixin):

    def __call__(self, state):
        values = (state.alpha * state.beta - ordering(state.representation)).real
        return self._make_result(values)

    def _gpu_meter(self):
        return meters_gpu.prepare_population

    def _make_result(self, values):
        return Result.from_samples(
            numpy.arange(1, values.shape[1] + 1),
            values,
            x_label='mode $m$',
            y_label='$<n(m)>$',
            )


class Moments(PrepareGPUMixin):

    def __init__(self, max_moment):
        self.max_moment = max_moment

    def __call__(self, state):
        res = numpy.empty((state.samples, self.max_moment))
        np2 = state.alpha * state.beta - ordering(state.representation)
        for order in range(1, self.max_moment + 1):
            res[:, order - 1] = np2[:,:order].prod(1).real
        return self._make_result(res)

    def _gpu_meter(self):
        return meters_gpu.prepare_moments

    def _make_result(self, values):
        return Result.from_samples(
            numpy.arange(1, self.max_moment + 1),
            values,
            x_label='order $o$',
            y_label='<moment>',
            )


class ClickProbability(PrepareGPUMixin):

    def __call__(self, state):
        if state.representation != Representation.POSITIVE_P:
            return None

        np2 = state.alpha * state.beta - ordering(state.representation)
        np = numpy.exp(-np2)
        cp = 1 - np
        values = cp.real
        return self._make_result(values)

    def _gpu_meter(self):
        return meters_gpu.prepare_click_probability

    def _make_result(self, values):
        return Result.from_samples(
            numpy.arange(1, values.shape[1] + 1),
            values,
            x_label='mode $m$',
            y_label='$<p(m)>$',
            )


class Clicks(PrepareGPUMixin):

    def __init__(self, max_click_order):
        self.max_click_order = max_click_order

    def __call__(self, state):
        if state.representation != Representation.POSITIVE_P:
            return None

        np2 = state.alpha * state.beta - ordering(state.representation)
        np = numpy.exp(-np2)
        cp = 1 - np

        res = numpy.zeros((state.samples, self.max_click_order))
        for order in range(1, self.max_click_order + 1):
            res[:, order - 1] = cp[:, :order].prod(1).real

        return self._make_result(res)

    def _gpu_meter(self):
        return meters_gpu.prepare_clicks

    def _make_result(self, values):
        return Result.from_samples(
            numpy.arange(1, self.max_click_order + 1),
            values,
            x_label='Clicks $n$',
            y_label='$T(n)$',
            )


class ZeroClicks(PrepareGPUMixin):

    def __init__(self, max_click_order):
        self.max_click_order = max_click_order

    def __call__(self, state):
        if state.representation != Representation.POSITIVE_P:
            return None

        np2 = state.alpha * state.beta - ordering(state.representation)
        np = numpy.exp(-np2)
        cp = 1 - np

        res = numpy.zeros((state.samples, self.max_click_order))
        for order in range(1, self.max_click_order + 1):
            res[:, order - 1] = (cp[:, :order].prod(1) * np[:, order:].prod(1)).real

        return self._make_result(res)

    def _gpu_meter(self):
        return meters_gpu.prepare_zero_clicks

    def _make_result(self, values):
        return Result.from_samples(
            numpy.arange(1, self.max_click_order + 1),
            values,
            x_label='clicks $n$',
            y_label='$T(n|0)$',
            )


def nchooseft(p, up_to_mode=None, average_to=1):
#  CF = NCHOOSEFT(M,p) calculates weighted binomial coefficients.
#  If p is a scalar it returns the successive coefficients of x^n in the
#  binomial expansion of (xp+(1-p))^M. If p is an M-vector,
#  it returns successive coefficients of x^n in prod_j(xp_j+(1-p_j))
#  If p has a second dimension, it averages the result over this.
#  Results are accurate to 15 decimals only

#  test case: suppose p =0.5, then nchooseft(M,p) ~ 2^(-M)*nchoosek(M,k)
#   CF = nchooseft(M,0.5)
#   for k=0:M
#       p2 = nchoosek(M,k)*2^(-M);
#       fprintf( 'k %d, fft %15.12e, test %15.12e, delt %15.12e\n',...
#       k,CF(k+1),p2, abs(CF(k+1)-p2));
#   end

    assert p.ndim == 2
    assert p.shape[0] % average_to == 0

    if up_to_mode is None:
        up_to_mode = p.shape[1]

    assert (p[:, up_to_mode:] == 0).all()

    cf = numpy.empty((average_to, up_to_mode + 1), numpy.complex128)
    theta = 2 * numpy.pi / (up_to_mode + 1)

    for k in range(up_to_mode + 1):
        cpk = (1 - p) + p * numpy.exp(-1j * theta * k)       #Get M factors p(j,:)
        probs = cpk[:,:up_to_mode].prod(1)                    # Get product over M factors

        probs = probs.reshape(p.shape[0] // average_to, average_to)

        cf[:, k] = probs.mean(0)             # Average over ensemble

    return numpy.fft.ifft(cf).real # Take inverse Fourier transform


class CompoundClickProbability(PrepareGPUMixin):

    def __init__(self, max_total_clicks):
        self.max_total_clicks = max_total_clicks

    def __call__(self, state):
        if state.representation != Representation.POSITIVE_P:
            return None

        np2 = state.alpha * state.beta - ordering(state.representation)
        np = numpy.exp(-np2)
        cp = 1 - np

        cf = nchooseft(cp, up_to_mode=self.max_total_clicks)
        res = numpy.zeros((1, self.max_total_clicks + 1))
        for order in range(self.max_total_clicks + 1):
            res[0, order] = cf[0, order]

        return self._make_result(res)

    def _gpu_meter(self):
        return meters_gpu.prepare_compound_click_probability

    def _make_result(self, values):
        return Result.from_samples(
            numpy.arange(0, self.max_total_clicks + 1),
            values,
            x_label='clicks $n$',
            y_label='$P(n)$',
            )
