import numpy

from squeezed_sim import *


def random_unitary(n, seed=None):
    """
    A random matrix distributed with Haar measure
    """
    rng = numpy.random.RandomState(seed)
    z = (rng.normal(size=(n,n)) + 1j * rng.normal(size=(n,n))) / 2**0.5
    q, r = numpy.linalg.qr(z)
    d = numpy.diagonal(r)
    ph = d / numpy.abs(d)
    q = q @ numpy.diag(ph)
    return q


def add_reference_uniform(system, merged_result_set):

    modes = system.modes

    t = system.transmission
    n = numpy.sinh(system.squeezing)**2 * t
    m = (1 - system.decoherence) * numpy.cosh(system.squeezing) * numpy.sinh(system.squeezing) * t
    p = 1 - (1 / ((n+1)**2 - m**2))**0.5

    n = numpy.concatenate([n, numpy.zeros(system.modes - system.inputs)])
    p = numpy.concatenate([p, numpy.zeros(system.modes - system.inputs)])

    countprob = nchooseft(p.reshape(1, p.size), up_to_mode=system.inputs)
    countprob = numpy.concatenate([countprob[0], numpy.zeros(modes - system.inputs, countprob.dtype)])

    for key, result in merged_result_set.results.items():

        label, stage, representation = key

        if label == 'population':
            merged_result_set.results[key] = result.with_reference(n)

        if label == 'moments' and stage == 'out':
            ref_moments = numpy.empty_like(result.values)
            for i, order in enumerate(result.x_values):
                ref_moments[i] = n[:order].prod()
            merged_result_set.results[key] = result.with_reference(ref_moments)

        if label == 'click_probability' and stage == 'out':
            merged_result_set.results[key] = result.with_reference(p)

        if label == 'clicks' and stage == 'out':
            ref_clicks = numpy.empty_like(result.values)
            for i, order in enumerate(result.x_values):
                ref_clicks[i] = p[:order].prod()
            merged_result_set.results[key] = result.with_reference(ref_clicks)

        if label == 'zero_clicks' and stage == 'out':
            ref_zero_clicks = numpy.empty_like(result.values)
            for i, order in enumerate(result.x_values):
                ref_zero_clicks[i] = p[:order].prod() * (1 - p[order:]).prod()
            merged_result_set.results[key] = result.with_reference(ref_zero_clicks)

        if label == 'compound_click_probability' and stage == 'out':
            merged_result_set.results[key] = result.with_reference(countprob)


def test_thermal(api_id='ocl', device_num=None):

    ensembles = 12
    samples_per_ensemble = 10000

    system = System(
        unitary=random_unitary(40),
        decoherence=1,
        squeezing=1,
        )

    for device_num in (None,) + ((device_num,) if device_num is not None else ()):

        merged_result_set = simulate_sequential(
            system=system,
            ensembles=ensembles,
            samples_per_ensemble=samples_per_ensemble,
            measurements=dict(
                population=Measure(Population()),
                moments=Measure(Moments(5), stages={'out'}),
                click_probability=Measure(ClickProbability(), stages={'out'}, representations={Representation.POSITIVE_P}),
                clicks=Measure(Clicks(5), stages={'out'}, representations={Representation.POSITIVE_P}),
                zero_clicks=Measure(ZeroClicks(5), stages={'out'}, representations={Representation.POSITIVE_P}),
                compound_click_probability=Measure(CompoundClickProbability(40), stages={'out'}, representations={Representation.POSITIVE_P})
            ),
            api_id=api_id,
            device_num=device_num)

        add_reference_uniform(system, merged_result_set)

        plot_results(
            merged_result_set,
            tags=dict(
                population={'lin', 'errors'},
                moments={'log', 'errors'},
                click_probability={'lin', 'errors'},
                clicks={'log', 'errors'},
                zero_clicks={'log', 'errors'},
                compound_click_probability={'lin', 'log', 'errors'}
            ),
            path="figures/thermal_" + ('cpu' if device_num is None else 'gpu'))


def test_squeezed(api_id='ocl', device_num=None):

    ensembles = 12
    samples_per_ensemble = 10000

    system = System(
        unitary=numpy.eye(20),
        inputs=10,
        squeezing=2,
        transmission=0.5,
        decoherence=0.1,
        )

    for device_num in (None,) + ((device_num,) if device_num is not None else ()):

        merged_result_set = simulate_sequential(
            system=system,
            ensembles=ensembles,
            samples_per_ensemble=samples_per_ensemble,
            measurements=dict(
                population=Measure(Population()),
                moments=Measure(Moments(5), stages={'out'}),
                click_probability=Measure(ClickProbability(), stages={'out'}, representations={Representation.POSITIVE_P}),
                clicks=Measure(Clicks(5), stages={'out'}, representations={Representation.POSITIVE_P}),
                zero_clicks=Measure(ZeroClicks(5), stages={'out'}, representations={Representation.POSITIVE_P}),
                compound_click_probability=Measure(CompoundClickProbability(20), stages={'out'}, representations={Representation.POSITIVE_P})
                ),
            api_id=api_id,
            device_num=device_num)

        add_reference_uniform(system, merged_result_set)

        plot_results(
            merged_result_set,
            tags=dict(
                population={'lin', 'errors'},
                moments={'log', 'errors'},
                click_probability={'lin', 'errors'},
                clicks={'log', 'errors'},
                zero_clicks={'log', 'errors'},
                compound_click_probability={'lin', 'log', 'errors'}
            ),
            path="figures/squeezed_" + ('cpu' if device_num is None else 'gpu'))


if __name__ == '__main__':
    test_thermal(api_id='ocl', device_num=2)
    test_squeezed(api_id='ocl', device_num=2)
