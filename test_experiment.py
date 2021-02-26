import csv

import numpy
from scipy.io import loadmat

from sq.simulate import simulate_sequential, Measure
from sq.meters_cpu import *
from sq.system import System, Representation
from sq.plot import plot_results


def transmission_matrix():

    re = []
    with open('experimental_data/matrix_re.csv') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        for row in r:
            re.append([float(x) for x in row])
    re = numpy.array(re)

    im = []
    with open('experimental_data/matrix_im.csv') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        for row in r:
            im.append([float(x) for x in row])
    im = numpy.array(im)

    U = re + 1j * im

    if U.shape[0] < U.shape[1]:
        U = numpy.concatenate([U, numpy.zeros((U.shape[1] - U.shape[0], U.shape[1]))], axis=0)

    return U.transpose()


def squeezing_coefficients():

    res = []
    with open('experimental_data/squeezing_parametersq.csv') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        for row in r:
            for x in row:
                res.append(float(x))

    res = numpy.array(res)

    return res


def add_reference_experiment(system, merged_result_set):
    res = loadmat('experimental_data/exp_cp.mat')

    for key, result in merged_result_set.results.items():

        label, stage, representation = key

        if label == 'click_probability' and stage == 'out':
            merged_result_set.results[key] = result.with_reference(res['Cp'][3][0][:,0])

        if label == 'clicks' and stage == 'out':
            merged_result_set.results[key] = result.with_reference(res['Cp'][4][0][:,0])

        if label == 'compound_click_probability' and stage == 'out':
            merged_result_set.results[key] = result.with_reference(res['Cp'][6][0][:,0])


def test_experiment():

    ensembles = 120
    samples_per_ensemble = 1000

    system = System(
        unitary=transmission_matrix(),
        inputs=50,
        squeezing=squeezing_coefficients(),
        )

    for gpu_id in (None, 2,):
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
                compound_click_probability=Measure(CompoundClickProbability(100), stages={'out'}, representations={Representation.POSITIVE_P})
                ),
            gpu_id=gpu_id)

        add_reference_experiment(system, merged_result_set)

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
            path="figures/experiment_" + ('cpu' if gpu_id is None else 'gpu'))


def test_experiment_dc():

    ensembles = 120
    samples_per_ensemble = 1000
    dc = 0.12

    system = System(
        unitary=transmission_matrix(),
        inputs=50,
        squeezing=squeezing_coefficients(),
        input_transmission=1 / (1 + dc)**0.5,
        thermal_noise=dc * numpy.sinh(squeezing_coefficients())**2,
        )

    for gpu_id in (None, 2,):
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
                compound_click_probability=Measure(CompoundClickProbability(100), stages={'out'}, representations={Representation.POSITIVE_P})
                ),
            gpu_id=gpu_id)

        add_reference_experiment(system, merged_result_set)

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
            path="figures/experiment_dc_" + ('cpu' if gpu_id is None else 'gpu'))


if __name__ == '__main__':
    test_experiment()
    test_experiment_dc()
