from pathlib import Path
import pickle
import matplotlib.pyplot as plt

import numpy

from sq import *
from sq.simulate_large_scale import simulate_mp


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


def prepare_unitary(parameter_set, results_dir):

    rng = numpy.random.RandomState(parameter_set['seed'])
    seed = make_seed(rng)

    fname = f"{results_dir}/unitary modes={parameter_set['modes']} seed={parameter_set['seed']}.npy"

    if Path(fname).exists():
        unitary = numpy.load(fname)
        #with open(fname, 'rb') as f:
        #    unitary = pickle.load(f)

    else:
        print(f"Preparing unitary in {fname}...")
        unitary = random_unitary(parameter_set['modes'], seed=parameter_set['seed'])
        print("    done.")
        numpy.save(fname, unitary)
        #with open(fname, 'wb') as f:
        #    pickle.dump(unitary, f)

    return unitary


def run_simulation(parameter_set, results_dir):

    modes = parameter_set['modes']
    ensembles = parameter_set['ensembles']
    samples_per_ensemble = parameter_set['samples_per_ensemble']

    rng = numpy.random.RandomState(parameter_set['seed'])
    seed = make_seed(rng)

    system = System(
        unitary=prepare_unitary(parameter_set, results_dir),
        inputs=modes // 2,
        squeezing=1,
        )

    merged_result_set = simulate_mp(
        dirname=results_dir,
        system=system,
        ensembles=ensembles,
        samples_per_ensemble=samples_per_ensemble,
        measurements=dict(
            compound_click_probability=Measure(
                CompoundClickProbability(modes),
                stages={'out'}, representations={Representation.POSITIVE_P})
            ),
        seed=seed)

    return merged_result_set


def run():

    results_dir = 'compound_click_probability'

    parameter_sets = [
        dict(modes=16, ensembles=100, samples_per_ensemble=10000, seed=123),
        dict(modes=32, ensembles=100, samples_per_ensemble=10000, seed=123),
        dict(modes=64, ensembles=100, samples_per_ensemble=10000, seed=123),
        dict(modes=128, ensembles=100, samples_per_ensemble=10000, seed=123),
        dict(modes=256, ensembles=100, samples_per_ensemble=1000, seed=123),
        dict(modes=512, ensembles=100, samples_per_ensemble=1000, seed=123),
        dict(modes=1024, ensembles=100, samples_per_ensemble=160, seed=123),
        dict(modes=2048, ensembles=100, samples_per_ensemble=80, seed=123),
        dict(modes=4096, ensembles=100, samples_per_ensemble=40, seed=123),
        dict(modes=8192, ensembles=100, samples_per_ensemble=20, seed=123),
        dict(modes=16384, ensembles=100, samples_per_ensemble=20, seed=123),
    ]

    results_dir = "compound_clicks_results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    merged_result_sets = []
    for parameter_set in parameter_sets:
        merged_result_set = run_simulation(parameter_set, results_dir)
        merged_result_sets.append(merged_result_set)


    fig = plt.figure(figsize=(12, 8))
    sp = fig.add_subplot(1, 1, 1)

    for ps, mrs in zip(parameter_sets, merged_result_sets):
        results = mrs.results[('compound_click_probability', 'out', Representation.POSITIVE_P)]
        x = results.x_values / ps['modes']
        sp.errorbar(x, results.values, yerr=results.errors, label=f"M={ps['modes']}")

    sp.legend()
    sp.set_xlabel('$n / M$')
    sp.set_ylabel('$P(n)$')

    fig.tight_layout()
    fig.savefig(f"{results_dir}/plot.pdf")
    plt.close(fig)


if __name__ == '__main__':
    run()
