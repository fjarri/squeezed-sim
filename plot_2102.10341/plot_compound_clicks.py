from pathlib import Path
import pickle

import numpy

from squeezed_sim import *
from squeezed_sim.simulate_large_scale import simulate_mp
from squeezed_sim.simulate import ResultSet
import squeezed_sim.utils.mplhelpers as mplh


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

    else:
        print(f"Preparing unitary in {fname}...")
        unitary = random_unitary(parameter_set['modes'], seed=parameter_set['seed'])
        print("    done.")
        numpy.save(fname, unitary)

    return unitary


def run_simulation(parameter_set, results_dir, api_id='ocl', gpu_name_filters=[]):

    modes = parameter_set['modes']
    ensembles = parameter_set['ensembles']
    samples_per_ensemble = parameter_set['samples_per_ensemble']
    seed = parameter_set['seed']

    results_fname = f"{results_dir}/result_sets modes={modes} seed={seed}.pickle"

    if Path(results_fname).exists():
        with open(results_fname, 'rb') as f:
            result_sets = pickle.load(f)

        if len(result_sets) >= ensembles:
            return ResultSet.merge(list(result_sets.values())[:ensembles])

    rng = numpy.random.RandomState(seed)

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
        seed=seed,
        api_id=api_id,
        gpu_name_filters=gpu_name_filters)

    return merged_result_set


def run(results_dir, api_id='ocl', gpu_name_filters=[]):

    parameter_sets = [
        dict(modes=16, ensembles=100, samples_per_ensemble=1000000, seed=123),
        #dict(modes=32, ensembles=100, samples_per_ensemble=10000, seed=123),
        dict(modes=64, ensembles=100, samples_per_ensemble=10000, seed=123),
        #dict(modes=128, ensembles=100, samples_per_ensemble=10000, seed=123),
        dict(modes=256, ensembles=100, samples_per_ensemble=1000, seed=123),
        #dict(modes=512, ensembles=100, samples_per_ensemble=1000, seed=123),
        dict(modes=1024, ensembles=100, samples_per_ensemble=160, seed=123),
        #dict(modes=2048, ensembles=100, samples_per_ensemble=80, seed=123),
        dict(modes=4096, ensembles=100, samples_per_ensemble=40, seed=123),
        #dict(modes=8192, ensembles=100, samples_per_ensemble=20, seed=123),
        dict(modes=16384, ensembles=100, samples_per_ensemble=20, seed=123),
    ]

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    width = 1 # 1 column
    column_width_inches = 8.6 / 2.54 # PRL single column
    aspect = (numpy.sqrt(5) - 1) / 2
    fig_width = column_width_inches * width
    fig_height = fig_width * aspect # height in inches

    merged_result_sets = []
    for parameter_set in parameter_sets:
        merged_result_set = run_simulation(parameter_set, results_dir, api_id=api_id, gpu_name_filters=gpu_name_filters)
        merged_result_sets.append(merged_result_set)


    linestyles = ['-', '--', ':', '-.', '--.', '-..']

    fig = mplh.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.set_yscale('log')
    sp.set_xlim(0, 1)
    limit = 1e-7
    sp.set_ylim(limit, 1e2)


    for i, (ps, mrs) in enumerate(zip(parameter_sets, merged_result_sets)):
        results = mrs.results[('compound_click_probability', 'out', Representation.POSITIVE_P)]
        modes = ps['modes']
        x = results.x_values / modes
        scale = 1 / modes # results.values.max()
        y = results.values / scale
        yerr = results.errors / scale

        idx_min = numpy.argmax(y > limit)
        idx_max = y.size - 1 - numpy.argmax(y[::-1] > limit)

        x = x[idx_min:idx_max]
        y = y[idx_min:idx_max]
        yerr = yerr[idx_min:idx_max]

        sp.fill_between(x, y - yerr, y + yerr, alpha=0.3, linewidth=0, facecolor='grey')
        label = f"$M=2^{{{numpy.log2(ps['modes']).astype(numpy.int32)}}}$"
        sp.plot(x, y, label=label, dashes=mplh.dash[linestyles[i]])

    sp.legend(handlelength=3)
    sp.set_xlabel('$m / M$')
    sp.set_ylabel('$M \\mathcal{P}_M(m)$')

    fig.tight_layout(pad=0.1)
    fig.savefig(str(Path('figures') / "compound_click_probability_log.pdf"))
    mplh.close(fig)


if __name__ == '__main__':

    figures_dir = Path('figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    run("compound_clicks_cache", api_id='ocl', gpu_name_filters=['Vega'])
