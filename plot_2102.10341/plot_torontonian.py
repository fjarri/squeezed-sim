from itertools import combinations
import os
from pathlib import Path
import pickle

from tqdm import tqdm
import numpy

from thewalrus import tor
from thewalrus.symplectic import interferometer
from thewalrus.quantum import Qmat, Amat, Xmat

import squeezed_sim.utils.mplhelpers as mplh
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


def reference_covariance(system):
    squeezing = numpy.concatenate([system.squeezing, numpy.zeros(system.modes - system.inputs)])
    cov = numpy.diag(numpy.concatenate([
        numpy.exp(2*squeezing),
        numpy.exp(-2*squeezing)])) # covariance matrix before interferometer
    interferom = interferometer(system.unitary)
    cov = interferom @ cov @ interferom.T # covariance matrix after interferometer
    return cov


def reference_normalization(cov):
    return 1 / numpy.sqrt(numpy.linalg.det(Qmat(cov)).real)


def reference_count_unnormalized(cov, only_modes=None):
    A = Amat(cov)
    modes = cov.shape[0] // 2

    if only_modes is not None:
        mask = only_modes + tuple(x + modes for x in only_modes)
        A = A[mask,:][:,mask]
        xmat_size = len(only_modes)
    else:
        xmat_size = modes

    O = Xmat(xmat_size) @ A
    return tor(O).real


def reference_max_count(system):
    """
    Calculates the value of getting the maximum click count (== system.modes)
    analytically using the Torontonian.
    """
    cov = reference_covariance(system)
    c = reference_count_unnormalized(cov)
    n = reference_normalization(cov)
    return c * n


def plot_varying_inputs(api_id='ocl', device_num=0):

    ensembles = 12
    samples_per_ensemble = 100000
    modes = 16
    unitary = random_unitary(modes)

    input_values = numpy.arange(1, modes + 1)
    max_count_values = []
    max_count_errors = []
    reference_values = []

    for inputs in input_values:
        system = System(
            unitary=unitary,
            inputs=inputs,
            squeezing=1,
            )

        merged_result_set = simulate_sequential(
            system=system,
            ensembles=ensembles,
            samples_per_ensemble=samples_per_ensemble,
            measurements=dict(
                compound_click_probability=Measure(
                    CompoundClickProbability(system.modes),
                    stages={'out'}, representations={Representation.POSITIVE_P})
                ),
            api_id=api_id,
            device_num=device_num)

        result = merged_result_set.results[('compound_click_probability', 'out', Representation.POSITIVE_P)]

        max_count_values.append(result.values[-1])
        max_count_errors.append(result.errors[-1])

        reference_values.append(reference_max_count(system))

    max_count_values = numpy.array(max_count_values)
    max_count_errors = numpy.array(max_count_errors)
    reference_values = numpy.array(reference_values)

    fig = mplh.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.set_xlim(0, modes)
    sp.set_xlabel('inputs')
    sp.set_ylabel('$\\mathcal{P}_M(M)$')

    sp.fill_between(
        input_values, max_count_values - max_count_errors, max_count_values + max_count_errors,
        linewidth=0, facecolor='grey')
    sp.plot(input_values, max_count_values, label='simulation')
    sp.plot(input_values, reference_values, dashes=mplh.dash['--'], label='analytical')

    sp.legend()

    fig.tight_layout(pad=0.1)
    fig.savefig('figures/torontonian/torontonian_varying_inputs_lin.pdf')


    fig = mplh.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.set_xlim(0, modes)
    sp.set_yscale('log')
    sp.set_xlabel('inputs')
    sp.set_ylabel('$\\mathcal{P}_M(M)$')

    sp.fill_between(
        input_values, max_count_values - max_count_errors, max_count_values + max_count_errors,
        linewidth=0, facecolor='grey')
    sp.plot(input_values, max_count_values - max_count_errors, color='grey', dashes=mplh.dash[':'])
    sp.plot(input_values, max_count_values + max_count_errors, color='grey', dashes=mplh.dash[':'])
    sp.plot(input_values, max_count_values, label='simulation')
    sp.plot(input_values, reference_values, dashes=mplh.dash['--'], label='analytical')

    sp.legend()

    fig.tight_layout(pad=0.1)
    fig.savefig('figures/torontonian/torontonian_varying_inputs_log.pdf')


def reference_varying_counts(system):

    fname = 'cache_torontonian/reference_varying_counts.pickle'
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    result_ref = numpy.zeros(system.modes + 1, numpy.float64)
    cov = reference_covariance(system)
    n = reference_normalization(cov)
    for i in range(0, system.modes + 1):
        print("Length", i)
        for combination in tqdm(combinations(list(range(system.modes)), i)):
            result_ref[i] += reference_count_unnormalized(cov, only_modes=combination)
        result_ref[i] *= n

    with open(fname, 'wb') as f:
        pickle.dump(result_ref, f)

    return result_ref


def plot_varying_counts(api_id='ocl', device_num=0):

    ensembles = 100
    samples_per_ensemble = 200000
    modes = 16
    inputs = 8
    unitary = random_unitary(modes, seed=123)

    system = System(
        unitary=unitary,
        inputs=inputs,
        squeezing=1,
        )

    merged_result_set = simulate_sequential(
        system=system,
        ensembles=ensembles,
        samples_per_ensemble=samples_per_ensemble,
        measurements=dict(
            compound_click_probability=Measure(
                CompoundClickProbability(system.modes),
                stages={'out'}, representations={Representation.POSITIVE_P})
            ),
        api_id=api_id,
        device_num=device_num)

    result = merged_result_set.results[('compound_click_probability', 'out', Representation.POSITIVE_P)]

    result_ref = reference_varying_counts(system)

    fig = mplh.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.set_xlim(0, modes)
    sp.set_xlabel('$m$')
    sp.set_ylabel('$\\mathcal{P}_M(m)$')

    sp.fill_between(
        result.x_values, result.values - result.errors, result.values + result.errors,
        linewidth=0, facecolor='grey')
    sp.plot(result.x_values, result.values, label='simulation')
    sp.plot(result.x_values, result_ref, dashes=mplh.dash['--'], label='analytical')

    sp.legend()

    fig.tight_layout(pad=0.1)
    fig.savefig('figures/torontonian/torontonian_varying_counts_lin.pdf')


    fig = mplh.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.set_yscale('log')
    sp.set_xlim(0, modes)
    sp.set_xlabel('$m$')
    sp.set_ylabel('$\\mathcal{P}_M(m)$')

    sp.fill_between(
        result.x_values, result.values - result.errors, result.values + result.errors,
        linewidth=0, facecolor='grey')
    sp.plot(result.x_values, result.values, label='simulation')
    sp.plot(result.x_values, result_ref, dashes=mplh.dash['--'], label='analytical')

    sp.legend()

    fig.tight_layout(pad=0.1)
    fig.savefig('figures/torontonian/torontonian_varying_counts_log.pdf')


    fig = mplh.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.set_xlim(0, modes)
    sp.set_xlabel('$m$')

    sp.plot(result.x_values, result.errors, label='errors($\\mathcal{P}_M(m)$)')
    sp.plot(result.x_values, numpy.abs(result.values - result_ref),
        dashes=mplh.dash['--'], label='$|\\mathcal{P}_M(m) - \\mathcal{P}_M^{\\mathrm{(ref)}}(m)|$')

    sp.legend()

    fig.tight_layout(pad=0.1)
    fig.savefig('figures/torontonian/torontonian_varying_counts_errors_lin.pdf')


    fig = mplh.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.set_yscale('log')
    sp.set_xlim(0, modes)
    sp.set_xlabel('$m$')

    sp.plot(result.x_values, result.errors, label='errors($\\mathcal{P}_M(m)$)')
    sp.plot(result.x_values, numpy.abs(result.values - result_ref),
        dashes=mplh.dash['--'], label='$|\\mathcal{P}_M(m) - \\mathcal{P}_M^{\\mathrm{(ref)}}(m)|$')

    sp.legend()

    fig.tight_layout(pad=0.1)
    fig.savefig('figures/torontonian/torontonian_varying_counts_errors_log.pdf')


if __name__ == '__main__':

    figures_dir = Path('figures/torontonian')
    figures_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = Path('cache_torontonian')
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_varying_inputs(api_id='ocl', device_num=2)
    plot_varying_counts(api_id='ocl', device_num=2)
