import numpy
from sq import *

import mplhelpers as mplh
from test_experiment import transmission_matrix, squeezing_coefficients, add_reference_experiment


def plot_experiment():

    ensembles = 120
    samples_per_ensemble = 10000
    dc = 0.12

    system = System(
        unitary=transmission_matrix(),
        inputs=50,
        squeezing=squeezing_coefficients(),
        input_transmission=1 / (1 + dc)**0.5,
        thermal_noise=dc * numpy.sinh(squeezing_coefficients())**2,
        )

    merged_result_set = simulate_sequential(
        system=system,
        ensembles=ensembles,
        samples_per_ensemble=samples_per_ensemble,
        measurements=dict(
            click_probability=Measure(ClickProbability(), stages={'out'}, representations={Representation.POSITIVE_P}),
            compound_click_probability=Measure(CompoundClickProbability(100), stages={'out'}, representations={Representation.POSITIVE_P})
            ))

    add_reference_experiment(system, merged_result_set)

    fig = mplh.figure()
    sp = fig.add_subplot(1, 1, 1)
    result = merged_result_set.results[('click_probability', 'out', Representation.POSITIVE_P)]

    sp.set_xlim(0, 100)
    sp.fill_between(result.x_values, result.values - result.errors, result.values + result.errors,
        linewidth=0, facecolor='grey')
    sp.plot(result.x_values, result.values, label='simulation')
    sp.plot(result.x_values, result.reference, dashes=mplh.dash['--'], label='experiment')
    sp.legend()
    sp.set_xlabel('$j$')
    sp.set_ylabel('$\\langle \\pi_j(1) \\rangle$')

    fig.tight_layout(pad=0.1)
    fig.savefig(f"experimental_comparison_clicks.pdf")
    mplh.close(fig)


    fig = mplh.figure()
    sp = fig.add_subplot(1, 1, 1)

    result = merged_result_set.results[('compound_click_probability', 'out', Representation.POSITIVE_P)]
    sp.set_yscale('log')
    sp.set_xlim(0, 100)
    sp.set_ylim(1e-7, 1e-1)
    sp.fill_between(result.x_values, result.values - result.errors, result.values + result.errors,
        linewidth=0, facecolor='grey')
    sp.plot(result.x_values, result.values, label='simulation')
    sp.plot(result.x_values, result.reference, dashes=mplh.dash['--'], label='experiment')
    sp.legend()
    sp.set_xlabel('$m$')
    sp.set_ylabel('$\\mathcal{P}(m)$')

    fig.tight_layout(pad=0.1)
    fig.savefig(f"experimental_comparison_compound.pdf")
    mplh.close(fig)


if __name__ == '__main__':
    plot_experiment()
