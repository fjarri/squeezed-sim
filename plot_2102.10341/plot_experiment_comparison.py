from pathlib import Path

import numpy

from squeezed_sim import *
from squeezed_sim.experiment import *
import squeezed_sim.utils.mplhelpers as mplh


def plot_experiment(api_id='ocl', device_num=0):

    ensembles = 120
    samples_per_ensemble = 10000

    system = System(
        unitary=transmission_matrix(),
        inputs=50,
        squeezing=squeezing_coefficients(),
        decoherence=0.1,
        )

    merged_result_set = simulate_sequential(
        system=system,
        ensembles=ensembles,
        samples_per_ensemble=samples_per_ensemble,
        measurements=dict(
            click_probability=Measure(ClickProbability(), stages={'out'}, representations={Representation.POSITIVE_P}),
            compound_click_probability=Measure(CompoundClickProbability(100), stages={'out'}, representations={Representation.POSITIVE_P})
            ),
        api_id=api_id,
        device_num=device_num)

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
    fig.savefig(str(Path('figures') / "experimental_comparison_clicks.pdf"))
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
    sp.set_ylabel('$\\mathcal{G}(m)$')

    fig.tight_layout(pad=0.1)
    fig.savefig(str(Path('figures') / "experimental_comparison_compound.pdf"))
    mplh.close(fig)


if __name__ == '__main__':

    figures_dir = Path('figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_experiment(api_id='ocl', device_num=2)
