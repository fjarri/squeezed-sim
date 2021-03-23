import numpy

from squeezed_sim import *
from squeezed_sim.experiment import *


def test_experiment_dc(api_id='ocl', device_num=None):

    ensembles = 120
    samples_per_ensemble = 1000

    system = System(
        unitary=transmission_matrix(),
        inputs=50,
        squeezing=squeezing_coefficients(),
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
                compound_click_probability=Measure(CompoundClickProbability(100), stages={'out'}, representations={Representation.POSITIVE_P})
                ),
            api_id=api_id,
            device_num=device_num)

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
            path="figures/experiment_dc_" + ('cpu' if device_num is None else 'gpu'))


if __name__ == '__main__':
    test_experiment_dc(api_id='ocl', device_num=2)
