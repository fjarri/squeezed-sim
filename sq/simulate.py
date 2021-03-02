from collections import defaultdict
from functools import partial

import numpy

from reikna.cluda import ocl_api

from .system import Representation
from .meters_cpu import Result
from . import generate_cpu
from . import generate_gpu


class Measure:

    def __init__(self, meter, stages=None, representations=None):
        if stages is None:
            stages = {'in', 'out'}
        if representations is None:
            representations = {Representation.POSITIVE_P, Representation.WIGNER, Representation.Q}

        self.stages = stages
        self.representations = representations
        self.meter = meter


class ResultSet:

    def __init__(self):
        self.results = defaultdict(list)

    def add_result(self, result, label, stage, representation):
        assert isinstance(result, Result)
        self.results[(label, stage, representation)].append(result)

    def add_result_set(self, result_set):
        for (label, stage, representation), results in result_set.results.items():
            for result in results:
                self.add_result(result, label, stage, representation)

    def merge_results(self):
        return MergedResultSet({key: Result.merge(chunks) for key, chunks in self.results.items()})

    @staticmethod
    def merge(result_sets):
        result_set = ResultSet()
        for rs in result_sets:
            result_set.add_result_set(rs)
        return result_set.merge_results()


class MergedResultSet:

    def __init__(self, results):
        self.results = results


def used_representations(meters):
    return set.union(*[m.representations for m in meters.values()])


def make_seed(rng):
    return rng.randint(0, 2**32-1)


def worker_gpu(system, samples, measurements, seed, repetitions=1, gpu_id=0):

    result_set = ResultSet()
    rng = numpy.random.RandomState(seed)

    api = ocl_api()
    device = api.get_platforms()[0].get_devices()[gpu_id]
    thread = api.Thread(device)

    for representation in used_representations(measurements):

        generate_input_state = generate_gpu.prepare_generate_input_state(thread, system, representation, samples)
        apply_matrix = generate_gpu.prepare_apply_matrix(thread, system, representation, samples)
        prepared_meters = {
            label: measurement.meter.prepare_gpu(thread, system, representation, samples)
            for label, measurement in measurements.items()
            if representation in measurement.representations}

        for repetition in range(repetitions):
            in_state = generate_input_state(make_seed(rng))
            out_state = apply_matrix(in_state, make_seed(rng))

            for label, measurement in measurements.items():
                if 'in' in measurement.stages and representation in measurement.representations:
                    result = prepared_meters[label](in_state)
                    result_set.add_result(result, label, 'in', representation)

                if 'out' in measurement.stages and representation in measurement.representations:
                    result = prepared_meters[label](out_state)
                    result_set.add_result(result, label, 'out', representation)

    return result_set


def worker_cpu(system, samples, measurements, seed, repetitions=1):

    result_set = ResultSet()
    rng = numpy.random.RandomState(seed)

    for representation in used_representations(measurements):
        for repetition in range(repetitions):
            in_state = generate_cpu.generate_input_state(system, representation, samples, make_seed(rng))
            out_state = generate_cpu.apply_matrix(in_state, make_seed(rng))

            for label, measurement in measurements.items():
                if 'in' in measurement.stages and representation in measurement.representations:
                    result = measurement.meter(in_state)
                    result_set.add_result(result, label, 'in', representation)

                if 'out' in measurement.stages and representation in measurement.representations:
                    result = measurement.meter(out_state)
                    result_set.add_result(result, label, 'out', representation)

    return result_set


def simulate_sequential(system, ensembles, samples_per_ensemble=1, measurements={}, seed=None, gpu_id=None):

    rng = numpy.random.RandomState(seed)

    if gpu_id is None:
        worker = worker_cpu
    else:
        worker = partial(worker_gpu, gpu_id=gpu_id)

    seed = make_seed(rng)
    result_set = worker(system, samples_per_ensemble, measurements, seed, repetitions=ensembles)

    return result_set.merge_results()
