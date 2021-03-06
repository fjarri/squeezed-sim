import pickle
from pathlib import Path
#from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue

import numpy

from reikna.cluda import ocl_api

from .system import Representation
from .meters_cpu import Result
from . import generate_gpu
from .simulate import make_seed, used_representations, ResultSet


def worker_gpu(worker_id, out_queue, in_queue, system, samples, measurements, gpu_id):

    api = ocl_api()
    device = api.get_platforms()[0].get_devices()[gpu_id]
    thread = api.Thread(device)

    prepared_calls = {}

    for representation in used_representations(measurements):

        generate_input_state = generate_gpu.prepare_generate_input_state(thread, system, representation, samples)
        apply_matrix = generate_gpu.prepare_apply_matrix(thread, system, representation, samples)
        prepared_meters = {
            label: measurement.meter.prepare_gpu(thread, system, representation, samples)
            for label, measurement in measurements.items()
            if representation in measurement.representations}
        prepared_calls[representation] = dict(
            generate_input_state=generate_input_state,
            apply_matrix=apply_matrix,
            prepared_meters=prepared_meters)

    while True:

        command = in_queue.get()

        if command == 'stop':
            print(f"Worker {worker_id}: stopping")
            break

        seed = command
        result_set = ResultSet()

        print(f"{worker_id}: processing seed {seed}")

        for representation in used_representations(measurements):

            #print(f"{worker_id}: processing representation {representation}")

            rng = numpy.random.RandomState(seed)
            in_state = prepared_calls[representation]['generate_input_state'](make_seed(rng))
            out_state = prepared_calls[representation]['apply_matrix'](in_state, make_seed(rng))

            for label, measurement in measurements.items():
                if 'in' in measurement.stages and representation in measurement.representations:
                    result = prepared_calls[representation]['prepared_meters'][label](in_state)
                    result_set.add_result(result, label, 'in', representation)

                if 'out' in measurement.stages and representation in measurement.representations:
                    result = prepared_calls[representation]['prepared_meters'][label](out_state)
                    result_set.add_result(result, label, 'out', representation)

        out_queue.put((worker_id, seed, result_set))


def simulate_mp(dirname, system, ensembles, samples_per_ensemble, measurements, seed, gpu_name_filters=[]):

    fname = f"{dirname}/result_sets modes={system.modes} seed={seed}.pickle"

    if Path(fname).exists():
        with open(fname, 'rb') as f:
            result_sets = pickle.load(f)
    else:
        result_sets = {}

    ensembles = ensembles - len(result_sets)
    print(fname, "- remaining ensembles:", ensembles)
    if ensembles <= 0:
        return ResultSet.merge(list(result_sets.values())[:ensembles])

    api = ocl_api()
    gpu_ids = [
        device_num for device_num, device in enumerate(api.get_platforms()[0].get_devices())
        if len(gpu_name_filters) == 0 or any(name in device.name for name in gpu_name_filters)]

    # Just in case, to make the code below airtight.
    gpu_ids = gpu_ids[:ensembles]

    result_queue = Queue()
    command_queues = [Queue() for worker_id in range(len(gpu_ids))]
    processes = [
        Thread(target=worker_gpu, args=(worker_id, result_queue, command_queue, system, samples_per_ensemble, measurements, gpu_id))
        for worker_id, (command_queue, gpu_id) in enumerate(zip(command_queues, gpu_ids))]

    for process in processes:
        process.start()

    rng = numpy.random.RandomState(seed)

    # Initial tasks
    for worker_id in range(len(gpu_ids)):
        seed = make_seed(rng)
        while seed in result_sets:
            seed = make_seed(rng)
        command_queues[worker_id].put(seed)

    ensembles_started = len(gpu_ids)
    ensembles_finished = 0
    while True:
        worker_id, worker_seed, worker_result_set = result_queue.get()
        result_sets[worker_seed] = worker_result_set
        ensembles_finished += 1

        if ensembles_finished == ensembles:
            break

        if ensembles_started < ensembles:
            seed = make_seed(rng)
            while seed in result_sets:
                seed = make_seed(rng)
            command_queues[worker_id].put(seed)

    for command_queue in command_queues:
        command_queue.put('stop')

    for process in processes:
        process.join()

    with open(fname, 'wb') as f:
        pickle.dump(result_sets, f)

    return ResultSet.merge(result_sets.values())
