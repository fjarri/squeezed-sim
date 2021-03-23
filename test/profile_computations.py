import time
import sys

import numpy

from reikna.cluda import get_api

from squeezed_sim import *
from squeezed_sim import generate_gpu, generate_cpu


def make_thread(api_id, device_num):
    api = get_api(api_id)
    device = api.get_platforms()[0].get_devices()[device_num]
    thread = api.Thread(device)
    return thread


def test_meter(meter, system, samples, test_cpu=False, api_id='ocl', device_num=0):

    representation = Representation.POSITIVE_P

    alpha = numpy.ones((samples, system.modes), numpy.complex128)
    beta = numpy.ones((samples, system.modes), numpy.complex128)

    if test_cpu:
        state = State(system, representation, alpha, beta)
        t1 = time.time()
        meter(state)
        time_cpu = time.time() - t1

    thread = make_thread(api_id, device_num)

    alpha_dev = thread.to_device(alpha)
    beta_dev = thread.to_device(beta)
    state_dev = State(system, representation, alpha_dev, beta_dev)

    meter_gpu = meter.prepare_gpu(thread, system, representation, samples)

    thread.synchronize()

    t1 = time.time()
    meter_gpu(state_dev)
    thread.synchronize()
    time_gpu = time.time() - t1

    print(f"Meter {type(meter)}: {time_gpu}")
    if test_cpu:
        print(f"    cpu: {time_cpu}")


def test_meters(modes, samples, test_cpu=False, api_id='ocl', device_num=0):

    system = System(unitary=numpy.eye(modes))

    meters = [
        Population(),
        Moments(5),
        ClickProbability(),
        Clicks(5),
        ZeroClicks(5),
        CompoundClickProbability(modes),
    ]

    for meter in meters:
        test_meter(meter, system, samples, test_cpu=test_cpu, api_id=api_id, device_num=device_num)


def test_generate_initial_state(modes, samples, test_cpu=False, api_id='ocl', device_num=0):

    system = System(unitary=numpy.eye(modes))
    representation = Representation.POSITIVE_P
    seed = 123

    if test_cpu:
        t1 = time.time()
        generate_cpu.generate_input_state(system, representation, samples, seed)
        time_cpu = time.time() - t1

    thread = make_thread(api_id, device_num)

    generate_input_state = generate_gpu.prepare_generate_input_state(thread, system, representation, samples)

    t1 = time.time()
    generate_input_state(seed)
    thread.synchronize()
    time_gpu = time.time() - t1

    print(f"generate_initial_state(): {time_gpu}")
    if test_cpu:
        print(f"    cpu: {time_cpu}")


def test_apply_matrix(modes, samples, test_cpu=False, api_id='ocl', device_num=0):

    system = System(unitary=numpy.ones((modes, modes)))
    representation = Representation.POSITIVE_P
    seed = 123

    alpha = numpy.ones((samples, system.modes), numpy.complex128)
    beta = numpy.ones((samples, system.modes), numpy.complex128)

    if test_cpu:
        state = State(system, representation, alpha, beta)
        t1 = time.time()
        generate_cpu.apply_matrix(state, seed)
        time_cpu = time.time() - t1

    thread = make_thread(api_id, device_num)

    alpha_dev = thread.to_device(alpha)
    beta_dev = thread.to_device(beta)
    state_dev = State(system, Representation.POSITIVE_P, alpha_dev, beta_dev)

    apply_matrix = generate_gpu.prepare_apply_matrix(thread, system, representation, samples)

    t1 = time.time()
    apply_matrix(state_dev, seed)
    thread.synchronize()
    time_gpu = time.time() - t1

    print(f"apply_matrix(): {time_gpu}")
    if test_cpu:
        print(f"    cpu: {time_cpu}")


if __name__ == '__main__':
    API_ID = 'ocl' # or 'cuda'
    DEVICE_NUM = 2

    modes = 40
    samples = 100000
    test_meters(modes, samples, test_cpu=True, api_id=API_NAME, device_num=DEVICE_NUM)
    test_generate_initial_state(modes, samples, test_cpu=True, api_id=API_NAME, device_num=DEVICE_NUM)
    test_apply_matrix(modes, samples, test_cpu=True, api_id=API_NAME, device_num=DEVICE_NUM)
