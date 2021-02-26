import numpy

from reikna.cluda import ocl_api

from sq.meters_cpu import *
from sq.system import System, Representation, State
from sq import generate_gpu, generate_cpu


def make_thread():
    api = ocl_api()
    device = api.get_platforms()[0].get_devices()[2]
    thread = api.Thread(device)
    return thread


def test_meter(meter, system, samples):

    representation = Representation.POSITIVE_P

    alpha = numpy.ones((samples, system.modes), numpy.complex128)
    beta = numpy.ones((samples, system.modes), numpy.complex128)

    state = State(system, representation, alpha, beta)
    result_cpu = meter(state)

    thread = make_thread()

    alpha_dev = thread.to_device(alpha)
    beta_dev = thread.to_device(beta)
    state_dev = State(system, representation, alpha_dev, beta_dev)

    meter_gpu = meter.prepare_gpu(thread, system, representation, samples)
    result_gpu = meter_gpu(state_dev)

    if not numpy.allclose(result_cpu.values, result_gpu.values):
        print(f"{meter} failed")
        print(result_cpu.values)
        print(result_gpu.values)
    else:
        print(f"{meter} passed")


def test_meters():

    modes = 50
    samples = 10000
    system = System(unitary=numpy.eye(modes))

    meters = [
        Population(),
        Moments(5),
        ClickProbability(),
        Clicks(5),
        ZeroClicks(5),
    ]

    for meter in meters:
        test_meter(meter, system, samples)

    # More tests for CompoundClickProbability to check the adaptive GPU algorithm
    for modes in (10, 20, 50, 64, 100, 128, 200, 500, 1000):
        test_meter(CompoundClickProbability(modes), System(unitary=numpy.eye(modes)), 10)


if __name__ == '__main__':
    test_meters()
