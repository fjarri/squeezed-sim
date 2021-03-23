Simulation code for the paper "Simulating complex networks in phase space: Gaussian boson sampling" by P. D. Drummond, B. Opanchuk, and M. D. Reid, [arXiv:2102.10341](https://arxiv.org/abs/2102.10341).

The CPU part is a direct port of the reference Matlab code written by P. D. Drummond (to be published).

The code models an experiment analogous to the one from "Quantum computational advantage using photons" (https://science.sciencemag.org/content/370/6523/1460).


# Installation

This package, being somewhat niche, is not distributed via PyPi. Install by downloading the repo and running `pip install -e .`.

If you want to run the Torontonian tests, install the feature explicitly as `pip install -e .[torontonian]` (this will install [The Walrus](https://github.com/XanaduAI/thewalrus), which is rather dependency-heavy).

If you want to run the GPU code, you need to have either [PyOpenCL](https://documen.tician.de/pyopencl/) or [PyCUDA](https://documen.tician.de/pycuda/) installed, via `pip install -e .[pyopencl]` or `pip install -e .[pycuda]`.

Note that you can install several features at once (e.g. `pip install -e .[torontonian,pycuda]`).
