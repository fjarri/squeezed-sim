import os
from setuptools import setup


NAME = 'squeezed_sim'
DESCRIPTION = 'Simulation code for arXiv:2102.10341.'
URL = 'https://github.com/fjarri/squeezed-sim'
EMAIL = 'bogdan@opanchuk.net'
AUTHOR = 'Bogdan Opanchuk'
REQUIRES_PYTHON = '>=3.7.0'


REQUIRED = [
    'numpy>=1.6.0',
    'scipy>=1.4',
    'mako>=1.0.0',
    'reikna>=0.7',
    'matplotlib>=3',
    'thewalrus>=0.14',
    'tqdm>=4'
    ]


EXTRAS = {
    'torontonian': [
        'thewalrus>=0.14',
    ],
    'pyopencl': [
        'pyopencl>=2019.1.1',
        ],
    'pycuda': [
        'pycuda>=2019.1.1',
    ],
}


here = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


setup(
    name="squeezed_sim",
    version="0.1.0",
    description="Simulation code for arXiv:2102.10341",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=[
        'squeezed_sim',
        ],
    package_data={
        'squeezed_sim': ['*.mako'],
        },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'Framework :: Pytest'
    ],
)
