import csv
import os
import pathlib

import numpy
from scipy.io import loadmat


EXPERIMENTAL_DATA = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / 'experimental_data'


def transmission_matrix():

    re = []
    with open(EXPERIMENTAL_DATA / 'matrix_re.csv') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        for row in r:
            re.append([float(x) for x in row])
    re = numpy.array(re)

    im = []
    with open(EXPERIMENTAL_DATA / 'matrix_im.csv') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        for row in r:
            im.append([float(x) for x in row])
    im = numpy.array(im)

    U = re + 1j * im

    if U.shape[0] < U.shape[1]:
        U = numpy.concatenate([U, numpy.zeros((U.shape[1] - U.shape[0], U.shape[1]))], axis=0)

    return U.transpose()


def squeezing_coefficients():

    res = []
    with open(EXPERIMENTAL_DATA / 'squeezing_parametersq.csv') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        for row in r:
            for x in row:
                res.append(float(x))

    res = numpy.array(res)

    return res


def add_reference_experiment(system, merged_result_set):
    res = loadmat(str(EXPERIMENTAL_DATA / 'exp_cp.mat'))

    for key, result in merged_result_set.results.items():

        label, stage, representation = key

        if label == 'click_probability' and stage == 'out':
            merged_result_set.results[key] = result.with_reference(res['Cp'][3][0][:,0])

        if label == 'clicks' and stage == 'out':
            merged_result_set.results[key] = result.with_reference(res['Cp'][4][0][:,0])

        if label == 'compound_click_probability' and stage == 'out':
            merged_result_set.results[key] = result.with_reference(res['Cp'][6][0][:,0])
