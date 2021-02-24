import os

import numpy
import matplotlib.pyplot as plt


def plot_results(merged_result_set, tags={}, path="."):

    if not os.path.exists(path):
        os.mkdir(path)

    for (label, stage, representation), result in merged_result_set.results.items():
        if label not in tags:
            continue

        lin = 'lin' in tags[label]
        log = 'log' in tags[label]
        errors = 'errors' in tags[label]

        title = f'{representation.value}, {stage}, {label}'
        prefix = f'{label}-{stage}-{representation.value}'

        if lin:
            fig = plt.figure(figsize=(6.4*2, 4.8*2))
            sp = fig.add_subplot(1, 1, 1)

            if errors:
                sp.errorbar(result.x_values, result.values, yerr=result.errors)
            else:
                sp.plot(result.x_values, result.values)

            sp.set_xlim((result.x_values[0], result.x_values[-1]))
            sp.set_title(title)
            sp.set_xlabel(result.x_label)
            sp.set_ylabel(result.y_label)
            if result.reference is not None:
                sp.plot(result.x_values, result.reference)

            fig.tight_layout()
            fig.savefig(os.path.join(path, f"{prefix}-lin.pdf"))
            plt.close(fig)

        if log:
            fig = plt.figure(figsize=(6.4*2, 4.8*2))
            sp = fig.add_subplot(1, 1, 1)
            sp.set_yscale('log')

            if errors:
                sp.errorbar(result.x_values, result.values, yerr=result.errors)
            else:
                sp.plot(result.x_values, result.values)

            sp.set_xlim((result.x_values[0], result.x_values[-1]))
            sp.set_title(title)
            sp.set_xlabel(result.x_label)
            sp.set_ylabel(result.y_label)
            if result.reference is not None:
                sp.plot(result.x_values, result.reference)

            fig.tight_layout()
            fig.savefig(os.path.join(path, f"{prefix}-log.pdf"))
            plt.close(fig)
