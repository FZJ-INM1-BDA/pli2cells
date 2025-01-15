import matplotlib.pyplot as plt
import numpy as np


def plot_overlap(
        title,
        eval_frame,
        plot_keys=['Motorcortex', 'Hippocampus', 'Cortex'],
        metric=None,
        file=None,
):

    mean = np.zeros(len(eval_frame))

    for k in plot_keys:
        if metric is None:
            m = eval_frame[k]
        else:
            m = [eval_frame[k][i][metric.lower()] for i in eval_frame[k].keys()]
        plt.plot(eval_frame['offset'], m, label=k)
        mean += m

    mean /= len(plot_keys)

    plt.plot(eval_frame['offset'], mean, label='Mean', linestyle='--', color='black')

    plt.legend()
    plt.title(title)
    plt.xlabel("Max. translation correction [px]")
    # plt.ylabel(metric)
    plt.grid()
    if file is not None:
        plt.savefig(file)
    plt.tight_layout()
    plt.show()


def plot_style(
        title,
        eval_frame,
        plot_keys=['Motorcortex', 'Hippocampus', 'Cortex'],
        metric=None,
        file=None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    mean = np.zeros(len(eval_frame))

    lambdas = eval_frame['lambda']

    for k in plot_keys:
        if metric is None:
            m = eval_frame[k]
        else:
            m = [eval_frame[k][i][metric] for i in eval_frame[k].keys()]
        plt.plot(lambdas, m, label=k)
        mean += m

    mean /= len(plot_keys)

    plt.plot(lambdas, mean, label='Mean', linestyle='--', color='darkgray')

    plt.legend()
    plt.title(title)
    plt.xlabel("lambda")
    plt.ylabel("pearson correlation")
    plt.grid()
    if file is not None:
        plt.savefig(file)
    plt.tight_layout()
    plt.show()
