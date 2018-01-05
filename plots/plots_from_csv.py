import numpy as np
import pandas as pd
import networkx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


def read_visualize_results(file_name, title_name, graph_type, type_weights):

    df = pd.read_csv('../results_csv/'+ graph_type + "/" + type_weights + "/" + file_name + '.csv', delimiter = '\t')

    fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', ['black', 'saddlebrown', '#ff3232', 'gold']))


    ax.plot(df['norm_load'].values, df['approxh1'].values, 'v-', linewidth=1.7, label=r'$LR+h_1$')
    ax.plot(df['norm_load'].values, df['h1'].values, 's--', linewidth=1.5, dashes=[5, 5, 5, 5], label=r'$h_1$')
    ax.plot(df['norm_load'].values, df['approxh2'].values, 'o-', linewidth=1.7, label=r'$LR+h_2$')
    ax.plot(df['norm_load'].values, df['h2'].values, 'd--', linewidth=1.5, dashes=[5, 5, 5, 5], label=r'$h_2$')

    ax.legend(loc='upper left', prop={'size': 14})
    ax.set_xlabel('normalized load', fontsize=14)
    ax.set_ylabel(title_name, fontsize=14)
    ax.set_xticklabels(labels=[], minor=True)
    ax.set_title(title_name, fontsize=16)

    ax.set_yscale("log")

    axins = fig.add_axes([0.57, 0.17, 0.3, 0.3])  # X, Y, width, height
    axins.set_prop_cycle(plt.cycler('color', ['black', '#ff3232','black', '#ff3232']))
    #axins.legend(loc='lower left', prop={'size': 14})

    ratio1 = df['approxh1'].values/df['h1'].values
    ratio2 = df['approxh2'].values/df['h2'].values

    axins.plot(df['norm_load'].values, ratio1, '-', linewidth=1.5, label=r'$\frac{LR + h1}{h_1}$')
    axins.plot(df['norm_load'].values, ratio2, '--', linewidth=1.5, dashes=[5, 5, 5, 5], label=r'$\frac{LR + h2}{h_2}$')

    if title_name.startswith("Sum"):
        axins.plot(df['norm_load'].values, np.mean(ratio1)*np.ones(len(ratio1)), '-', linewidth=0.8)
        axins.plot(df['norm_load'].values, np.mean(ratio2) * np.ones(len(ratio2)), '--', linewidth=0.8)

        axins.set_ylim(0.5, 1.0)  # apply the y-limits

    axins.legend()

    #plt.show()
    plt.draw()
    plt.pause(5)
    plt.savefig(graph_type + "/" + type_weights + "/" + file_name + '.pdf')


if __name__ == "__main__":

    graph_type = ['BA', 'USAnet', 'GEANT']
    type_weights = ['uniform', 'proportional_hopcount', 'uniform_proportional_hopcount']


    for g_type in graph_type:
        for w_type in type_weights:
            read_visualize_results('sums_bad_weights', 'Sum of filterd flow weights', g_type, w_type)
            read_visualize_results('running_times', 'Average running time (s)', g_type, w_type)


