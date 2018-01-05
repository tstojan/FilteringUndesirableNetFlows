import numpy as np
import pandas as pd
import networkx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib as mpl
import utils.create_ba_input_graph as create_ba_input_graph
import utils.graph_processing as graph_processing
import linear_program as lin_prog
import local_ratio_h1 as lr_algo
import local_ratio_h2 as lr_algo2
import h1_algo as heur_val_weight
import h2_algo as heur_inter_val_weight
print("networkx version:", networkx.__version__) # check the version

def created_input(nb_bad, sd, type_weights_bad_flows, graph_type):

    G, good_flows, bad_flows, bad_flows_values, bad_flows_weights = create_ba_input_graph.create_input_graph(
        nb_bad_flows=nb_bad, sd=sd, type_weights_bad_flows=type_weights_bad_flows, type_graph = graph_type)
    G = graph_processing.remove_non_utilized_edges(G, good_flows, bad_flows)

    return  G, good_flows, bad_flows, bad_flows_values, bad_flows_weights

def calculate_normalized_load(G, bad_flows, bad_flows_values):

    edges_utilization_dictionary = graph_processing.edges_utilization_bad_flows(G, bad_flows, bad_flows_values)
    normalized_load = max([edges_utilization_dictionary[(e[0],e[1])]/e[2]['capacity'] for e in G.edges(data=True)])

    return normalized_load

def save_results(file_name, graph_type, type_weights, fin_nb_bad, fin_normalized_load, fin_approx_h1, fin_approx_h2, fin_h1, fin_h2):

    df = pd.DataFrame({"nb_bad": np.array(fin_nb_bad), "norm_load": np.array(fin_normalized_load), "approxh1": np.array(fin_approx_h1),
                       "approxh2": np.array(fin_approx_h2), "h1": np.array(fin_h1), "h2": np.array(fin_h2)})

    df.to_csv('results_csv/' + graph_type + "/" + type_weights + "/" + file_name + '.csv', "\t", header= True,
              columns=["nb_bad","norm_load","approxh1", "approxh2", "h1", "h2"], index=False)


def visualize_results(file_name, title_name, graph_type, type_weights, fin_normalized_load, fin_approx_h1, fin_approx_h2, fin_h1, fin_h2):

    fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', ['black', 'saddlebrown', '#ff3232', 'gold']))

    ax.plot(fin_normalized_load, fin_approx_h1, 'v-', linewidth=1.7, label='local-ratio ' + r'$h_1$')
    ax.plot(fin_normalized_load, fin_h1, 's--', linewidth=1.5, dashes=[5, 5, 5, 5], label=r'$h_1$')
    ax.plot(fin_normalized_load, fin_approx_h2, 'o-', linewidth=1.7, label='local-ratio ' + r'$h_1$')
    ax.plot(fin_normalized_load, fin_h2, 'd--', linewidth=1.5, dashes=[5, 5, 5, 5], label=r'$h_2$')

    ax.legend(loc='upper left')
    ax.set_xlabel('normalized load')
    ax.set_ylabel(title_name)
    ax.set_xticklabels(labels=[], minor=True)
    ax.set_title(title_name)
    ax.set_yscale("log")

    plt.draw()
    plt.pause(5)
    plt.savefig('plots/' + graph_type+ "/" + type_weights + "/" + file_name + '.pdf')


if __name__ == "__main__":

    graph_type = "USAnet" #'BA', 'USAnet', 'GEANT'
    number_of_instances = 100
    type_weights = 'uniform'#'uniform' 'proportional_hopcount', 'uniform_proportional_hopcount'
    sd_start= 20
    sd = sd_start
    sd_step = 4
    max_nb_bad_flows = 100
    nb_flow_step = 10
    visualize_results = False

    fin_nb_bad = []
    fin_normalized_load = []
    fin_approx_h1 = []
    fin_approx_h2 = []
    fin_h1 = []
    fin_h2 = []

    fin_approx_h1_time = []
    fin_approx_h2_time = []
    fin_h1_time = []
    fin_h2_time = []

    print("nb_bad\tnorm_load\tapproxh1\tapproxh2\th1\th2")

    lst_bad_flows = [2, 3, 4, 5, 7, 8, 10, 12, 16, 20, 30, 40, 50, 60, 65]# 75] ,#95

    for nb_bad in lst_bad_flows:

        sd = sd_start

        i = 0

        normalized_load = []

        approx_val_good = []
        approx_val_bad = []
        approx_time = []

        approx_val_good_h2 = []
        approx_val_bad_h2 = []
        approx_time_h2 = []

        h1_val_good = []
        h1_val_bad = []
        h1_time = []

        h2_val_good = []
        h2_val_bad = []
        h2_time = []

        while i<number_of_instances:
            G, good_flows, bad_flows, bad_flows_values, bad_flows_weights = created_input(nb_bad = nb_bad, sd=sd,
                                                                                          type_weights_bad_flows=type_weights,
                                                                                          graph_type = graph_type)

            if (lin_prog.check_feasibility(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights)):
                G, good_flows, bad_flows, bad_flows_values, bad_flows_weights = created_input(nb_bad=nb_bad, sd=sd,
                                                                                              type_weights_bad_flows=type_weights,
                                                                                              graph_type=graph_type)

                #print(G.edges())

                nld = calculate_normalized_load(G, bad_flows, bad_flows_values)

                l_rt_val_sum_good, l_rt_x, l_rt_S, l_rt_val_sum_filtered, l_rt_t_total = \
                    lr_algo.local_ratio_algo(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights,
                                             verbose=False)

                if l_rt_val_sum_good==-1:
                    sd += sd_step
                    continue

                G, good_flows, bad_flows, bad_flows_values, bad_flows_weights = created_input(nb_bad=nb_bad, sd=sd,
                                                                                              type_weights_bad_flows=type_weights,
                                                                                              graph_type=graph_type)

                l_rt_val_sum_good_h2, l_rt_x_h2, l_rt_S_h2, l_rt_val_sum_filtered_h2, l_rt_t_total_h2 = \
                    lr_algo2.local_ratio_algo(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, verbose=False)

                if l_rt_val_sum_good_h2 == -1:
                    sd += sd_step
                    continue

                G, good_flows, bad_flows, bad_flows_values, bad_flows_weights = created_input(nb_bad=nb_bad, sd=sd,
                                                                                              type_weights_bad_flows=type_weights,
                                                                                              graph_type=graph_type)

                h_rt_wval_val_sum_good, h_rt_wval_l_rt_x, h_rt_wval_l_rt_S, h_rt_wval_l_rt_val_sum_filtered, h_rt_wval_l_rt_t_total =  \
                    heur_val_weight.heuristic_algo(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, verbose=False)

                if h_rt_wval_val_sum_good == -1:
                    sd += sd_step
                    continue


                G, good_flows, bad_flows, bad_flows_values, bad_flows_weights = created_input(nb_bad=nb_bad, sd=sd,
                                                                                              type_weights_bad_flows=type_weights,
                                                                                              graph_type=graph_type)

                h_inter_wval_val_sum_good, h_inter_wval_l_rt_x, h_inter_wval_l_rt_S, h_inter_wval_l_inter_val_sum_filtered, h_inter_wval_l_rt_t_total = \
                    heur_inter_val_weight.heuristic_algo(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, verbose=False)


                if h_inter_wval_val_sum_good == -1:
                    sd += sd_step
                    continue

                if nld<1.0:
                    normalized_load.append(nld)
                else:
                    sd += sd_step
                    continue

                approx_val_good.append(l_rt_val_sum_good)
                approx_val_bad.append(l_rt_val_sum_filtered)
                approx_time.append(l_rt_t_total)

                approx_val_good_h2.append(l_rt_val_sum_good_h2)
                approx_val_bad_h2.append(l_rt_val_sum_filtered_h2)
                approx_time_h2.append(l_rt_t_total_h2)

                h1_val_good.append(h_rt_wval_val_sum_good)
                h1_val_bad.append(h_rt_wval_l_rt_val_sum_filtered)
                h1_time.append(h_rt_wval_l_rt_t_total)

                h2_val_good.append(h_inter_wval_val_sum_good)
                h2_val_bad.append(h_inter_wval_l_inter_val_sum_filtered)
                h2_time.append(h_inter_wval_l_rt_t_total)

                i += 1

            sd += sd_step

        print("{}\t{}\t{}\t{}\t{}\t{}".format(nb_bad, np.mean(normalized_load), np.mean(approx_val_bad),
                                              np.mean(approx_val_bad_h2), np.mean(h1_val_bad), np.mean(h2_val_bad)))

        #print("{}\t{}\t{}\t{}\t{}\t{}".format(nb_bad, np.mean(normalized_load), np.mean(approx_time),
        #                                      np.mean(approx_time_h2), np.mean(h1_time), np.mean(h1_time)))

        fin_nb_bad.append(nb_bad)
        fin_normalized_load.append(np.mean(normalized_load))
        fin_approx_h1.append(np.mean(approx_val_bad))
        fin_approx_h2.append(np.mean(approx_val_bad_h2))
        fin_h1.append(np.mean(h1_val_bad))
        fin_h2.append(np.mean(h2_val_bad))

        fin_approx_h1_time.append(np.mean(approx_time))
        fin_approx_h2_time.append(np.mean(approx_time_h2))
        fin_h1_time.append(np.mean(h1_time))
        fin_h2_time.append(np.mean(h2_time))

    save_results('sums_bad_weights', graph_type, type_weights, fin_nb_bad, fin_normalized_load, fin_approx_h1,
                      fin_approx_h2, fin_h1, fin_h2)

    if visualize_results:
        visualize_results('sums_bad_weights', 'Sum of filterd flow weights', graph_type, type_weights,
                          fin_normalized_load, fin_approx_h1, fin_approx_h2, fin_h1, fin_h2)

    save_results('running_times', graph_type, type_weights, fin_nb_bad, fin_normalized_load, fin_approx_h1_time,
                      fin_approx_h2_time, fin_h1_time, fin_h2_time)

    if visualize_results:
        visualize_results('running_times', 'Average running time (s)', graph_type, type_weights,
                          fin_normalized_load, fin_approx_h1_time, fin_approx_h2_time, fin_h1_time, fin_h2_time)







