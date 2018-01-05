import utils.graph_processing as graph_processing
import utils.create_ba_input_graph as create_ba_input_graph
import linear_program
import numpy as np
import time as t

def check_zero_weights_flows(bad_flows_weights, S):

    zero_weights_flows = set()

    for i, v in enumerate(bad_flows_weights):
        if (i not in S) and v == 0.:
            zero_weights_flows.add(i)

    return zero_weights_flows

def heuristic_algo(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, verbose = False):

    t_start = t.time()

    max_value, _, _, _ = linear_program.solve_linear_program(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, set(range(0,len(bad_flows))))

    bad_flows_updated = []
    bad_flows_values_updated = []
    bad_flows_weights_updated = []
    bad_flow_original_index = []

    for ind, val in enumerate(bad_flows_weights):
        if val>0:
            bad_flow_original_index.append(ind)
            bad_flows_updated.append(bad_flows[ind])
            bad_flows_values_updated.append(bad_flows_values[ind])
            bad_flows_weights_updated.append(bad_flows_weights[ind])

    if verbose:
        print("Remaining bad flow original: ", bad_flow_original_index)

    sort_order = np.argsort((-1)*np.array(bad_flows_values_updated)/np.array(bad_flows_weights_updated))

    #print("Sort order 1: ", sort_order)

    # sort the bad flows according to value(*)/weight(*)
    bad_flow_original_index = list(np.array(bad_flow_original_index)[sort_order])

    if verbose:
        print("Remaining bad flow sorted: ", bad_flow_original_index)

    S = set()
    curr_value = 0
    i = 0

    while abs(curr_value-max_value)>0.1:#True:

        if i>(len(sort_order)-1):
            break

        if verbose:
            print("Current value: ", curr_value, bad_flow_original_index[i], len(S))


        S.add(bad_flow_original_index[i])
        curr_value, x, _, _ = linear_program.solve_linear_program(G, good_flows, bad_flows, bad_flows_values,
                                                                  bad_flows_weights, S)
        i += 1

    sum_weights = sum([bad_flows_weights[i] for i in S])

    t_total = t.time() - t_start

    if verbose:
        print("Final value:", curr_value)
        print("Max value:", max_value)
        print("Filtered flows:", S)
        print("Size filtered flows:", len(S))
        print("Total bad flows:", len(bad_flows))
        print("Total sum of bad filtered flows:", sum_weights)
        print("Total time: ", t_total, " sec")

    return curr_value, x, S, sum_weights, t_total

if __name__ == "__main__":
    G, good_flows, bad_flows, bad_flows_values, bad_flows_weights = create_ba_input_graph.create_input_graph()
    G = graph_processing.remove_non_utilized_edges(G, good_flows, bad_flows)

    if (linear_program.check_feasibility(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights)):
        heuristic_algo(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, verbose=True)
    else:
        print("The capacities of some edges are exceeded with the given bad flows.")






