import utils.create_ba_input_graph as create_ba_input_graph
import utils.graph_processing as graph_processing
import linear_program
import math
import numpy as np
import time as t

def check_zero_weights_flows(bad_flows_weights, S):

    zero_weights_flows = set()

    for i, v in enumerate(bad_flows_weights):
        if (i not in S) and v == 0.:
            zero_weights_flows.add(i)

    return zero_weights_flows

def local_ratio_algo(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, verbose = False):

    edges_content = G.edges(data=True)
    edges = G.edges(data=False)

    bad_flows_weights_original = list(bad_flows_weights)
    edges_flow_incidence_matrix, flow_inresections_good_good, goodflow_goodflow_edge_dictionary, flow_inresections_good_bad, goodflow_badflow_edge_dictionary \
        = create_ba_input_graph.flow_intersection(G, good_flows, bad_flows)

    t_start = t.time()

    max_value, _, _, _ = linear_program.solve_linear_program(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, set(range(0,len(bad_flows))))


    S = set()

    opt_value, x, A, b_mat = linear_program.solve_linear_program(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, S)
    dict_edge_ind = {k: v for v, k in enumerate(edges)}

    if verbose:
        print("Starting value (no-flow filtered): ", opt_value)

    A = np.array(A)
    b_mat = np.array(b_mat)

    if x is None:
        return -1, _, _, _, _

    for ind_gflow, g_flow in enumerate(good_flows):

        zero_weights_flows = check_zero_weights_flows(bad_flows_weights, S)

        if len(zero_weights_flows)>0:
            S.update(zero_weights_flows)
            opt_value, x, _, b_mat = linear_program.solve_linear_program(G, good_flows, bad_flows, bad_flows_values,
                                                                         bad_flows_weights, S)

            x = np.array(x)
            b_mat = np.array(b_mat)
            if verbose:
                print("Opt value: ", opt_value)
        else:
            flow_inter_can_be_increased = True

            for ind_bflow in flow_inresections_good_bad[ind_gflow]:
                if flow_inter_can_be_increased:
                    for ind_ed in goodflow_badflow_edge_dictionary[(ind_gflow,ind_bflow)]:
                        if abs(np.sum(np.array(A[ind_ed,:,None])*x) - edges_content[ind_ed][2]['capacity'])<0.001:
                            flow_inter_can_be_increased = False

                            break
                else:
                    break

                if flow_inter_can_be_increased:
                    G_g_prime = set(flow_inresections_good_good[ind_gflow])

                    G_g = set()
                    for ind_ggflow in G_g_prime:
                        is_saturated = False
                        for i in range(1,len(good_flows[ind_ggflow])):
                            edge = (good_flows[ind_ggflow][i-1],good_flows[ind_ggflow][i])
                            if abs(np.sum(np.array(A[dict_edge_ind[edge],:,None])*x) - edges_content[dict_edge_ind[edge]][2]['capacity'])<0.001:
                                is_saturated = True
                                break

                        if not is_saturated:
                            G_g.add(ind_ggflow)

                    F_g = set()

                    for ind_ggflow in G_g:
                        for i in range(1,len(good_flows[ind_ggflow])):
                            edge = (good_flows[ind_ggflow][i-1],good_flows[ind_ggflow][i])

                            if abs(np.sum(np.array(A[dict_edge_ind[edge],:,None])*x) - b_mat[dict_edge_ind[edge]][0])<0.001:
                                F_g.add(dict_edge_ind[edge])

                    B_f_g = set()

                    for ind_bf, b in enumerate(bad_flows):
                        edge_indeces = create_ba_input_graph.path_edges_indeces(G, b, dict_edge_ind)
                        if bool(edge_indeces.intersection(F_g)):
                            B_f_g.add(ind_bf)

                    w_min = math.inf

                    for ind_b in B_f_g.difference(S):

                        w_min = min(w_min, bad_flows_weights[ind_b])


                    for ind_b in B_f_g.difference(S):
                        bad_flows_weights[ind_b] -= w_min

                else:
                    break  # try another flow. This one with filtering all his bad won't be increased.

    if verbose:
        print("Flows filtered local-ratio: ", S)
        print("Total flows filtered local-ratio: ", len(S))

    edges_flow_incidence_matrix, flow_inresections_bad_good, badflow_goodflow_edge_dictionary = \
        graph_processing.flow_intersection_bad_good(G, good_flows, bad_flows)

    nb = []

    for ind_b in range(0, len(bad_flows)):
        if bad_flows_weights[ind_b]>0:
            nb.append(len(flow_inresections_bad_good[ind_b]))

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
        elif ind not in S:
            S.add(ind)

    if verbose:
        print("Remaining bad flow original: ", bad_flow_original_index)

    sort_order = np.argsort((-1)*np.array(nb)*np.array(bad_flows_values_updated)/np.array(bad_flows_weights_updated))

    # sort the bad flows according to value(*)/weight(*)
    bad_flow_original_index = list(np.array(bad_flow_original_index)[sort_order])

    if verbose:
        print("Remaining bad flow sorted: ", bad_flow_original_index)

    curr_value = opt_value
    #print("Sort order", len(sort_order))
    i = 0

    while abs(curr_value-max_value)>0.1:

        if i>(len(sort_order)-1):
            break

        if verbose:
            print("Current value: ", curr_value, bad_flow_original_index[i], len(S))


        S.add(bad_flow_original_index[i])
        curr_value, x, _, _ = linear_program.solve_linear_program(G, good_flows, bad_flows, bad_flows_values,
                                                                  bad_flows_weights, S)
        i += 1


    sum_weights = sum([bad_flows_weights_original[i] for i in S])

    t_total = t.time() - t_start

    if verbose:
        print("Final value:", curr_value)
        print("Max value:", max_value)
        print("Filtered flows:", S)
        print("Size filtered flows:", len(S))
        print("Total bad flows:", len(bad_flows))
        print("Total sum of bad filtered flows:", sum_weights)
        print("Total time: ", t_total," sec")

    return curr_value, x, S, sum_weights, t_total

if __name__ == "__main__":
    G, good_flows, bad_flows, bad_flows_values, bad_flows_weights = create_ba_input_graph.create_input_graph()
    G = graph_processing.remove_non_utilized_edges(G, good_flows, bad_flows)

    if (linear_program.check_feasibility(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights)):
        local_ratio_algo(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, verbose=True)
    else:
        print("The capacities of some edges are exceeded with the given bad flows.")






