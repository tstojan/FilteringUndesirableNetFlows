import numpy as np
import matplotlib.pyplot as plt
from networkx import *
import random
from random import choice

def plot_graph(G):
    pos = spring_layout(G)  # positions for all nodes
    draw(G, pos=pos)
    draw_networkx_labels(G, pos=pos)
    draw_networkx_edge_labels(G, pos=pos)
    plt.show()

def pick_two_random_nodes(G):
    f_node = choice(G.nodes())  # pick a random node
    possible_nodes = set(G.nodes())
    for conn_comp in strongly_connected_component_subgraphs(G):
        if f_node in conn_comp:
            possible_nodes = set(nodes(conn_comp))

    possible_nodes.difference_update([f_node])  # remove the first node and all its neighbours from the candidates
    s_node = choice(list(possible_nodes))

    return f_node, s_node

def remove_non_utilized_edges(G, good_flows, bad_flows, verbose = True):
    edges = G.edges(data=False)
    non_utilized_edges = edges

    if not verbose:
        print("Number edges bofore", len(edges))

    for _flow_ind, _flow in enumerate(good_flows):
        for i in range(1, len(_flow)):
            if (_flow[i - 1], _flow[i]) in non_utilized_edges:
                non_utilized_edges.remove((_flow[i - 1], _flow[i]))

    for _flow_ind, _flow in enumerate(bad_flows):
        for i in range(1, len(_flow)):
            if (_flow[i - 1], _flow[i]) in non_utilized_edges:
                non_utilized_edges.remove((_flow[i - 1], _flow[i]))

    for e in non_utilized_edges:
        G.remove_edge(e[0], e[1])

    if not verbose:
        print("Number edges after deleting non-utilized", len(edges))

    return G

def path_edges_indeces(G, path, dict_edge_ind):

    edge_indeces = set()
    for i in range(1, len(path)):
        edge_indeces.add(dict_edge_ind[(path[i-1],path[i])])

    return edge_indeces


def flow_intersection(G, good_flows, bad_flows):

    edges = G.edges(data=False)

    dict_edge_ind = {k: v for v, k in enumerate(edges)}

    edges_flow_incidence_matrix = np.zeros((len(edges),len(good_flows)+len(bad_flows)))

    for _flow_ind, _flow in enumerate(good_flows):
        for i in range(1,len(_flow)):
            edges_flow_incidence_matrix[dict_edge_ind[(_flow[i-1],_flow[i])]][_flow_ind] = 1

    for _flow_ind, _flow in enumerate(bad_flows):
        for i in range(1,len(_flow)):
            edges_flow_incidence_matrix[dict_edge_ind[(_flow[i-1],_flow[i])]][len(good_flows) + _flow_ind] = 1

    flow_inresections_good_good = []
    goodflow_goodflow_edge_dictionary = {}
    flow_inresections_good_bad = []
    goodflow_badflow_edge_dictionary = {}

    for i in range(0,len(good_flows)):
        flow_inresections_good_i = []
        edges_flow = np.array(edges_flow_incidence_matrix[:,i])
        edges_indeces = [ind_v for ind_v, v in enumerate(edges_flow) if v == 1.]

        for j in range(0,len(good_flows)):
            for ed in edges_indeces:
                if edges_flow_incidence_matrix[ed][j] == 1.:
                   if j is not flow_inresections_good_i:
                        flow_inresections_good_i.append(j)
                   if (i,j) in goodflow_goodflow_edge_dictionary:
                       goodflow_goodflow_edge_dictionary[(i,j)].append(ed)
                   else:
                       goodflow_goodflow_edge_dictionary[(i,j)] = [ed]

        flow_inresections_good_good.append(flow_inresections_good_i)

        flow_inresections_bad_i = []

        for j in range(len(good_flows),len(good_flows) + len(bad_flows)):
            for ed in edges_indeces:
                if edges_flow_incidence_matrix[ed][j] == 1.:
                   if (j-len(good_flows)) is not flow_inresections_bad_i:
                        flow_inresections_bad_i.append(j-len(good_flows))
                   if (i,j) in goodflow_badflow_edge_dictionary:
                       goodflow_badflow_edge_dictionary[(i,j-len(good_flows))].append(ed)
                   else:
                       goodflow_badflow_edge_dictionary[(i,j-len(good_flows))] = [ed]

        flow_inresections_good_bad.append(flow_inresections_bad_i)


    return edges_flow_incidence_matrix, flow_inresections_good_good, goodflow_goodflow_edge_dictionary, flow_inresections_good_bad, goodflow_badflow_edge_dictionary

def flow_intersection_bad_good(G, good_flows, bad_flows):

    edges = G.edges(data=False)

    dict_edge_ind = {k: v for v, k in enumerate(edges)}

    edges_flow_incidence_matrix = np.zeros((len(edges), len(good_flows) + len(bad_flows)))

    for _flow_ind, _flow in enumerate(good_flows):
        for i in range(1, len(_flow)):
            edges_flow_incidence_matrix[dict_edge_ind[(_flow[i - 1], _flow[i])]][_flow_ind] = 1

    for _flow_ind, _flow in enumerate(bad_flows):
        for i in range(1, len(_flow)):
            edges_flow_incidence_matrix[dict_edge_ind[(_flow[i - 1], _flow[i])]][len(good_flows) + _flow_ind] = 1

    flow_inresections_bad_good = []
    badflow_goodflow_edge_dictionary = {}


    #print("here: ", edges_flow_incidence_matrix.shape)
    for i in range(0,len(bad_flows)):
         flow_inresections_good_i = []
         edges_flow = np.array(edges_flow_incidence_matrix[:,i+len(good_flows)])
         edges_indeces = [i for i,v in enumerate(edges_flow) if v == 1.]

         for j in range(0,len(good_flows)):
             for ed in edges_indeces:
                 if edges_flow_incidence_matrix[ed][j] == 1.:
                    if j is not flow_inresections_good_i:
                         flow_inresections_good_i.append(j)
                    if (i,j) in badflow_goodflow_edge_dictionary:
                        badflow_goodflow_edge_dictionary[(i,j)].append(ed)
                    else:
                        badflow_goodflow_edge_dictionary[(i,j)] = [ed]

         flow_inresections_bad_good.append(flow_inresections_good_i)

    return edges_flow_incidence_matrix, flow_inresections_bad_good, badflow_goodflow_edge_dictionary

def edges_utilization_bad_flows(G, bad_flows, bad_flows_values):

    edges_utilization_dictionary = {}

    for e in G.edges(data=False):
        edges_utilization_dictionary[e] = 0

    for _flow_ind, _flow in enumerate(bad_flows):
        for i in range(1, len(_flow)):
            if (_flow[i - 1], _flow[i]) in edges_utilization_dictionary:
                edges_utilization_dictionary[(_flow[i - 1], _flow[i])] +=  bad_flows_values[_flow_ind]

    return edges_utilization_dictionary
