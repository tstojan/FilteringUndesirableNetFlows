import numpy as np
from utils.graph_processing import *
import matplotlib.pyplot as plt
from networkx import *
import random
from random import choice

def usa_net():
    G = Graph()
    G.add_edges_from([(0, 1), (0, 5), (1, 2), (1, 5), (2, 3),
                      (2, 4), (2, 6), (3, 4), (3, 6), (4, 7),
                      (5, 6), (5, 8), (5, 10), (6, 7), (6, 8),
                      (7, 9), (8, 9), (8, 10), (8, 11), (9, 12),
                      (9, 13), (10, 11), (10, 14), (10, 18), (11, 12),
                      (11, 15), (12, 13), (12, 16), (13, 17), (14, 15),
                      (14, 19), (15, 16), (15, 20), (15, 21), (16, 17),
                      (16, 21), (16, 22), (17, 23), (18, 19), (19, 20),
                      (20, 21), (21, 22), (22, 23)])

    G = DiGraph(G)

    return G

def geant():
    G = Graph()
    G.add_edges_from([(0, 6), (0, 10), (1, 3), (1, 6), (2, 3), (3, 6),
                      (4, 5), (4, 6), (5, 9), (6, 11), (6, 17),
                      (7, 10), (8, 17), (9, 13), (10, 14), (10, 30),
                      (10, 39), (11, 14), (11, 17), (12, 13), (13, 17),
                      (13, 18), (13, 19), (15, 16), (15, 22), (15, 31),
                      (16, 17), (16, 22), (17, 18), (17, 21), (17, 39),
                      (18, 20), (20, 21), (20, 23), (20, 27), (21, 28),
                      (21, 29), (22, 29), (22, 31), (23, 25), (23, 26),
                      (24, 25), (25, 33), (25, 35), (27, 32), (29, 37),
                      (29, 38), (30, 31), (33, 35), (33, 36)])

    G = DiGraph(G)

def create_input_graph(nb_nodes = 24, m = 2, nb_good_flows = 20, nb_bad_flows = 30, C = 500, V = 15, W = 5,
                       type_weights_bad_flows ='uniform', sd = 0, verbose = True, type_graph = "BA"): #26

    np.random.seed(sd)
    random.seed()

    good_flows = []
    bad_flows = []
    bad_flows_weights = []
    bad_flows_values = []

    if type_graph == "BA":
        G = barabasi_albert_graph(n=nb_nodes, m=m, seed=sd)
        G = DiGraph(G)
    elif type_graph == "USAnet":
        nb_nodes = 24
        G = usa_net()
    elif type_graph == "GEANT":
        G = geant()
    else:
        raise "%s is not implemented network type" % type_graph

    # Set capacities
    unif = np.random.uniform(0, C, G.number_of_edges())
    i = 0
    for u, v, e in G.edges(data=True):
        e['capacity'] = unif[i]
        i += 1

    #print(G.edges(data=True))

    unif_bad_values = np.random.uniform(0, V, nb_bad_flows)
    unif_bad_weights = np.random.uniform(0, W, nb_bad_flows)

    if not verbose:
        print("Select bad flows:")

    for i in range(0,nb_bad_flows):
        f, s = pick_two_random_nodes(G)
        hopcount = shortest_path(G, f, s)
        bad_flows.append(hopcount)
        bad_flows_values.append(unif_bad_values[i])
        if type_weights_bad_flows == 'uniform':
            bad_flows_weights.append(unif_bad_weights[i])
        elif type_weights_bad_flows == 'proportional_hopcount':
            bad_flows_weights.append((1.0*W) / 1 * len(hopcount))
        elif type_weights_bad_flows == 'uniform_proportional_hopcount':
            bad_flows_weights.append(1.0 / 1 * unif_bad_weights[i] * len(hopcount))
        else:
            raise "%s is not implemented for bad flow weights determination" % type_weights_bad_flows
        #bad_flows[(f,s)] = hopcount
        #bad_flows_weights[(f, s)] = len(hopcount)

    if not verbose:
        print("Bad flows:", bad_flows)
        print("Values bad flows:", bad_flows_values)
        print("Weights bad flows:", bad_flows_weights)

        print("Select good flows:")
    for i in range(0,nb_good_flows):
        f, s = pick_two_random_nodes(G)
        hopcount = shortest_path(G, f, s)
        good_flows.append(hopcount)

    if not verbose:
        print("Good flows:", good_flows)

    return G, good_flows, bad_flows, bad_flows_values, bad_flows_weights


if __name__ == "__main__":

    create_input_graph()

