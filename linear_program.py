from cvxopt import matrix, solvers
from cvxopt.modeling import op, variable
import numpy as np
import utils.graph_processing

def check_feasibility(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights):

    edges_utilization_dictionary = utils.graph_processing.edges_utilization_bad_flows(G, bad_flows, bad_flows_values)

    for e in G.edges(data = True):
        if edges_utilization_dictionary[(e[0],e[1])] > e[2]['capacity']:
            return False


    val, _, _, _ = solve_linear_program(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights)

    return val>0

def solve_linear_program(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights, S=[], verbose = False):
    # it solves the linear program
    # max c*x, subject to Ax<=B, X>=0

    edges = G.edges(data=False)
    edges_content = G.edges(data=True)

    # variables
    x = variable(size=len(good_flows))

    #input matrices
    A = np.zeros((len(edges),len(good_flows)))
    b = np.zeros((len(edges),1))
    c = matrix((-1)*np.ones((1,len(good_flows))))

    dict_edge_ind = {k: v for v, k in enumerate(edges)}

    # populate matrix A
    for _flow_ind, _flow in enumerate(good_flows):
        if verbose:
            print(_flow_ind)
            print(_flow)
        for i in range(1,len(_flow)):
            A[dict_edge_ind[(_flow[i-1],_flow[i])]][_flow_ind] = 1

    A = matrix(A)

    # populate matrix b, by reducing the values of the bad flows not in S
    for e in edges_content:
        b[dict_edge_ind[(e[0],e[1])]] = e[2]['capacity']

    for _flow_ind, _flow in enumerate(bad_flows):
        if _flow_ind not in S:
            for i in range(1,len(_flow)):
                b[dict_edge_ind[(_flow[i-1],_flow[i])]] -= bad_flows_values[_flow_ind]

    #print("Matrix:", b)
    b = matrix(b)

    solvers.options['show_progress'] = verbose
    solvers.options
    lp = op(c*x, [A*x<=b, x>=0])#1e-08
    lp.solve()

    if lp.status == 'optimal':
        opt_value = (-1) * lp.objective.value()

        if verbose:
            print("Variables:", x.value)
            print("Optimal flow value:", opt_value[0])
    else:
        opt_value = [-1]

    return opt_value[0], x.value, A, b


if __name__ == "__main__":
    G, good_flows, bad_flows, bad_flows_values, bad_flows_weights = utils.graph_processing.create_input_graph()
    solve_linear_program(G, good_flows, bad_flows, bad_flows_values, bad_flows_weights)

