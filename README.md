# Filtering Undesirable Network Flows

This is the code, written in Python, for the paper:

**Approximation Algorithms for Filtering Undesirable Network Flows** 

by Gleb Polevoy, Stojan Trajanovski, Paola Grosso and Cees de Laat



The code contains implementations of approximation algorithms and heuristics for filtering undesirable network flows. It also contains evaluation and scripts used for the plots. For graph representations, [NetworkX](https://networkx.github.io/) python library has been used.


**Abstract.** 
We study the problem of fully mitigating the effects of denial of service by filtering the minimum necessary set of the undesirable flows. First, we model this problem and then we concentrate on a subproblem where every good flow has a bottleneck. We prove that unless P = NP, this subproblem is inapproximable within factor $2^{\log^{1-\frac{1}{\log \log^c (n)}} (n)}$ for $n = \mid E \mid + \mid GF \mid $ and any $c<0.5$. We then provide a $q(k+ 1)$ - factor polynomial approximation, where k bounds the number of the desirable flows that a desirable flow intersects, and q bounds the number of the undesirable flows that can intersect a desirable one at a given edge. Our algorithm uses the local ratio technique. We show this algorithm has an online version as well. For the general problem, we augment the local ratio to solve it and run simulations to compare the heuristic augmentations of the local ratio algorithm with the heuristics alone. We observe that the augmented local ratio performs better than respective heuristic by itself, while taking comparable time.

If you use the code, please cite as:
```
@inproceedings{FilterNetFlows_17,
title = {Filtering Undesirable Flows in Networks},
author = {Polevoy, Gleb and Trajanovski, Stojan and Grosso, Paola and {de Laat}, Cees},
booktitle = {In Proc. of COCOA (The 11th Annual Intl. Conference on Combinatorial Optimization and Applications)},
volume = {10627},
pages={3-17},
month = {December},
year = {2017},
location = {Shanghai, China},
publisher = {Springer},
url = {http://dx.doi.org/10.1007/978-3-319-71150-8_1},
}
```
