# Supply-Chain
Code related to studying supply chain failures using dynamical systems
Please see the tutorial notebook for examples of how to use the built in methods.


ACI Subnetwork is defined as:
$ \mathcal{N} \subseteq N, \mathcal{E} \subseteq E$ where it is
"Connected": For each node $n \in \mathcal{N}$, there exist edges $(n,i), (j,n) \in \mathcal{E}$ with $i,j \in \mathcal{N}$

And for each node $n \in \mathcal{N}$, for all edges directed to $n$, there is one color of edges. i.e.,
$$| \{c(i,n): i\in \mathcal{N} \text{ and } (i, n) \in \mathcal{E} \}| = 1$$
where $c(i,j)$ gives the color from node $i$ to $j$.

And finally if multiple edges are directed to the same destination node and on the same color, they must all be included. 

