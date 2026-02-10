import random
from collections import defaultdict
from colorsys import hsv_to_rgb
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


class SupplyChainNetwork:
    """Graph class to model supply chain networks."""

    _GRAPH_BUILDERS = {
        "random": "create_random_graph",
        "scale_free": "create_scale_free_graph",
    }

    def __init__(self):
        self._subnetworks = None


    @classmethod
    def from_file(
        cls, 
        filename: str,
        seed: int | None = None,
    ):
        obj = cls()
        num_colors = obj._build_network(filename)
        num_nodes = obj.G.number_of_nodes()
        num_edges = obj.G.number_of_edges()
        method = "file"
        rho = 1  # Max edge weight is 1 from file
        obj._initialize_parameters(num_nodes, num_edges, num_colors, method, rho, seed)
        obj._initialize_palette(num_colors)

        return obj
    

    @classmethod
    def generate(
        cls,
        *,
        num_nodes: int,
        num_edges: int,
        num_colors: int,
        method: str = "random",
        seed: int | None = None,
        rho: int = 2,
    ):
        obj = cls()
        obj._initialize_parameters(num_nodes, num_edges, num_colors, method, rho, seed)
        obj._initialize_palette(num_colors)
        obj._build_generated_graph(method)
        return obj
    

    def _initialize_parameters(self, num_nodes, num_edges, num_colors, method, rho, seed):
        self._validate_parameters(num_nodes, num_edges, num_colors, method, rho)
        self.method = method
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_colors = num_colors
        self.rho = rho
        self.rng = random.Random(seed)


    def _validate_parameters(self, num_nodes, num_edges, num_colors, method, rho):
        if num_nodes < 1:
            raise ValueError("Number of nodes must be positive.")
        
        if method == "random":
            max_edges = num_colors * num_nodes * (num_nodes - 1)
            if not (num_nodes <= num_edges <= max_edges):
                raise ValueError(
                    "Strongly connected graph is not possible with given parameters."
                )
            
        if rho <= 0:
            raise ValueError("Maximum edge weight must be positive.")
            

    def _initialize_palette(self, num_colors: int):
        self.palette = [
            self._color_from_index(i, num_colors)
            for i in range(num_colors)
        ]

    
    @staticmethod
    def _color_from_index(i: int, n: int) -> str:
        h = i / n 
        r, g, b = hsv_to_rgb(h, s=0.7, v=0.9)
        return "#{:02x}{:02x}{:02x}".format(
            int(255 * r), int(255 * g), int(255 * b)
        )
    

    def _build_generated_graph(self, method):
        try:
            builder_name = self._GRAPH_BUILDERS[method]
        except KeyError:
            raise ValueError(f"Unknown graph generation method: {method}")
        
        getattr(self, builder_name)()


    @property
    def subnetworks(self):
        if self._subnetworks is None:
            self._subnetworks = self.calculate_subnetworks()
        return self._subnetworks
    

    def _build_network(self, filename: str) -> int:
        import csv
        self.G = nx.MultiDiGraph()
        colors = []

        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
        
            for row in reader:
                u, v, business = row
                start = business.find(":") + 1
                end = business.find("}", start)
                business_type = business[start:end].strip()
                if business_type not in colors:
                    colors.append(business_type)

                c = colors.index(business_type)

                # Due to insufficient data, set all weights for edges equal to 1
                self.G.add_edge(u, v, key=c, color=c, weight=1)

        return len(colors)


    def create_random_graph(self):
        """Build a random weakly-connected supply-chain network.
        Each node has in-degree >= 1 (no zero columns in the adjacency matrix).

        * Graph type            : nx.MultiDiGraph (parallel arcs allowed)
        * Edge attributes       : color - int (category index into self.palette)
                                  weight - float (uniform[0, rho))
        * Constraints           : at most one edge of a given color between any (u, v)
        """
        self.G = nx.MultiDiGraph()
        used = set()

        # Ensure each node has in-degree >= 1
        for v in range(self.num_nodes):
            while True:
                u = self.rng.randrange(self.num_nodes)
                # Avoid self-loops
                if u == v:
                    continue

                c = self.rng.randrange(self.num_colors)
                if (u, v, c) in used:
                    continue

                w = self.rng.uniform(0, self.rho)
                self.G.add_edge(u, v, key=c, color=c, weight=w)
                used.add((u, v, c))
                break

        remaining = self.num_edges - self.num_nodes

        # Add in remaining required edges at random
        while remaining > 0:
            u = self.rng.randrange(self.num_nodes)
            v = self.rng.randrange(self.num_nodes)
            if u == v:
                continue

            c = self.rng.randrange(self.num_colors)
            if (u, v, c) in used:
                continue

            w = self.rng.uniform(0, self.rho)
            self.G.add_edge(u, v, key=c, color=c, weight=w)
            used.add((u, v, c))
            remaining -= 1


    def create_scale_free_graph(self, rng: random.Random):
        """Build a weakly-connected, scale-free-like supply-chain network
        with preferential attachment in both directions.

        * Graph type      : nx.MultiDiGraph (parallel arcs allowed)
        * Edge attributes : color - int (category index into self.palette)
                            weight - float (uniform[0, rho))
        * Generation      : Uses nx.scale_free_graph (directed, power-law in/out degrees)
        """
        base_G = nx.scale_free_graph(
            self.num_nodes,
            seed=self.rng.randint(0, 10**6)
        )

        self.G = nx.MultiDiGraph()

        for u, v in base_G.edges():
            c = rng.randint(0, self.num_colors - 1)
            w = rng.uniform(0, self.rho)
            self.G.add_edge(u, v, key=c, color=c, weight=w)

        if not nx.is_weakly_connected(self.G):
            order = list(self.G.nodes)
            rng.shuffle(order)
            for i in range(len(order)):
                u = order[i]
                v = order[(i + 1) % len(order)]
                c = rng.randint(0, self.num_colors - 1)
                w = rng.uniform(0, self.rho)
                self.G.add_edge(u, v, key=c, color=c, weight=w)

        self.num_edges = self.G.number_of_edges()


    def visualize_supply_chain(self, pos=None, weight_range=(1.2, 4.0)):
        if pos is None:
            pos = nx.spring_layout(self.G, seed=0)

        cmap = mcolors.ListedColormap(self.palette)
        vmin, vmax = 0, self.num_colors - 1
        fig, ax = plt.subplots(figsize=(7, 7))

        nx.draw_networkx_nodes(
            self.G, pos, node_size=350, node_color="white",
            edgecolors="black", linewidths=1.2, ax=ax
        )
        nx.draw_networkx_labels(
            self.G, pos, labels={v: str(v) for v in self.G}, font_size=10, ax=ax
        )

        w_raw = [d["weight"] for _, _, _, d in self.G.edges(keys=True, data=True)]
        wr_min, wr_max = (min(w_raw), max(w_raw)) if w_raw else (1, 1)
        wp_min, wp_max = weight_range 
        scale = ((wp_max - wp_min) / (wr_max - wr_min)) if wr_max != wr_min else 0

        def width(w):
            return wp_min + (w - wr_min) * scale if scale else (wp_min + wp_max) / 2

        slot = defaultdict(int)

        for u, nbrdict in self.G.adj.items():  # sources
            for v, keyedgedict in nbrdict.items():  # targets
                pair = frozenset((u, v))
                for _, data in keyedgedict.items():  # parallel edges
                    k = slot[pair]
                    slot[pair] += 1

                    sign = 1 if k & 1 else -1
                    magnitude = (k + 1) // 2
                    rad = sign * 0.15 * magnitude

                    if u > v:
                        rad = -rad

                    nx.draw_networkx_edges(
                        self.G, pos, 
                        edgelist=[(u, v)], 
                        edge_color=self.palette[data["color"]],
                        edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax,
                        width=width(data["weight"]),
                        arrows=True, arrowstyle="-|>", arrowsize=15,
                        connectionstyle=f"arc3,rad={rad}", ax=ax
                    )

        present_cols = {d["color"] for _, _, _, d in self.G.edges(keys=True, data=True)}
        handles = [mpatches.Patch(color=self.palette[i], label=f"Color {i}")
                   for i in sorted(present_cols)]
        if handles:
            ax.legend(handles=handles, title="Edge Colors",
                      loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        ax.axis("off")
        fig.tight_layout()
        plt.show()


    def diversify(self, p: float | int, lam: float = 0.5):
        """Diversify a percentage of edges in the network by factor lambda.
        
        :param p: Percentage of edges to diversify 
        :type p: float | int
        :param lam: Diversification factor 
        If original edge has weight a, the original edge has weight lam*a, new edge has weight (1-lam)*a.
        :type lam: float
        """
        # Validate parameters
        if not (0 < p <= 1):
            raise ValueError("Percentage of nodes must be 0 < p <= 1")
        
        if not (0 < lam < 1):
            raise ValueError("Diversification factor must be 0 < lam < 1")
        
        nodes_with_color = defaultdict(set)
        edges = list(self.G.edges(keys=True, data=True))

        for _, v, _, data in edges:
            color = data["color"]
            nodes_with_color[color].add(v)
        
        num_edges = round(p * self.G.number_of_edges())
        edges_to_diversify = self.rng.sample(edges, num_edges)

        diversified = set()

        for u, v, key, data in edges_to_diversify:
            if (u, v, key) in diversified:
                continue
            diversified.add((u, v, key))

            color = data["color"]
            weight = data["weight"]
            destinations = nodes_with_color[color]

            # Sample new destination that's different than current one
            candidates = destinations - {v}
            
            if u in destinations:
                candidates = candidates | {u}
                
            if not candidates:
                continue

            v2 = self.rng.choice(tuple(candidates))
            new_weight = (1 - lam) * weight

            for _, d2 in self.G[u][v2].items() if self.G.has_edge(u, v2) else []:
                # Merge weight if edge already exists
                if d2["color"] == color:
                    d2["weight"] += new_weight
                    break

            else:
                self.G.add_edge(u, v2, key=color, color=color, weight=new_weight)

            self.G[u][v][key]['weight'] = lam * weight

        self.num_edges = self.G.number_of_edges()
        self._subnetworks = None


    def _materialize_subgraph(self, stars, star_defs):
        H = nx.MultiDiGraph()
        for idx in stars:
            target, color, sources = star_defs[idx]
            for source in sources:
                data = self.G[source][target][color]
                H.add_edge(source, target, key=color, **data)
                
        return H


    def calculate_subnetworks(self):
        """DFS algorithm for calculating ACI Subnetworks.
        Returns only maximal subnetworks (no duplicates or strict subsets).
        """
        stars = []
        star_of_v = defaultdict(list)
        for v in self.G.nodes:
            by_color = defaultdict(list)
            for u, _, data in self.G.in_edges(v, data=True):
                by_color[data["color"]].append(u)
            for c, sources in by_color.items():
                idx = len(stars)
                stars.append((v, c, tuple(sources)))
                star_of_v[v].append(idx)

        has_star = {v: bool(star_of_v[v]) for v in self.G.nodes}
        results = []
        seen = set()  # prevent exact-duplicate closures

        assigned = [-1] * self.num_nodes

        def dfs(chosen, frontier):
            """Recursively explore combinations of stars."""
            # Pop already-assigned nodes from the frontier
            while frontier and assigned[frontier[-1]] != -1:
                frontier.pop()

            # If no nodes left to expand, record closure
            if not frontier:
                frozen = frozenset(chosen)
                if frozen not in seen:
                    seen.add(frozen)
                    results.append(self._materialize_subgraph(sorted(chosen), stars))
                return

            # Otherwise, expand next node in frontier
            w = frontier.pop()
            for idx in star_of_v.get(w, []):
                v2, c2, sources2 = stars[idx]

                # Skip if star depends on nodes without stars
                if any(not has_star.get(u, False) for u in sources2):
                    continue

                # Skip color conflicts
                if assigned[w] != -1 and assigned[w] != c2:
                    continue

                old_color = assigned[w]
                assigned[w] = c2

                new_frontier = frontier + [u for u in sources2 if assigned[u] == -1]
                dfs(chosen | {idx}, new_frontier)

                assigned[w] = old_color  # rollback

        for seed_idx, (v, c, sources) in enumerate(stars):
            if any(not has_star.get(w, False) for w in sources):
                continue
            assigned[v] = c
            frontier = [u for u in sources if assigned[u] == -1]
            dfs({seed_idx}, frontier)
            assigned[v] = -1

        maximal_results = []
        node_sets = [set(H.nodes) for H in results]
        for i, s1 in enumerate(node_sets):
            if not any(s1 < s2 for j, s2 in enumerate(node_sets) if i != j):
                maximal_results.append(results[i])

        return maximal_results
    

    def visualize_network(self, G: nx.MultiDiGraph, title=None, pos=None, weight_range=(1.2, 4.0)):
        if pos is None:
            pos = nx.spring_layout(G, seed=0)

        cmap = mcolors.ListedColormap(self.palette)
        vmin, vmax = 0, self.num_colors - 1
        fig, ax = plt.subplots(figsize=(7, 7))

        nx.draw_networkx_nodes(
            G, pos, node_size=350, node_color="white",
            edgecolors="black", linewidths=1.2, ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, labels={v: str(v) for v in G}, font_size=10, ax=ax
        )

        w_raw = [d["weight"] for _, _, _, d in G.edges(keys=True, data=True)]
        wr_min, wr_max = (min(w_raw), max(w_raw)) if w_raw else (1, 1)
        wp_min, wp_max = weight_range 
        scale = ((wp_max - wp_min) / (wr_max - wr_min)) if wr_max != wr_min else 0

        def width(w):
            return wp_min + (w - wr_min) * scale if scale else (wp_min + wp_max) / 2
        
        slot = defaultdict(int)

        for u, nbrdict in G.adj.items():  # sources
            for v, keyedgedict in nbrdict.items():  # targets
                pair = frozenset((u, v))
                for _, data in keyedgedict.items():  # parallel edges
                    k = slot[pair]
                    slot[pair] += 1

                    sign = 1 if k & 1 else -1
                    mag = (k + 1) // 2
                    rad = sign * 0.15 * mag

                    if u > v:
                        rad = -rad

                    nx.draw_networkx_edges(
                        G, pos, 
                        edgelist=[(u, v)], 
                        edge_color=self.palette[data["color"]],
                        edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax,
                        width=width(data["weight"]),
                        arrows=True, arrowstyle="-|>", arrowsize=15,
                        connectionstyle=f"arc3,rad={rad}", ax=ax
                    )

        present_cols = {d["color"] for _, _, _, d in G.edges(keys=True, data=True)}
        handles = [mpatches.Patch(color=self.palette[i], label=f"Color {i}")
                   for i in sorted(present_cols)]
        if handles:
            ax.legend(handles=handles, title="Edge Colors",
                      loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        ax.axis("off")
        if title:
            plt.title(title)
        fig.tight_layout()

        radius = eigs(nx.adjacency_matrix(G), k=1, which="LM")
        plt.title(f"{np.abs(radius[0][0])}")

        plt.show()

    def visualize_subnetworks(self):
        for subnetwork in self.subnetworks:
            self.visualize_network(subnetwork)
