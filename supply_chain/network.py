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
    def from_file(cls, filename: str):
        obj = cls()
        obj._build_network(filename)
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
        obj._initialize_parameters(num_nodes, num_edges, num_colors, method, rho)
        obj._initialize_palette(num_colors)
        obj._build_generated_graph(method, seed)
        return obj
    

    def _initialize_parameters(self, num_nodes, num_edges, num_colors, method, rho):
        self._validate_parameters(num_nodes, num_edges, num_colors, method, rho)
        self.method = method
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_colors = num_colors
        self.rho = rho
    

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
    

    def _build_generated_graph(self, method, seed):
        try:
            builder_name = self._GRAPH_BUILDERS[method]
        except KeyError:
            raise ValueError(f"Unknown graph generation method: {method}")
        
        rng = random.Random(seed)
        getattr(self, builder_name)(rng)


    @property
    def subnetworks(self):
        if self._subnetworks is None:
            self._subnetworks = self.calculate_subnetworks()
        return self._subnetworks
    

    def _build_network(self, network_data):
        import csv
        self.G = nx.MultiDiGraph()
        colors = []

        with open(network_data, 'r', newline='') as csvfile:
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

        self.num_nodes = len(self.G.nodes)
        self.num_edges = len(self.G.edges)
        self.num_colors = len(colors)
        self.palette = []

        for i in range(self.num_colors):
            h = i / self.num_colors
            r, g, b = hsv_to_rgb(h, s=0.7, v=0.9)
            self.palette.append("#{:02x}{:02x}{:02x}".format(*[int(255*x) for x in (r, g, b)]))

        self._subnetworks = None


    def create_random_graph(self, rng: random.Random):
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
                u = rng.randrange(self.num_nodes)
                # Avoid self-loops
                if u == v:
                    continue

                c = rng.randrange(self.num_colors)
                if (u, v, c) in used:
                    continue

                w = rng.uniform(0, self.rho)
                self.G.add_edge(u, v, key=c, color=c, weight=w)
                used.add((u, v, c))
                break

        remaining = self.num_edges - self.num_nodes

        # Add in remaining required edges at random
        while remaining > 0:
            u = rng.randrange(self.num_nodes)
            v = rng.randrange(self.num_nodes)
            if u == v:
                continue

            c = rng.randrange(self.num_colors)
            if (u, v, c) in used:
                continue

            w = rng.uniform(0, self.rho)
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
            seed=rng.randint(0, 10**6)
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


    def diversify(self, color):
        """Diversify a given color in the network."""
        if not (0 <= color < self.num_colors):
            raise ValueError("Given color does not exist in network.")
        
        nodes_with_color = []
        for v in self.G.nodes:
            for _, _, data in self.G.in_edges(v, data=True):
                if data["color"] == color:
                    nodes_with_color.append(v)
                    break

        # Network can't diverisfy with less than two nodes
        if len(nodes_with_color) < 2:
            return
        
        color_edges = []
        for u, v, key, data in self.G.edges(keys=True, data=True):
            if data['color'] == color:
                color_edges.append((u, v, key, data['weight']))

        for u, v, key, w in color_edges:
            new_weight = w / 2

            self.G[u][v][key]['weight'] = new_weight

            candidates = [x for x in nodes_with_color if x != v]
            v2 = random.choice(candidates)

            existing_edge_key = None

            if self.G.has_edge(u, v2):
                for k2, d2 in self.G[u][v2].items():
                    if d2.get('color') == color:
                        existing_edge_key = k2
                        break

            if existing_edge_key is not None:
                self.G[u][v2][existing_edge_key]['weight'] += new_weight
            else:
                self.G.add_edge(u, v2, key=color, color=color, weight=new_weight)


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
        # --- Step 1. Build 'star' objects: (target, color, tuple(sources)) ---
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

        # --- Step 2. DFS helper function ---
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

        # --- Step 3. Launch DFS from valid seeds ---
        for seed_idx, (v, c, sources) in enumerate(stars):
            if any(not has_star.get(w, False) for w in sources):
                continue
            assigned[v] = c
            frontier = [u for u in sources if assigned[u] == -1]
            dfs({seed_idx}, frontier)
            assigned[v] = -1

        # --- Step 4. Remove non-maximal subnetworks ---
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
