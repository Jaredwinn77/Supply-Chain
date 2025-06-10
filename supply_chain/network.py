import random
from collections import defaultdict, deque
from copy import deepcopy
from colorsys import hsv_to_rgb
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

class SupplyChainNetwork:
    """Graph class to model supply chain networks."""
    def __init__(self, num_nodes: int, num_edges: int, num_colors: int, method: str = "random", seed: int | None = None, rho: int = 2):
        """
        """
        # Check for a positive number of nodes
        if num_nodes < 1:
            raise ValueError("Number of nodes must be non-negative.")
        
        # Check to make sure weakly connected graph is possible given edge, node, and color counts
        if not (num_nodes <= num_edges <= num_colors * num_nodes * (num_nodes - 1)):
            raise ValueError("Strongly connected graph is not possible with given edge and node counts.")
        
        rng = random.Random(seed)
        self.num_nodes = num_nodes 
        self.num_edges = num_edges
        self.palette = []
        for i in range(num_colors):
            h = i / num_colors
            r, g, b = hsv_to_rgb(h, s=0.7, v=0.9)
            self.palette.append("#{:02x}{:02x}{:02x}".format(*[int(255*x) for x in (r, g, b)]))

        self.num_colors = num_colors

        match method:
            case "random":
                self.create_random_graph(rng, rho)

        self._subnetworks = None


    @property
    def subnetworks(self):
        if self._subnetworks is None:
            self._subnetworks = self.calculate_subnetworks()
        return self._subnetworks


    def create_random_graph(self, rng: random.Random, rho: int):
        """Build a random weakly-connected supply-chain network.

        * Graph type            : nx.MultiDiGraph (parallel arcs allowed)
        * Edge attributes       : color - int (category index into self.palette)
                                  weight - float (uniform[0, rho))
        * Constraints           : at most one edge of a given color between any (u, v)
        """
        # Create random directed cycle through all nodes (guarantees weak-connectivity) 
        order = list(range(self.num_nodes))
        rng.shuffle(order)

        self.G = nx.MultiDiGraph()

        for i in range(self.num_nodes):
            u = order[i]
            v = order[(i + 1) % self.num_nodes]
            c = rng.randint(0, self.num_colors - 1)
            w = rng.uniform(0, rho)

            # Ensure duplicate colors are impossible by settting it as the key
            self.G.add_edge(u, v, key=c, color=c, weight=w)

        # Add in remaining directed edges at random
        remaining = self.num_edges - self.num_nodes
        if remaining <= 0:
            return
        
        def color_exists(u: int, v: int, color: int) -> bool:
            """Helper function to determine if colored edge exists between (u, v)"""
            if not self.G.has_edge(u, v):
                return False
            for attr in self.G[u][v].values():
                if attr.get("color") == color:
                    return True
            return False
        
        # all (u, v, color) triples that are still possible
        possible = [
            (u, v, c)
            for u in self.G.nodes
            for v in self.G.nodes
            if u != v
            for c in range(self.num_colors)
            if not color_exists(u, v, c)
        ]

        chosen = rng.sample(possible, remaining)

        self.G.add_edges_from(
            (u, v, c, {"color": c, "weight": rng.uniform(0, rho)}) 
            for u, v, c in chosen
        )


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


    def _materialize_subgraph(self, stars, star_defs):
        H = nx.MultiDiGraph()
        for idx in stars:
            s = star_defs[idx]
            target = s["target"]
            color = s["color"]
            for source in s["sources"]:
                data = self.G[source][target][color]
                H.add_edge(source, target, key=color, **data)
        return H


    def calculate_subnetworks(self):
        """DFS Algorithm for calculating ACI Subnetworks"""
        # Preprocess data into 'star' objects, which is a node and all same-color edges incoming
        stars = []
        star_of_v = defaultdict(list)

        for v in self.G.nodes:
            by_color = defaultdict(list)
            for u, _, data in self.G.in_edges(v, data=True):
                by_color[data["color"]].append(u)
            for c, sources in by_color.items():
                idx = len(stars)
                stars.append({"target": v,
                              "color": c,
                              "sources": tuple(sources)})
                star_of_v[v].append(idx)
        
        def dfs(stars, star_of_v):
            stack = deque()
            uf0 = nx.utils.UnionFind()
            stack.append(([], set(), set(), uf0, list(star_of_v.keys()), dict()))

            while stack:
                chosen, included, resolved, uf, frontier, chosen_colors = stack.pop()

                if not frontier:
                    included_uf = nx.utils.UnionFind([x for x in included])
                    for idx in chosen:
                        s = stars[idx]
                        nodes = set(s["sources"]) | {s["target"]}
                        root = next(iter(nodes))
                        for w in nodes:
                            included_uf.union(root, w)
                    if sum(1 for _ in included_uf.to_sets()) == 1:
                        yield self._materialize_subgraph(chosen, stars)
                    continue
                
                v, *rest_frontier = frontier

                if v not in included:
                    stack.append((chosen, 
                                  included, 
                                  resolved | {v},
                                  uf, 
                                  rest_frontier,
                                  chosen_colors.copy()))

                for idx in star_of_v[v]:
                    s = stars[idx]
                    color = s["color"]

                    if v in chosen_colors and chosen_colors[v] != color:
                        continue

                    new_nodes = set(s["sources"]) | {s["target"]}
                    new_included = included | new_nodes
                    new_resolved = resolved | {v}

                    new_uf = deepcopy(uf)
                    root = next(iter(new_nodes))
                    for w in new_nodes:
                        new_uf.union(root, w)

                    new_frontier = rest_frontier + [w for w in new_nodes if w not in included and w not in new_resolved]

                    new_chosen_colors = chosen_colors.copy()
                    new_chosen_colors[v] = color

                    stack.append((chosen + [idx],
                                  new_included,
                                  new_resolved,
                                  new_uf,
                                  new_frontier,
                                  new_chosen_colors))

        return list(dfs(stars, star_of_v))
    

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

        plt.show()

    def visualize_subnetworks(self):
        pass
