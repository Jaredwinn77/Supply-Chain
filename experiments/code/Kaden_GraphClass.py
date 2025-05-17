from Graphclass_module import Graphclass
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches



class Kaden_Graphclass(Graphclass):
    def __init__(self, num_nodes, num_edges, num_colors, method, full=False):
        super().__init__(num_nodes, num_edges, num_colors, method, full=False)
    

    def visualize_graph(self, G=None, radius=None): # I thought it was dumb that calling this method required me to pass in the graph object when it is a method already tied to that object
        G = self.G
        return super().visualize_graph(G, radius)

    def kill_node(self, subnetwork, dead_node):
        # subnetwork is a list of edges: (from_node, to_node, color)
        alive_nodes = set(n for edge in subnetwork for n in edge[:2])
        # alive_nodes.discard(dead_node)  # remove the dead node

        # Build adjacency and reverse adjacency maps by color
        out_map = {}  # node -> list of (to_node, color)
        in_map = {}   # node -> list of (from_node, color)
        
        for src, dst, color in subnetwork: # thisncaputres all of the edges in the subnetwork into dictionaries of to and from nodes
            out_map.setdefault(src, []).append((dst, color))
            in_map.setdefault(dst, []).append((src, color))
        
        dead_set = {dead_node}
        queue = [dead_node]

        while queue: # while our queue is nonempty
            current = queue.pop()
            for receiving_node, color in out_map.get(current, []):
                if receiving_node in dead_set: # nodes already dead, so skip this part of the loop
                    continue
                # Check if receiving node still has a valid set of incoming edges of the same color
                remaining_incoming = [src for src, c in in_map[receiving_node] if src not in dead_set and c == color]
                expected_incoming = [src for src, c in in_map[receiving_node] if c == color]
                
                if set(remaining_incoming) != set(expected_incoming):
                    # receiving node
                    dead_set.add(receiving_node)
                    queue.append(receiving_node)

        return dead_set, alive_nodes
    
    def kill_node_X(self, G, dead_node):
        dead_set = {dead_node}
        queue = deque([dead_node])
        visited_in_queue = {dead_node}
        edge_colors = set(nx.get_edge_attributes(G, "color").values())
        while queue:
            current = queue.popleft()
            for successor in G.successors(current):
                if successor in dead_set:
                    continue
                should_die = False
                for color in edge_colors:
                    color_preds = [
                        pred for pred in G.predecessors(successor)
                        if G.edges[pred, successor]["color"] == color
                    ]
                    alive_predecessors = [p for p in color_preds if p not in dead_set]
                    if len(alive_predecessors) == 0 and len(color_preds) > 0:
                        should_die = True
                        break
                if should_die:
                    dead_set.add(successor)
                    if successor not in visited_in_queue:
                        queue.append(successor)
                        visited_in_queue.add(successor)
        return dead_set

    def cascading_failure_of_subnetworks(self): # for each subnetwork, for each node, try assuming that node dies, kill it and see which nodes die.
        results = {}
        for subnetwork_id, subnetwork in enumerate(self.subnetworks): # each graph has multiple subnetworks, loop over each
            # subnetwork_id = self.spectral_radius([subnetwork])[0] if you want the spectral radius to be the identifier
            G_sub = nx.DiGraph()
            G_sub.add_edges_from([(u, v, {"color": color}) for u, v, color in subnetwork])
            all_nodes = set(G_sub.nodes())
            for node in all_nodes: # loop over each edge in the subnetwork
                dead_nodes = self.kill_node_X(G_sub, node)
                G_dead = nx.DiGraph()
                for u, v, color in subnetwork:
                    if u not in dead_nodes and v not in dead_nodes:
                        G_dead.add_edge(u, v, color=color)
                results[(subnetwork_id, node)] = G_dead
        return results
    
    def visualize_failed_graphs_sideby_side_w_subnetworks(self, failed_graphs_dict): # plot subnetworks side by side with cascading ones.
        subnetworks = self.subnetworks

        for graph_key_values in failed_graphs_dict.items():
            label_nodenum = graph_key_values[0] 
            label, nodenum = label_nodenum # unpack even more
            surviving_graph = graph_key_values[1]
            if surviving_graph:
                fig, axs = plt.subplots(1, 2)
                subnet = nx.DiGraph()
                for u, v, color in subnetworks[label]:
                    subnet.add_edge(u, v, color=color)
                # visualize graph method, with some slight changes here so I can visualize side by side
                colors = [data['color'] for _, _, data in subnet.edges(data=True)]
                cmap = plt.get_cmap('viridis')
                norm = mcolors.Normalize(vmin=min(colors), vmax=max(colors))
                edge_colors = [cmap(norm(color)) for color in colors]
                unique_colors = sorted(set(colors))
                patches_list = [patches.Patch(color=cmap(norm(color)), label=f'Color {color}') for color in unique_colors]
                # pos = nx.spring_layout(subnet, seed=42)
                nx.draw(subnet, ax=axs[0], with_labels=True, edge_color=edge_colors, width=2, node_size=500, connectionstyle='arc3, rad = 0.05')
                axs[0].legend(handles=patches_list, title="Edge Colors")
                # axs[0].set_title("Original Subnetwork w Radius of: " + str(self.radii[label]))
                axs[0].set_title(f"Original Subnetwork: {label}")

                colors_surviving = [data['color'] for _, _, data in surviving_graph.edges(data=True)]
                suriving_edges_color = [cmap(norm(color)) for color in colors_surviving]
                axs[1].set_title("Surviving starting at node: " + str(nodenum))
                nx.draw(surviving_graph, ax=axs[1], with_labels=True, edge_color = suriving_edges_color, width=2, node_size=500, connectionstyle='arc3, rad = 0.05')



                plt.show()
            else: 
                print(f"For subnetwork: {label}, and starting node: {nodenum}, network fully died.")




# graph_object = Kaden_Graphclass(4, 8, 2, 'random')
# graph_object.visualize_subnetworks(graph_object.subnetworks)
# print(len(graph_object.subnetworks))
# dicty = graph_object.cascading_failure_of_subnetworks()

# graph_object.visualize_failed_graphs_sideby_side_w_subnetworks(dicty)


# graph_object.visualize_graph()
