import numpy as np
import networkx as nx
import random
import copy
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML, display


class Graphclass:
    """"Class for studying the behavior of dynamical systems through networks
    Attributes:
     num_nodes-- from constructor 
     G-- random graph object
     weights-- weighted adjacency matrix
     subnetworks-- list of subnetworks returned from a BFS with a single starting node
     radii-- the spectral radii of each subnetwork
     history-- a dictionary containing the simulated behavior of each node over time
     converged-- a list of the values that each node converged to
     zero_nodes-- a list of all the nodes whose flow went to zero
     predicted-- a list of the nodes who were predicted to go to zero
     full_subnetwork-- a list of the subnetworks obtained using each node as a starting point
     full_predicted-- a list of the nodes predicted to go to zero using full_subnetwork
     T -- boolean value whether the system converged
     full_radii-- a list of radii corrosponding to the dirty full predicted networks
     p-- the matrix used for pertubations (only exists if this is a perturbation)
     full -- boolean, whether or not the full BFS is used
    Methods:
        use_premade(G,W)-- overrides the random graph and replaces it with a premade graph object
        make_custom_graph(edge_list)-- overrides the random graph and with a graph created from the list of tuples passed in
        create_weights(rho=2)-- creates a random weighted adjacency matrix
        breadth_first(starting_node=0)-- performs a BFS, returns a list of subnetworks
        perturb_weights(prange=(-0.5,0.5)-- replaces the weights attribute with a perturbed version 
        
        visualize_graph: creates and displays a plot of graph structure
        visualize_subnetworks: creates and displays a plot of each subnetwork
        simulate_linear: returns the behavior of the system and the value each node converged to
            optional argument initial 
        visualize_behavior: creates and displays a plot of the behavior of the system
        visualize_flow: creates and displays an animation of the behavior of the system 
        perturb_weights: randomly varies the weight matrix
        visualize_node_flow(node, depth): creates an animation showing the flow of the target node and all neighbors within a specified depth
        """""


    def __init__(self, num_nodes, num_edges, num_colors, full=False):
        """"Allows for a graph object to be input or the generation of a random graph"""
        self.num_nodes = num_nodes
        self.full = full
        self.G = self.create_graph(num_nodes, num_edges, num_colors)
        self.weights = self.create_weights()
        self.subnetworks, self.dirty_subnetworks = self.breadth_first()
        self.dirty_radii = self.spectral_radius(self.dirty_subnetworks)
        self.dirty_predicted = self.predict(self.dirty_radii, self.dirty_subnetworks)
        self.radii = self.spectral_radius(self.subnetworks)
        self.history, self.converged, self.zero_nodes = self.simulate_linear()
        self.predicted = self.predict(self.radii,self.subnetworks)
        if full:
            self.full_subnetwork, self.dirty_full_subnetwork = self.full_subnetworks()
            self.full_predicted, self.full_radii = self.predict_full(self.full_subnetwork)
            self.dirty_full_predicted, self.dirty_full_radii = self.predict_full(self.dirty_full_subnetwork)

    def use_premade(self, G, W):
        self.G = G
        self.weights = W
        self.num_nodes = len(G.nodes)
    def create_graph(self, num_nodes, num_edges, num_colors):
        """Returns a strongly connected graph with a specified number of nodes, edges, and colored edges"""
        strong = False
        while not strong:
            # Create graph
            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))

            # Calculate possible edges-- this does not include self-referential edges
            all_possible_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
            random.shuffle(all_possible_edges)

            # selects the specified number of edges and adds them to the graph
            edges_to_add = all_possible_edges[:num_edges]
            G.add_edges_from(edges_to_add)
            # Checks if the graph is strongly connected
            strong = nx.is_strongly_connected(G)
        # Assign random colors to edges
        edge_colors = [random.randint(0, num_colors - 1) for _ in range(len(G.edges))]
        edge_color_map = dict(zip(G.edges(), edge_colors))
        nx.set_edge_attributes(G, edge_color_map, 'color')
        return G
    def make_custom_graph(self, edges):
        #edges = [(0, 2, 1), (0, 1, 1), (1, 3, 0), (2, 0, 1), (2, 3, 0), (3, 2, 0)]
        #edges = [(0, 2, 1),(0, 1, 0),(1, 0, 1),(1, 3, 1),(2,1,1),(3,2,0)]
        #edges = [(0,3,1),(1,2,1),(2,1,1),(2,0,0),(3,0,1),(3,2,1)]
        G = nx.DiGraph()
        for u, v, color in edges:
            G.add_edge(u, v, color=color)
        self.G = G
        self.weights = self.create_weights()
        self.num_nodes = len(edges)
    def create_weights(self, rho=2):
        """"Creates a random weight matrix for a given graph"""
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.G.edges():
            matrix[i, j] = np.random.uniform(0, rho)
        return matrix
    def breadth_first(self, starting_node=0):
        """Implementation of the BFS algorithm for subnetwork detection """
        completed_subnetworks = []
        working_subnetworks = []
        # [[working levels],[visited nodes],[edge list]]
        working_subnetworks.append([[starting_node], [], []])
        # Loops as long as a subnetwork remains incomplete
        while working_subnetworks:
            # Processes each subnetwork
            for subnetwork in working_subnetworks:
                subnetwork[1] = []
                for entry in subnetwork[2]:
                    node1, node2, color = entry
                    subnetwork[1].append(node1)
                    subnetwork[1].append(node2)
                # Identifies subnetwork and pops it off
                index = working_subnetworks.index(subnetwork)
                working_subnetworks.pop(index)
                # Selects the level to begin searching from and the number of branches
                working_level = subnetwork[0]
                if isinstance(working_level, list):
                    num_branches = len(working_level)
                else:
                    num_branches = 1
                # each branch will be added to branch list, the combinations will be used to create the new subnetworks
                branch_list = []
                edges = {}
                for i in range(num_branches):
                    # this will contain the next level of nodes to be searched for this branch
                    branch = []
                    # Determines which node to evaluate
                    if isinstance(working_level, list):
                        node = working_level[i]
                    else:
                        node = working_level

                    incoming_edges = list(self.G.in_edges(node, data=True))
                    # creates color dictionary
                    color_groups = {}
                    for u, v, data in incoming_edges:
                        color = data.get('color')
                        if color not in color_groups:
                            color_groups[color] = []
                        color_groups[color].append(u)
                    for key in color_groups.keys():
                        colorset = []
                        for value in color_groups[key]:
                            if value in subnetwork[1]:
                                colorset.append((value, node, key))
                            else:
                                colorset.append(value)
                                subnetwork[1].append(value)
                                # add the edge to the edge dict
                                edges[value] = (node, key)
                        branch.append(colorset)
                    branch_list.append(branch)
                cleaned = []
                if len(branch_list) == 1:
                    for i in range(len(branch_list[0])):
                        cleaned.append(branch_list[0][i])
                else:
                    combinations = list(itertools.product(*branch_list))
                    for combo in combinations:
                        combo = self.flatten(combo)
                        cleaned.append(combo)
                for combo in cleaned:
                    new_subnetwork = copy.deepcopy(subnetwork)
                    # new code to add dead edges correctly
                    # make sure to iterate over a copy or the modifications will cause skipped indices
                    for entry in combo[:]:
                        if isinstance(entry, tuple):
                            new_subnetwork[2].append(entry)
                            combo.remove(entry)
                    for node1, (node2, color) in edges.items():
                        if node1 in combo:
                            new_subnetwork[2].append((node1, node2, color))
                    combo = [item for item in combo if isinstance(item, int)]
                    if len(combo) == 0:
                        completed_subnetworks.append(new_subnetwork[2])
                    else:
                        new_subnetwork[0] = combo
                        working_subnetworks.append(new_subnetwork)

        subnetworks = self.remove_dup(self.remove_hangers(completed_subnetworks))
        return subnetworks, completed_subnetworks
    def flatten(self, nested_list):
        """Helper function to flatten nested lists"""
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(self.flatten(item))  # Recursively flatten the nested list
            else:
                result.append(item)  # Append the item if it's not a list
        return result
    def remove_dup(self,subnetworks):
        """removes any duplicate subnetworks"""
        seen = set()
        result = []

        for sublist in subnetworks:
            # Sort the tuples in the sublist to get a consistent representation
            sorted_sublist = tuple(sorted(sublist))

            # If this sorted sublist is already in the seen set, skip it
            if sorted_sublist not in seen:
                result.append(sublist)  # Keep this sublist
                seen.add(sorted_sublist)  # Add the sorted version to the seen set

        return result
    def remove_hangers(self, subnetworks):
        """Remove all edges of nodes of degree 1."""
        cleaned = []

        for subnetwork in subnetworks:
            # Track node degrees across the entire subnetwork
            node_degree = Counter()
            # First pass: count the degree of each node in the subnetwork
            for node1, node2, color in subnetwork:
                node_degree[node1] += 1
                node_degree[node2] += 1

            # Remove edges with nodes that have degree 1
            done = False
            while not done:
                # Identify nodes with degree 1 (hanging nodes)
                hanging_nodes = {node for node, degree in node_degree.items() if degree == 1}

                if not hanging_nodes:
                    # No hanging nodes, we can stop
                    done = True
                else:
                    # Filter out the edges involving hanging nodes
                    filtered_edges = [(node1, node2, color) for (node1, node2, color) in subnetwork if
                                      node1 not in hanging_nodes and node2 not in hanging_nodes]

                    # Update the node degrees after filtering
                    node_degree = Counter()
                    for node1, node2, color in filtered_edges:
                        node_degree[node1] += 1
                        node_degree[node2] += 1

                    # Set the filtered edges as the new subnetwork
                    subnetwork = filtered_edges

            # After cleaning, add the subnetwork to the result
            cleaned.append(subnetwork)
        return cleaned
    def visualize_graph(self, G):
        """Draws a graph with colored edges, provides a legend"""
        plt.clf()
        colors = [data['color'] for _, _, data in G.edges(data=True)]
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=min(colors), vmax=max(colors))
        edge_colors = [cmap(norm(color)) for color in colors]
        unique_colors = sorted(set(colors))
        patches_list = [patches.Patch(color=cmap(norm(color)), label=f'Color {color}') for color in unique_colors]
        pos = nx.spring_layout(G)  # Usually uses spring
        nx.draw(G, pos, with_labels=True, edge_color=edge_colors, width=2, node_size=500,
                connectionstyle='arc3, rad = 0.05')
        plt.legend(handles=patches_list, title="Edge Colors")
        plt.show()
    def visualize_subnetworks(self, subnetworks):
        for subnetwork in subnetworks:
            G = nx.DiGraph()
            for u, v, color in subnetwork:
                G.add_edge(u, v, color=color)
            self.visualize_graph(G)
    def spectral_radius(self, subnetworks):
        """Calculates the spectral radii for each subnetwork in a list of subnetworks"""
        num = self.num_nodes
        weights = self.weights
        radii = []
        for subnetwork in subnetworks:
            matrix = np.zeros((num,num))
            for node1, node2, _ in subnetwork:
                matrix[node1][node2] = weights[node1][node2]
            rho = max(abs(np.linalg.eigvals(matrix)))
            radii.append(rho)


        return radii
    def count_subnetworks(self,subnetworks):
        """counts number of subnetworks"""
        num = 0
        for sub in subnetworks:
            num += 1
        self.num_subnetworks = num
    def are_subnetworks_identical(self, subnetworks):
        """Check if all subnetworks in the list are identical, ignoring order."""
        example = subnetworks[0]
        correctlen = len(example)
        for network in subnetworks:
            if len(network) != correctlen:
                self.identical =  False
                return
        seen = set()
        for sublist in example:
            sorted_sublist = tuple(sorted(sublist))
            seen.add(sorted_sublist)
        for network in subnetworks[1:]:
            for sublist in network:
                s = tuple(sorted(sublist))
                if s not in seen:
                    self.identical = False
                    return
        self.identical = True
    def simulate_linear(self, initial=None, maxiter=100, epsilon=1e-10):
        G = self.G
        weights = self.weights
        self.T = False
        if initial == None:
            initial = {node: random.uniform(1, 10) for node in G.nodes}
        self.initial=initial
        history = {node: [initial[node]] for node in G.nodes}
        current = initial.copy()
        for i in range(maxiter):
            next = {}
            max_change = 0
            for node in G.nodes:
                incoming_edges = list(G.in_edges(node, data=True))
                color_groups = {}
                for u, v, data in incoming_edges:
                    color = data.get('color')
                    if color is not None:
                        edge_weight = weights[u][v]
                        state_of_neighbor = current[u]
                        if color not in color_groups:
                            color_groups[color] = 0
                        color_groups[color] += edge_weight * state_of_neighbor
                min_next_state = float('inf')
                for color, total_sum in color_groups.items():
                    if total_sum > 0:
                        min_next_state = min(min_next_state, total_sum)
                if min_next_state == float('inf'):
                    next[node] = 0
                else:
                    next[node] = min_next_state
                max_change = max(max_change, abs(next[node] - current[node]))

            for node, state in next.items():
                history[node].append(state)
            current = next
            if max_change < epsilon:
                self.T = True
                break
        converged = [current[node] for node in G.nodes]
        zero_nodes = set()
        for i in range(len(converged)):
            if np.abs(converged[i]) <= 1e-2:
                zero_nodes.add(i)
        return history, converged, zero_nodes
    def visualize_behavior(self):
        for node, states in self.history.items():
            plt.plot(states, label=f"Node {node}")

        plt.xlabel('Iteration')
        plt.ylabel('State')
        plt.title('Behavior Propagation through Network')
        plt.legend()
        plt.show()
    def predict(self,radii,subnetworks):
        predicted = set()
        for i in range(len(radii)):
            if radii[i] < 1:
                target = subnetworks[i]
                for node1,node2,_ in target:
                    predicted.add(node1)
                    predicted.add(node2)

        return predicted
    def full_subnetworks(self):
        full_subnetworks = []
        full_dirty = []
        for i in range(self.num_nodes):
            subnetworks, dirt = self.breadth_first(i)
            full_subnetworks.append(subnetworks)
            full_dirty.append(dirt)


        return full_subnetworks, full_dirty
    def predict_full(self, s):
        full_predicted = set()
        full_radii = []
        for subnetworks in s:
            radii = self.spectral_radius(subnetworks)
            full_radii.append(radii)
            nodes = self.predict(radii, subnetworks)
            for node in nodes:
                full_predicted.add(node)
       
        return full_predicted, full_radii

 

    def perturb_weights(self, prange=(-0.5,0.5)):
        new = Graphclass(4,6,1,self.full)
        new.use_premade(self.G, self.weights)
        p = self.weights.copy()
        perturbation = np.random.uniform(prange[0], prange[1], p.shape)
        p[p != 0] += perturbation[p != 0]
        new.p = p
        new.weights = p
        new.initial = self.initial.copy()
        new.subnetworks, new.dirty_subnetworks = self.subnetworks, self.dirty_subnetworks
        new.dirty_radii = new.spectral_radius(new.dirty_subnetworks)
        new.dirty_predicted = new.predict(new.dirty_radii, new.dirty_subnetworks)
        new.radii = new.spectral_radius(new.subnetworks)
        new.history, new.converged, new.zero_nodes = new.simulate_linear(self.initial)
        new.predicted = new.predict(new.radii, new.subnetworks)
        if self.full:
            new.dirty_full_subnetwork, new.full_subnetwork = self.dirty_full_subnetwork, self.full_subnetwork
            new.full_predicted, new.full_radii = new.predict_full(new.full_subnetwork)
            new.dirty_full_predicted, new.dirty_full_radii = new.predict_full(new.dirty_full_subnetwork)

        return new
    
    def visualize_flow(self):
     
      
        
        fig, ax = plt.subplots(figsize=(7, 7))
       
       
        pos = nx.spring_layout(self.G, seed=42, k=0.9, scale=4)  # k adjusts spacing between nodes, scale makes them bigger

        
        colors = ['black','red', 'orange', 'yellow', 'lightgreen', 'green']
        cmap = LinearSegmentedColormap.from_list("red_orange_yellow_green", colors)
        norm = mcolors.Normalize(vmin=min(min(steps) for steps in self.history.values()),
                                vmax=max(max(steps) for steps in self.history.values()))

   
        edge_colors = [data['color'] for _, _, data in self.G.edges(data=True)]
        edge_cmap = plt.get_cmap('viridis')
        edge_norm = mcolors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
        edge_colors = [edge_cmap(edge_norm(color)) for color in edge_colors]

     
        nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=0.5, alpha=0.5, ax=ax)
        node_colors = [self.history[node][0] for node in self.G.nodes]
        node_scatter = nx.draw_networkx_nodes(self.G, pos, node_size=20, node_color=node_colors, cmap=cmap, ax=ax)
        cbar = plt.colorbar(node_scatter, ax=ax, orientation='vertical')
        def update_frame(t):
            node_colors = [self.history[node][t] for node in self.G.nodes]
            node_scatter.set_array(np.array(node_colors))  
            return node_scatter,

        ani = animation.FuncAnimation(fig, update_frame, frames=len(next(iter(self.history.values()))),
                                    interval=1000, blit=True)

        plt.title('Node Flow Animation')
        plt.axis('off')
    

        html = HTML(ani.to_jshtml())
        display(html)

    def visualize_node_flow(self, node, depth):
        fig, ax = plt.subplots(figsize=(7, 7))
        subgraph_nodes = self.get_nodes_within_depth(node, depth)
        subgraph = self.G.subgraph(subgraph_nodes)

      
        pos = nx.spring_layout(subgraph, seed=42, k=0.9, scale=4)

        colors = ['black','red', 'orange', 'yellow', 'lightgreen', 'green']
        cmap = LinearSegmentedColormap.from_list("red_orange_yellow_green", colors)
        norm = mcolors.Normalize(vmin=min(min(steps) for steps in self.history.values()),
                                vmax=max(max(steps) for steps in self.history.values()))
        
        edge_colors = [data['color'] for _, _, data in subgraph.edges(data=True)]
        edge_cmap = plt.get_cmap('viridis')
        edge_norm = mcolors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
        edge_colors = [edge_cmap(edge_norm(color)) for color in edge_colors]

       
        nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, width=0.5, alpha=0.5, ax=ax)

       
        node_colors = [self.history[node][0] for node in subgraph.nodes]
        node_scatter = nx.draw_networkx_nodes(subgraph, pos, node_size=20, node_color=node_colors, cmap=cmap, ax=ax)
        cbar = plt.colorbar(node_scatter, ax=ax, orientation='vertical')

        def update_frame(t):
            """
            Update the node colors for each frame in the animation.
            Only nodes within the specified depth from the central node are updated.
            """
            node_colors = [self.history[node][t] for node in subgraph.nodes]
            node_scatter.set_array(np.array(node_colors))  # Update the node colors for each frame
            return node_scatter,

        ani = animation.FuncAnimation(fig, update_frame, frames=len(next(iter(self.history.values()))),
                                    interval=100, blit=True)

        plt.title(f'Node Flow Animation (Center: Node {node}, Depth: {depth})')
        plt.axis('off')


        html = HTML(ani.to_jshtml())
        display(html)

    def get_nodes_within_depth(self, node, depth):
        """
        Returns the list of nodes within the specified depth from the given node.
        This uses a breadth-first search (BFS) to find the nodes within the specified distance.
        """
        visited = set()
        to_visit = [(node, 0)]  # (current_node, current_depth)
        visited.add(node)
        nodes_within_depth = [node]

        while to_visit:
            current_node, current_depth = to_visit.pop(0)
            if current_depth < depth:
                # Add neighbors to the list if they haven't been visited
                for neighbor in self.G.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nodes_within_depth.append(neighbor)
                        to_visit.append((neighbor, current_depth + 1))

        return nodes_within_depth
    def perturb_initial(self, prange=(-1e-5,1e-5)):
        new = Graphclass(4,6,1,self.full)
        new.use_premade(self.G, self.weights)
        p = self.initial.copy()
        for i, value in p.items():
            perturbation = np.random.uniform(prange[0], prange[1])
            p[i] += perturbation
            
        new.initial = p 
        new.subnetworks, new.dirty_subnetworks = self.subnetworks, self.dirty_subnetworks
        new.dirty_radii = new.spectral_radius(new.dirty_subnetworks)
        new.dirty_predicted = new.predict(new.dirty_radii, new.dirty_subnetworks)
        new.radii = new.spectral_radius(new.subnetworks)
        new.history, new.converged, new.zero_nodes = new.simulate_linear(new.initial)
        new.predicted = new.predict(new.radii, new.subnetworks)
        if self.full:
            new.dirty_full_subnetwork, new.full_subnetwork = self.dirty_full_subnetwork, self.full_subnetwork
            new.full_predicted, new.full_radii = new.predict_full(new.full_subnetwork)
            new.dirty_full_predicted, new.dirty_full_radii = new.predict_full(new.dirty_full_subnetwork)

        return new
    
    def initial_lyapunov(self, initial=None, maxiter=100, epsilon=1e-10,perturb=1e-5):
        if initial is None:
            initial = {node: random.uniform(1, 10) for node in self.G.nodes}
        self.initial = initial
        
        # Initialize the unperturbed and perturbed states
        current = initial.copy()
        perturbed = {node: current[node] + random.uniform(-perturb, perturb) for node in self.G.nodes}
        
        history_current = {node: [current[node]] for node in self.G.nodes}
        history_perturbed = {node: [perturbed[node]] for node in self.G.nodes}
        
        max_change = 0
        for i in range(maxiter):
            next_current = {}
            next_perturbed = {}
            
            for node in self.G.nodes:
                incoming_edges = list(self.G.in_edges(node, data=True))
                color_groups_current = {}
                color_groups_perturbed = {}
                
                for u, v, data in incoming_edges:
                    color = data.get('color')
                    if color is not None:
                        edge_weight = self.weights[u][v]
                        state_of_neighbor_current = current[u]
                        state_of_neighbor_perturbed = perturbed[u]
                        
                        if color not in color_groups_current:
                            color_groups_current[color] = 0
                        color_groups_current[color] += edge_weight * state_of_neighbor_current
                        
                        if color not in color_groups_perturbed:
                            color_groups_perturbed[color] = 0
                        color_groups_perturbed[color] += edge_weight * state_of_neighbor_perturbed
                
                min_next_state_current = float('inf')
                min_next_state_perturbed = float('inf')
                
                for color, total_sum in color_groups_current.items():
                    if total_sum > 0:
                        min_next_state_current = min(min_next_state_current, total_sum)
                for color, total_sum in color_groups_perturbed.items():
                    if total_sum > 0:
                        min_next_state_perturbed = min(min_next_state_perturbed, total_sum)
                
                next_current[node] = 0 if min_next_state_current == float('inf') else min_next_state_current
                next_perturbed[node] = 0 if min_next_state_perturbed == float('inf') else min_next_state_perturbed
            
            # Calculate the difference between the perturbed and unperturbed states
            perturbation_norm = np.linalg.norm([next_perturbed[node] - next_current[node] for node in self.G.nodes])
            perturbation_growth = np.log(perturbation_norm) if perturbation_norm > 0 else 0
            
            # Track the states and perturbation growth over time
            for node, state in next_current.items():
                history_current[node].append(state)
            for node, state in next_perturbed.items():
                history_perturbed[node].append(state)
            
            # Update the system states
            current = next_current
            perturbed = next_perturbed
            
            # If max_change is smaller than epsilon, break
            max_change = max(abs(next_current[node] - current[node]) for node in self.G.nodes)
            if max_change < epsilon:
                break
        
        # Compute the Lyapunov exponent
        total_lyapunov = 0
        for i in range(1, len(history_current[next(iter(self.G.nodes))])):
            growth = np.log(np.linalg.norm([history_perturbed[node][i] - history_current[node][i] for node in self.G.nodes]) /
                        np.linalg.norm([history_perturbed[node][0] - history_current[node][0] for node in self.G.nodes]))
            total_lyapunov += growth

        lyapunov_exponent = total_lyapunov / len(history_current[next(iter(self.G.nodes))])
        
        return lyapunov_exponent
    def weight_lyapunov(self, initial=None, maxiter=100, epsilon=1e-10,perturb=1e-5):
        pass