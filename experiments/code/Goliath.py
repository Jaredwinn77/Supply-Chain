from Graphclass_module import Graphclass
import networkx as nx
import numpy as np
import random


def generate_random_strongly_connected_graph(num_nodes, edge_probability=0.01, edge_color_range=(0, 3),
                                             weight_range=(0, 2)):
    """
    Generates a strongly connected random directed graph with specified number of nodes,
    assigns random integer edge color attributes, random edge weight attributes, and returns the graph.

    Parameters:
    - num_nodes: Number of nodes in the graph.
    - edge_probability: Probability that an edge exists between any two nodes.
    - edge_color_range: The range for edge color attributes (defaults to [0, 255]).
    - weight_range: The range for edge weight attributes (defaults to [1, 10]).

    Returns:
    - G: A strongly connected directed graph with edge color and weight attributes.
    """
    # Step 1: Create an empty directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    # Step 2: Add random edges with a given probability (excluding self-loops)
    for i in range(num_nodes):
        print(i)
        for j in range(num_nodes):
            if i != j and random.random() < edge_probability:
                G.add_edge(i, j)

    # Step 3: Ensure the graph is strongly connected
    if not nx.is_strongly_connected(G):
        print("not connected")
        components = list(nx.strongly_connected_components(G))
        print(len(components))
        # If the graph is not strongly connected, connect the components
        for i in range(1, len(components)):
            src_component = components[i - 1]
            dst_component = components[i]
            src_node = random.choice(list(src_component))
            dst_node = random.choice(list(dst_component))
            G.add_edge(src_node, dst_node)

    # Step 4: Add random integer edge colors and random weights
    for u, v in G.edges():
        # Random integer color and weight attributes for each edge
        G[u][v]['color'] = random.randint(*edge_color_range)  # Random color in the specified range
        G[u][v]['weight'] = random.randint(*weight_range)  # Random weight in the specified range

    return G


def save_graph_as_npy(G, filename="Goliath.npy"):
    """
    Saves the graph and its attributes to a .npy file for efficient storage.

    Parameters:
    - G: The graph to be saved.
    - filename: The file path where the graph will be saved.
    """
    # Save the graph structure (nodes and edges)
    node_list = list(G.nodes)
    edge_list = list(G.edges)

    # Store the edges and corresponding color and weight attributes
    edge_colors = np.array([G[u][v]['color'] for u, v in edge_list])
    edge_weights = np.array([G[u][v]['weight'] for u, v in edge_list])

    # Save nodes, edges, edge colors, and edge weights in an efficient format using numpy
    np.save(filename,
            {'nodes': node_list, 'edges': edge_list, 'edge_colors': edge_colors, 'edge_weights': edge_weights})


def load_graph_from_npy(filename="Goliath.npy"):
    """
    Loads the graph from the saved .npy file.

    Parameters:
    - filename: The file path from which to load the graph.

    Returns:
    - G: The loaded graph with edge color and weight attributes.
    """
    data = np.load(filename, allow_pickle=True).item()

    # Create the graph
    G = nx.DiGraph()
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])

    # Create the adjacency matrix for edge weights
    num_nodes = len(G.nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Fill the adjacency matrix with weights
    edge_colors = data['edge_colors']
    edge_weights = data['edge_weights']

    for idx, (u, v) in enumerate(data['edges']):
        # Assuming nodes are indexed from 0 to num_nodes-1
        adj_matrix[u, v] = edge_weights[idx]

        # Add the attributes to the graph
        G[u][v]['color'] = edge_colors[idx]
        G[u][v]['weight'] = edge_weights[idx]

    return G, adj_matrix
def main():
    num_nodes = 25000
    G = generate_random_strongly_connected_graph(num_nodes, edge_probability=0.01, edge_color_range=(0, 100), weight_range=(0, 2))
    save_graph_as_npy(G, filename="GoliathV.npy")

    # Load the graph back from the .npy file
    #loaded_G, weights = load_graph_from_npy(filename="GoliathI.npy")
    #print("loaded g")
    #g = Graphclass(loaded_G, weights)
    #g.simulate_linear()
    #print("simulated")
    #g.visualize_graph(loaded_G)
    #g.visualize_flow()



if __name__=="__main__":
    main()