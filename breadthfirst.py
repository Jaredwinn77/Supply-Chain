import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from Skeleton import visualize_graph

def make_graph():
    """{'nodes': [0, 1, 2, 3],
    'edges': [(0, 3), (1, 0), (1, 2), (1, 3), (2, 0),
     (2, 1), (3, 1), (3, 0)]
    'attributes': {(0, 3): 0, (1, 0): 1, (1, 2): 1, (1, 3): 0, (2, 0): 1,
     (2, 1): 0, (3, 1): 1, (3, 0): 0}}"""
    num_nodes = 4
    edges = [(0, 3), (1, 0), (1, 2), (1, 3), (2, 0),
     (2, 1), (3, 1), (3, 0)]
    colors =[0,1,1,0,1,0,1,0]
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for edge, color in zip(edges,colors):
      start_node, end_node = edge
      G.add_edge(start_node,end_node, color = color)
    return G




def breadth_first(G,starting_node):
    completed_subnetworks = []
    working_subnetworks = []

    # Initialize with starting node which contains both the subnetwork and the list of visited nodes
    first_sub = [[starting_node],[starting_node]]
    working_subnetworks.append(first_sub)
    i = 0
    while working_subnetworks:
        # This is to store the new subnetworks that will be created
        next_working_subnetworks = []
        for subnetwork in working_subnetworks:
            print(i)
            i +=1
            # set up code
            working_level = subnetwork[0]
            visited_nodes = subnetwork[1]
            visited = False

            # Collects incoming edges
            for node in working_level:
                incoming_edges = list(G.in_edges(node, data=True))
                color_groups = {}
                # Creates dictionary of colors flowing into the working node
                for u, v, data in incoming_edges:
                    color = data.get('color')
                    if color not in color_groups:
                        color_groups[color] = []
                    color_groups[color].append(u)
                for color, nodes in color_groups.items():
                    if any(n in visited_nodes for n in nodes):
                        visited = True
                        completed_subnetwork = subnetwork.copy()
                        completed_subnetwork[0].append(node)
                        completed_subnetworks.append(completed_subnetwork)
                        break
                    new_subnetwork = [subnetwork[0].copy(), visited_nodes.copy()]  # Copy current nodes and visited nodes
                    new_subnetwork[0].append(nodes)  # Append the new group of nodes
                    next_working_subnetworks.append(new_subnetwork)
            if visited:
                continue
        working_subnetworks = next_working_subnetworks

    return completed_subnetworks






def main():
    G = make_graph()

    #visualize_graph(G)

    subnetworks = breadth_first(G,0)
    print(subnetworks)



if __name__=="__main__":
    main()