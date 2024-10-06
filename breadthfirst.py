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
    """This function is a breadth first approach to identifying subnetworks, it iterates through each subnetwork in working_subnetworks, adding a single level
    this could be more than a single edge/node pair if the edge is connected to more than one other node via edges of the same color. Once the stopping criteria are met the
    subnetwork is popped off and added to completed_subnetworks"""
    """Psuedo code:
    Create a subnetwork containing the starting node, add it to working_subnetworks, record it in visited_nodes
    for every subnetwork in working subnetworks
    for each of the nodes in the last entry (just in case more than node flowed into the predecessor with the same color,this should usually be only one node)
    find all of the nodes which flow into that node and group them by color
    for each of the colors, check if the nodes have been visited
    if they have, add them to the graph, add it to the completed_subnetworks, and remove the graph from working_subnetworks
    if not add them to a copy of a subnetwork and remove the original
    when working_subnetworks is empty return the completed_subnetworks
    """
    # Initialize lists
    completed_subnetworks = []
    working_subnetworks = []

    # We add each element as a list
    working_subnetworks.append([[starting_node],[starting_node]])

    visited = False

    # Loop through each subnetwork, adding one node to each network and popping of networks once they are finished
    while working_subnetworks:
        # We work on each subnetwork individually
        for subnetwork in working_subnetworks:
            # This should pop the current subnetwork not the first one
            index = working_subnetworks.index(subnetwork)
            working_subnetworks.pop(index)
            working_level = subnetwork[0][-1]
            if isinstance(working_level,list):
                k = len(working_level)
            else:
                k = 1
            for i in range(k):
                if isinstance(working_level, list):
                    node = working_level[i]
                else:
                    node = working_level
                # Collect all incoming edges
                incoming_edges = list(G.in_edges(node,data = True))
                # Creates a dictionary of all colors flowing into a node
                color_groups = {}
                for u, v, data in incoming_edges:
                    color = data.get('color')
                    if color not in color_groups:
                        color_groups[color] = []
                    color_groups[color].append(u)
                # for each of the color groups we create a copy of the subnetwork and append all of the nodes in that group
                for key in color_groups.keys():
                    # using deep copy ensures that the correct subnetworks are modified
                    new_subnetwork = copy.deepcopy(subnetwork)
                    for value in color_groups[key]:
                    # we need to check if the nodes are already in visited
                        if value in new_subnetwork[1]:
                            visited = True
                        else:
                            new_subnetwork[1].append(value)
                    if visited:
                        new_subnetwork[0].append(color_groups[key])
                        completed_subnetworks.append(new_subnetwork[0])
                        visited = False
                    else:
                        new_subnetwork[0].append(color_groups[key])
                        working_subnetworks.append(new_subnetwork)
    final = []
    for network in completed_subnetworks:
        flat = []
        stack = list(network)
        while stack:
            item = stack.pop()
            if isinstance(item,list):
                stack.extend(item)
            else:
                flat.append(item)
        final.append(flat[::-1])
    return final





def main():
    G = make_graph()

    #visualize_graph(G)

    subnetworks = breadth_first(G,0)
    print(subnetworks)



if __name__=="__main__":
    main()
