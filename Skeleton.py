import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import copy

""""Contains the bones used for supply chain experiments"""

def create_graph(num_nodes,num_edges,num_colors):
  """Returns a strongly connected graph with a specified number of nodes, edges, and colored edges"""
  strong = False
  while not strong:
    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    # Calculate possible edges-- this does not inlcude self referential edges
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
  nx.set_edge_attributes(G,edge_color_map,'color')

  return G

def visualize_graph(G):
  """Draws a graph with colored edges, provides a legend"""
  colors = [data['color'] for _, _, data in G.edges(data=True)]
  cmap = plt.get_cmap('viridis')
  norm = mcolors.Normalize(vmin=min(colors), vmax=max(colors))
  edge_colors = [cmap(norm(color)) for color in colors]
  unique_colors = sorted(set(colors))
  patches_list = [patches.Patch(color=cmap(norm(color)), label=f'Color {color}') for color in unique_colors]
  pos = nx.spring_layout(G) #Usually uses spring
  nx.draw(G, pos, with_labels=True, edge_color=edge_colors, width=2, node_size=500,connectionstyle='arc3, rad = 0.05')
  plt.legend(handles=patches_list, title="Edge Colors")
  plt.show()


def draw_subnetwork(G, subnetwork_nodes, title):
    """Draw a single subnetwork, this function is called as part of visualize_subnetworks"""
    subgraph = G.subgraph(subnetwork_nodes)
    pos = nx.spring_layout(subgraph)  # Position nodes using the spring layout
    edges = subgraph.edges()
    labels = nx.get_edge_attributes(subgraph, 'color')

    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, pos, with_labels=True, edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=labels, font_color='red')
    plt.title(title)
    plt.show()

def visualize_subnetworks(G, subnetworks):
  """Plots all of the subnetworks found by find_subnetworks"""
  for i, subnetwork in enumerate(subnetworks):
      draw_subnetwork(G, subnetwork, f'Subnetwork {i + 1}')

def find_subnetworks(G,start,visited=None,current_subnetwork=None):
  """Algorithm recursively finds subnetworks, this version does not take into account edge colors"""
  # Base cases
  if visited is None:
    visited = set()
  if current_subnetwork is None:
    current_subnetwork = []
  # Adds current node to both visited and current subnetwork
  visited.add(start)
  current_subnetwork.append(start)
  # Finds all incoming edges
  incoming_edges = list(G.in_edges(start))
  subnetworks = []
  # u and v are each nodes which are connected by one of the edges in incoming_edges
  for u,v in incoming_edges:
    # checks if u has alread been evaluated
    if u not in visited:
      # If u was not evaluated it is used as the starting node in a new search
      new_subnetworks = find_subnetworks(G,u,visited.copy(),current_subnetwork.copy())
      # the results of the new search are added to the subnetworks
      subnetworks.extend(new_subnetworks)
    else:
      # This logic deals with when there is a loop
      if u in current_subnetwork:
        loop_subnetwork = current_subnetwork + [u]
        subnetworks.append(loop_subnetwork)
  # removes the current network once it has been fully explored
  current_subnetwork.pop()
  # allows for the same node to appear in different subnetworks
  visited.remove(start)
  # stops when there are no more unexplored paths
  if not incoming_edges or not subnetworks:
    return [current_subnetwork]
  # subnetworks is a list of all subnetworks
  return subnetworks
def find_subnetworks_color(G,start,visited=None,current_subnetwork=None):
  """Implementation very similar to above, with the exception of the logic which is implemented to deal with colors"""
  # Base cases
  if visited is None:
    visited = set()
  if current_subnetwork is None:
    current_subnetwork = []
  # Adds current node to both visited and current subnetwork
  visited.add(start)
  current_subnetwork.append(start)
  # Finds all incoming edges
  incoming_edges = list(G.in_edges(start,data=True))
  color_groups = {}
  # This section creates a dictionary of all the colors flowing into the node
  for u,v, data in incoming_edges:
    color = data.get('color')
    if color not in color_groups:
      color_groups[color] = []
    color_groups[color].append(u)
  # Starts a subnetworks list, each subnetwork will be appended once it has been fully explored
  subnetworks = []
  # This section is essentially the same as before, except now it does all edges of the same color at the same time
  for color, nodes in color_groups.items():
    for u in nodes:
      if u not in visited:
        new_subnetworks = find_subnetworks_color(G, u, visited.copy(), current_subnetwork.copy())
        subnetworks.extend(new_subnetworks)
      else:
        # Deals with loops
        if u in current_subnetwork:
          loop_subnetwork = current_subnetwork + [u]
          subnetworks.append(loop_subnetwork)
  # Moves on to the next subnetwork and clears the variables that need to reset
  current_subnetwork.pop()
  visited.remove(start)
  # Stopping criteria
  if not incoming_edges or not subnetworks:
    return [current_subnetwork]
  return subnetworks

def create_weights(G,num_nodes):
  n = num_nodes
  matrix = np.random.rand(n,n)
  column_sums = matrix.sum(axis=0)
  column_sums[column_sums == 0] = 1
  normed = matrix / column_sums
  return normed


def spectral_radius(weighted, subnetworks):
  radii = []
  for subnetwork in subnetworks:
    subnetwork_matrix = weighted[np.ix_(subnetwork, subnetwork)]
    rho = max(abs(np.linalg.eigvals(subnetwork_matrix)))
    radii.append(rho)

  return radii

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
    edge_lists = []
    # We add each element as a list
    working_subnetworks.append([[starting_node],[starting_node],[]])

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
                        # add an edge list
                        new_subnetwork[-1].append((node,color_groups[key]))
                        completed_subnetworks.append(new_subnetwork[0])
                        edge_lists.append(new_subnetwork[-1])
                        visited = False
                    else:
                        new_subnetwork[0].append(color_groups[key])
                        new_subnetwork[-1].append((node, color_groups[key]))
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
    return final, edge_lists
