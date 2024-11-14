import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import copy
import itertools
from collections import Counter

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


def visualize_subnetworks(G, subnetworks):
  """Plots all of the subnetworks using the helper function draw_subnetworks"""
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

  for i, subnetwork in enumerate(subnetworks):
      draw_subnetwork(G, subnetwork, f'Subnetwork {i + 1}')


def create_weights(G,num_nodes):
  """"Creates a random weight matrix for a given graph"""
  n = num_nodes
  matrix = np.random.rand(n,n)
  column_sums = matrix.sum(axis=0)
  column_sums[column_sums == 0] = 1
  normed = matrix / column_sums
  return normed


def spectral_radius(weighted, subnetworks):
  """Calculates the spectral radii for each subnetwork in a list of subnetworks"""
  radii = []
  for subnetwork in subnetworks:
    subnetwork_matrix = weighted[np.ix_(subnetwork, subnetwork)]
    rho = max(abs(np.linalg.eigvals(subnetwork_matrix)))
    radii.append(rho)

  return radii

def flatten(nested_list):
    """Helper function to flatten nested lists"""
    result = []
    for item in nested_list:
      if isinstance(item, list):
        result.extend(flatten(item))  # Recursively flatten the nested list
      else:
        result.append(item)  # Append the item if it's not a list
    return result


def breadth_first(G, starting_node):
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
          node1, node2 = entry
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
        dead = {}
        edges = {}
        for i in range(num_branches):
          # this will contain the next level of nodes to be searched for this branch
          branch = []
          # Determines which node to evaluate
          if isinstance(working_level, list):
            node = working_level[i]
          else:
            node = working_level

          incoming_edges = list(G.in_edges(node, data=True))
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
              # found an error, a subnetwork inherits dead edges from other subnetworks
              if value in subnetwork[1]:
                # here we create the entry for the dead dict key: edge value: living nodes of a different color
                dead[(value, node)] = [
                  other_value for other_key, other_values in color_groups.items()
                  if other_key != key  # Ensure it's a different color group
                  for other_value in other_values
                  if other_value not in subnetwork[1]  # Only include nodes not in subnetwork[1]
                ]
                # if there are no other living nodes of the color we need to append a filler to the colorset
                if len(color_groups[key]) == 1:
                  colorset.append('d')
              else:
                colorset.append(value)
                subnetwork[1].append(value)
                # add the edge to the edge dict
                edges[value] = node
            branch.append(colorset)
          branch_list.append(branch)
        cleaned = []
        if len(branch_list) == 1:


          for i in range(len(branch_list[0])):
            cleaned.append(branch_list[0][i])
        else:
          combinations = list(itertools.product(*branch_list))
          for combo in combinations:
            combo = flatten(combo)
            cleaned.append(combo)
        for combo in cleaned:
          new_subnetwork = copy.deepcopy(subnetwork)
          # adds dead edges
          # this does not function correctly when there are multiple dead subnetworks
          for key, value_list in dead.items():
            if not any(item in combo for item in value_list):
              new_subnetwork[2].append(key)
          for node1, node2 in edges.items():
            if node1 in combo:
              new_subnetwork[2].append((node1, node2))
          combo = [item for item in combo if isinstance(item, int)]
          if len(combo) == 0:
            completed_subnetworks.append(new_subnetwork[2])
          else:
            new_subnetwork[0] = combo
            working_subnetworks.append(new_subnetwork)

    return remove_dup((remove_hangers(completed_subnetworks))), remove_dup(completed_subnetworks)


# redo to trim iteratively
def remove_hangers(subnetworks):
  """Remove all edges of nodes of degree 1."""
  cleaned = []

  for subnetwork in subnetworks:
    # Track node degrees across the entire subnetwork
    node_degree = Counter()
    # First pass: count the degree of each node in the subnetwork
    for edge in subnetwork:
      node_degree[edge[0]] += 1
      node_degree[edge[1]] += 1

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
        filtered_edges = [edge for edge in subnetwork if edge[0] not in hanging_nodes and edge[1] not in hanging_nodes]

        # Update the node degrees after filtering
        node_degree = Counter()
        for edge in filtered_edges:
          node_degree[edge[0]] += 1
          node_degree[edge[1]] += 1

        # Set the filtered edges as the new subnetwork
        subnetwork = filtered_edges

    # After cleaning, add the subnetwork to the result
    cleaned.append(subnetwork)
  return cleaned

def remove_dup(subnetworks):
    """removes any duplicate subnetworks"""
    seen = set()
    result = []

    for sublist in subnetworks:
      # Convert the sublist to a tuple so it can be added to a set
      tuple_sublist = tuple(sublist)

      if tuple_sublist not in seen:
        seen.add(tuple_sublist)
        result.append(sublist)

    return result
def num_duplicates(lists):
  """Counts the number of duplicate subnetworks"""
  seen = set()
  numseen = 0
  for sublist in lists:
    # Convert sublist to a frozenset to make it hashable
    sublist_set = frozenset(sublist)
    if sublist_set in seen:
      numseen += 1
    seen.add(sublist_set)
  return numseen

def count_subnetworks(subnetworks):
    """counts number of subnetworks"""
    num = 0
    for sub in subnetworks:
      num += 1
    return num
