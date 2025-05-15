from Graphclass_module import Graphclass
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# shape of each subnetwork is [(node1, node2, colorNumber), (node1, node2, colorNumber)] list of lists of 3 tuple

def kill_node(subnetwork, dead_node):
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

def main(graph): # input a graph 
    for subnetwork in graph.subnetworks: # each graph has multiple subnetworks, loop over each
        all_nodes = set(n for edge in subnetwork for n in edge[:2])
        for node in all_nodes: # loop over each edge in the subnetwork
            failed, alive = kill_node(subnetwork, node)
            if failed == alive:
                print("Total Failure")
            else:
                print(f"Starting at node: {node} in subnetwork: {subnetwork}, these die: {failed}")
            
