from network import SupplyChainNetwork
import networkx as nx
import numpy as np
import sys


def diversify(p):
    improvement_matrix = np.empty((19,))
    lambda_vals = np.linspace(0.05, 0.95, 19)

    for j, lam in enumerate(lambda_vals):
        original_ratios = []
        diversified_ratios = []
        for _ in range(5000):

            sc = SupplyChainNetwork.generate(num_nodes=25, num_edges=40, num_colors=4)
            subnetwork_radii = [max(abs(np.linalg.eigvals(nx.adjacency_matrix(G).toarray()))) for G in sc.subnetworks]
            ratio = sum(1 for radius in subnetwork_radii if radius >= 1) / len(subnetwork_radii)
            original_ratios.append(ratio)

            sc.diversify(p, lam)
            subnetwork_radii = [max(abs(np.linalg.eigvals(nx.adjacency_matrix(G).toarray()))) for G in sc.subnetworks]
            ratio = sum(1 for radius in subnetwork_radii if radius >= 1) / len(subnetwork_radii)
            diversified_ratios.append(ratio)

        improvement_ratio = np.mean(diversified_ratios) - np.mean(original_ratios)
        improvement_matrix[j] = improvement_ratio

    np.save(f"improvement_matrix_{p}.npy", improvement_matrix)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        p = sys.argv[1]
        diversify(p)
