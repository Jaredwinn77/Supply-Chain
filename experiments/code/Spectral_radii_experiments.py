import random
from Skeleton import create_graph, visualize_graph, draw_subnetwork, visualize_subnetworks, find_subnetworks_color, create_weights,spectral_radius, breadth_first
""" This code runs experiments to analyze how starting with different nodes affects 
the spectral radii of each subnetwork

create_graph(num_nodes,num_edges,num_colors) ---> nx.Digraph object
visualize_graph(G)---> shows a plot of the graph
draw_subnetwork(G, subnetwork_nodes, title) ---> draws an individual subnetwork, acts as a helper function for visualize subnetworks
visualize_subnetworks(G, subnetworks) ---> shows the graph of each subnetwork in the list subnetworks
find_subnetworks_color(G,start,visited=None,current_subnetwork=None) ---> returns a list of subnetworks 
create_weights(G,num_nodes) ---> returns a weight matrix
spectral_radius(weighted, subnetworks) ---> returns a list of the spectral radii of each subnetwork


"""






def main():
    num_graphs = 10
    num_colors = 3
    different_number_of_subnetworks = []
    different_futures = []
    for j in range(num_graphs):
        num_nodes = 20 #random.randint(3,10)
        num_edges = 30 #num_nodes*random.randint(1,3)
        graph = create_graph(num_nodes,num_edges,num_colors)
        weights = create_weights(graph,num_nodes)
        length = []
        radii = []
        for i in range(num_nodes):
            subnetworks = breadth_first(graph,i)
            radius = spectral_radius(weights,subnetworks)
            length.append(len(subnetworks))
            radii.append(radius)


        if not all(x == length[0] for x in length):
            different_number_of_subnetworks.append(graph)
        else:
            masked = [[1 if value >= 1 else 0 for value in sublist] for sublist in radii]
            for index in zip(*masked):
                if len(set(index)) != 1:
                    different_futures.append(graph)
    print(f"different subnetworks {len(different_number_of_subnetworks)}")
    #for graph in different_number_of_subnetworks:
        #visualize_graph(graph)
    print(f"different futures {len(different_futures)}")
    #for graph in different_futures:
        #visualize_graph(graph)





if __name__=="__main__":
    main()
