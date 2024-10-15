import os
import sys
import networkx as nx

def print_degree_ranges_count(degrees):
    degree_ranges_count = {f"{i*10}-{(i+1)*10-1}": 0 for i in range((max(degrees.values()) // 10) + 1)}
    for degree in degrees.values():
        degree_range = (degree // 10) * 10
        degree_ranges_count[f"{degree_range}-{degree_range + 9}"] += 1
    for degree_range, count in degree_ranges_count.items():
        print(f"{degree_range}: {count} nodes")

def connect_nodes_based_on_jaccard(G):
    for i in range(10):
        print (i)
        jaccard_coefficients = {}
        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2 and not G.has_edge(node1, node2):
                    neighbors1 = set(G.neighbors(node1))
                    neighbors2 = set(G.neighbors(node2))
                    if len(neighbors1.union(neighbors2)) == 0:
                        jaccard_coefficient = 0
                    else:
                        jaccard_coefficient = len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2))
                    jaccard_coefficients[(node1, node2)] = jaccard_coefficient

        sorted_coefficients = sorted(jaccard_coefficients.items(), key=lambda x: x[1], reverse=True)
        added_edges = 0
        for (node1, node2), coefficient in sorted_coefficients:
            if added_edges >= 40:
                break
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2)
                added_edges += 1
if __name__ == "__main__":

    for i in range(1, 11):

        input_file = f"../../real_data/preprocessed_data/cora/aug_ai_nodes/exp_decay/expdecay_graph_{i}.gml"
        G = nx.read_gml(input_file)
        print (i)
        print (G.number_of_nodes())
        print (G.number_of_edges())
        aug_G = G.copy()
        connect_nodes_based_on_jaccard(aug_G)
        print (aug_G.number_of_nodes())
        print (aug_G.number_of_edges())
        filename = f"../../real_data/preprocessed_data/cora/evolve/exp_decay/graph_{i}.gml"
        print(f"\nGenerating {filename}...")
        nx.write_gml(aug_G, filename)