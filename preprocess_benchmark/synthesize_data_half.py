import numpy as np
import networkx as nx
from networkx.algorithms import community
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import random

file_path = "football/football-follows.mtx"
data = np.loadtxt(file_path, skiprows=1)
G = nx.Graph()

for row in data:
    node1, node2, weight = map(int, row)
    G.add_edge(node1, node2, weight=weight)


print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

degrees = dict(G.degree())

# 計算平均Degree
average_degree = sum(degrees.values()) / len(degrees)
print("Average Degree:", average_degree)

def read_communities_file(filename):
    network_data = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            community_name = parts[0]
            member_ids = [int(id) for id in parts[1].split(',')]
            network_data[community_name] = member_ids
    return network_data

filename = "football/football.communities"  
communities = read_communities_file(filename)
communities['west-brom'].remove(240002233) #因為他本來就沒朋友

num_ai_nodes = 60
half_num_ai_nodes = num_ai_nodes // 2
ai_nodes = range(G.number_of_nodes(), G.number_of_nodes() + num_ai_nodes)


for i in range(1, 11):  # 生成十次
    aug_G = G.copy()
    aug_G.add_nodes_from(ai_nodes)
    ai_nodes_inner = list(ai_nodes[:half_num_ai_nodes])
    ai_nodes_assigned = ai_nodes_inner.copy()
    for community_name, member_ids in communities.items():
        num_ai_nodes_for_community = random.randint(1, 2)
        ai_nodes_for_community = random.sample(ai_nodes_assigned, min(num_ai_nodes_for_community, len(ai_nodes_assigned)))
        for member in member_ids:
            for ai_node in ai_nodes_for_community:
                # 對於每個社群中的AI點，以50%的概率與該社群中的每個成員相連
                if random.random() < 0.5:
                    aug_G.add_edge(member, ai_node)
        # 從已分配的AI節點中移除已經分配給這個社群的節點
        for node in ai_nodes_for_community:
            ai_nodes_assigned.remove(node)

    remaining_ai_nodes = list(set(ai_nodes) - set(ai_nodes_inner))
    # 對於自由的AI點，以1/len(community)的概率與每個社群中的每個成員相連

    for ai_node in remaining_ai_nodes:
        for community_name, member_ids in communities.items():
            for member in member_ids: 
                if random.random() < 1 / len(member_ids):
                    aug_G.add_edge(member, ai_node)

    for existing_new_node in remaining_ai_nodes:
        for existing_anothor_new_node in remaining_ai_nodes:
            if np.random.rand() < 0.1:
                aug_G.add_edge(existing_anothor_new_node, existing_new_node)


    print("Number of nodes after adding AI nodes:", aug_G.number_of_nodes())
    print("Number of edges after adding AI nodes:", aug_G.number_of_edges())
    print("Average Degree:", aug_G.number_of_edges()*2/aug_G.number_of_nodes())
    filename = f"data/half/60nodes_half_aug_football_graph_{i}.gml"
    #filename = "60nodes_half_aug_football_graph.gml"
    print(f"\nGenerating {filename}...")
    aug_G = nx.convert_node_labels_to_integers(aug_G, first_label=0, ordering='default')
    nx.write_gml(aug_G, filename)