import numpy as np
import networkx as nx
from networkx.algorithms import community
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import argparse 
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
args = parser.parse_args()

print (args.dataset)
adj, features, labels, n_classes = read_dataset(args.dataset)
G = nx.Graph()

# 遍歷稀疏矩陣的表示方式的變數，將邊添加到圖中
rows, cols = adj.nonzero()
for source, target in zip(rows, cols):
    weight = adj[source, target]  # 將權重轉換為字串
    G.add_edge(str(source), str(target), weight=weight)  # 將節點編號轉換為字串
G = nx.relabel_nodes(G, {node: i for i, node in enumerate(G.nodes())})


# 打印圖的信息
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
nx.write_gml(G, "../../real_data/preprocessed_data/cora_graph.gml")

degrees = dict(G.degree())

#每個新增的AI節點有k%機率與其他人(包含真人和AI)相連
num_new_nodes = int(G.number_of_nodes() * 0.1)  # 修改為您想要添加的新節點數量
print (num_new_nodes)
prob_connect_human = 0.0005  # 10%的機率與其他真人節點相連
prob_connect_ai = 0.0005  # 10%的機率與其他AI節點相連

for i in range(1, 11):  # 生成十次
    new_nodes = range(G.number_of_nodes(), G.number_of_nodes() + num_new_nodes)
    aug_G = G.copy()
    aug_G.add_nodes_from(new_nodes)

    for new_node in new_nodes:
        for existing_node in aug_G.nodes():
            # 新增的AI節點與其他節點(包含真人和AI)相連
            if np.random.rand() < prob_connect_human:
                aug_G.add_edge(new_node, existing_node)

    # 生成檔案
    filename = f"../../real_data/preprocessed_data/cora/aug_ai_nodes/k%/k%_graph_{i}.gml"
    print(f"\nGenerating {filename}...")
    print("Number of nodes:", aug_G.number_of_nodes())
    print("Number of edges:", aug_G.number_of_edges())
    print ("Average Degree:", aug_G.number_of_edges()*2/aug_G.number_of_nodes())
    nx.write_gml(aug_G, filename)



#每個新增的AI節點與其他真人相連的機率跟該真人degree數成反比 (exponential decay)
def exponential_decay_weight(d, initial_weight, decay_rate):
    prob = initial_weight * np.exp(-decay_rate * d)
    return prob




initial_weight = 0.0005
decay_rate = 0.05

for i in range(1, 11):  # 生成十次
    new_nodes = range(G.number_of_nodes(), G.number_of_nodes() + num_new_nodes)
    aug_G = G.copy()
    aug_G.add_nodes_from(new_nodes)
    # AI與真人節點相連    
    for new_node in new_nodes:
        for existing_node in G.nodes():
            prob_connect = exponential_decay_weight(degrees[existing_node]-1, initial_weight, decay_rate)
            if np.random.rand() < prob_connect:
                aug_G.add_edge(new_node, existing_node, weight=1)

    degrees = dict(aug_G.degree())

    # AI與AI節點相連
    for existing_new_node in new_nodes:
        for existing_anothor_new_node in new_nodes:
            prob_connect = exponential_decay_weight(degrees[existing_anothor_new_node]-1, initial_weight, decay_rate)
            if np.random.rand() < prob_connect:
                aug_G.add_edge(existing_anothor_new_node, existing_new_node, weight=1)
    filename = f"../../real_data/preprocessed_data/cora/aug_ai_nodes/exp_decay/expdecay_graph_{i}.gml"
    print(f"\nGenerating {filename}...")
    nx.write_gml(aug_G, filename)

    print("\nNumber of nodes after adding:", aug_G.number_of_nodes())
    print("Number of edges after adding:", aug_G.number_of_edges())
    average_degree = sum(degrees.values()) / len(degrees)
    print("Average Degree:", average_degree)
