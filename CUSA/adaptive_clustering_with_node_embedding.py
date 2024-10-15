import os
import sys
import random
import numpy as np
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community.quality import modularity
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import LocalOutlierFactor
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from scipy.linalg import eig
from networkx.algorithms.community.quality import is_partition
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def convert_to_partition_list(cluster_result):
    max_cluster_id = np.max(cluster_result)
    partition_sets = [frozenset(str(i) for i in range(len(cluster_result)) if cluster_result[i] == j) for j in range(0, max_cluster_id + 1)]
    return list(partition_sets)

def print_degree_ranges_count(degrees):
    degree_ranges_count = {f"{i*10}-{(i+1)*10-1}": 0 for i in range((max(degrees.values()) // 10) + 1)}
    for degree in degrees.values():
        degree_range = (degree // 10) * 10
        degree_ranges_count[f"{degree_range}-{degree_range + 9}"] += 1
    for degree_range, count in degree_ranges_count.items():
        print(f"{degree_range}: {count} nodes")

def generate_node2vec_embeddings(G):
    node2vec = Node2Vec(G, dimensions=16, walk_length=20, num_walks=30, workers=4, seed=42)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_embeddings = {node: model.wv[node] for node in G.nodes()}
    return node_embeddings


def remove_one_outlier_lof(node_embeddings, G, num_outliers):
    X = np.array(list(node_embeddings.values()))
    lof = LocalOutlierFactor(n_neighbors=min(20, len(X)-1), contamination=num_outliers)
    lof_scores = -lof.fit_predict(X)  # 获取 LOF 分数

    # 将每个节点的 LOF 分数与节点关联
    node_lof_scores = {node: lof_scores[i] for i, node in enumerate(node_embeddings.keys())}

    # 找到具有最高 LOF 分数的节点
    outlier_node = max(node_lof_scores, key=node_lof_scores.get)

    print("Most likely outlier:", outlier_node)
    
    G.remove_node(outlier_node)
    mapping = {node: str(i) for i, node in enumerate(G.nodes)}
    G = nx.relabel.relabel_nodes(G, mapping)

    return G


def remove_nodes_and_evaluate(G, sc, num_ai_nodes):
    random.seed(42)
    modularity_scores = []
    for i in range(1, num_ai_nodes):
        k = num_ai_nodes - i + 1
        selected_nodes = list(G.nodes())[-k:]
        node2vec_embeddings = generate_node2vec_embeddings(G)
        selected_embeddings = {node: node2vec_embeddings[node] for node in selected_nodes}
        G = remove_one_outlier_lof(selected_embeddings, G, num_outliers=0.05)

        mapping = {node: str(i) for i, node in enumerate(G.nodes)}
        G = nx.relabel.relabel_nodes(G, mapping)

        adj_matrix = nx.to_numpy_array(G)
        partition = sc.fit_predict(adj_matrix)
        partition_list = convert_to_partition_list(partition)
        #modularity = community.modularity(G, partition_list)
        modularity = calculate_weighted_modularity(G, partition_list)
        modularity_scores.append(modularity)
        print(f"刪第{i}顆:", modularity)
    
    return G, modularity_scores

def calculate_weighted_modularity(G, partition_list):
    total_modularity = 0

    for partition in partition_list:
        # 計算分群內的實際邊數
        intra_edges = G.subgraph(partition).number_of_edges()

        # 計算分群內節點的度數總和
        intra_degree_sum = sum(G.degree(node) for node in partition)

        # 計算分群內的模塊性
        modularity = intra_edges / (len(G.edges())) - (intra_degree_sum / (2 * len(G.edges())))**2

        # 計算分群中真實節點的比例
        total_nodes_in_partition = len(partition)
        real_nodes_count = sum(1 for node in partition if int(node) < 247)
        real_nodes_proportion = real_nodes_count / total_nodes_in_partition
        # 將模塊性和真實節點的比例相乘，得到加權模塊性
        weighted_modularity = real_nodes_proportion * modularity

        # 將加權模塊性加到總模塊性中
        total_modularity += weighted_modularity

    return total_modularity

def modify_edge_weights(G):
    for node in G.nodes():
        for adj_node in G[node]:
            if int(node) < 247 and int(adj_node) < 247:
                # 真人相連
                G[node][adj_node]['weight'] = 3
            elif int(node) >= 247 and int(adj_node) >= 247:
                # AI 相連
                G[node][adj_node]['weight'] = 1
            else:
                # AI 連到真人
                G[node][adj_node]['weight'] = 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 adaptive_clustering.py --input_graph <file_path>")
        sys.exit(1)

    input_file = sys.argv[2]
    base_name = os.path.basename(input_file)
    num_ai_nodes = int(base_name[:2])
    print (num_ai_nodes)
    G = nx.read_gml(input_file)
    t_G = G.copy()
    degrees = dict(G.degree())

    print_degree_ranges_count(degrees)

    n_clusters = 20
    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    adj_matrix = nx.to_numpy_array(G)
    partition = sc.fit_predict(adj_matrix)
    partition_list = convert_to_partition_list(partition)

    modularity = calculate_weighted_modularity(G, partition_list)
    print("全沒刪時 Modularity:", modularity)
    G, modularity_scores = remove_nodes_and_evaluate(G, sc, num_ai_nodes)
    modularity_scores.insert(0, modularity)
    max_index = np.argmax(modularity_scores)
    print ("全刪時 Modularity:", modularity_scores[-1])
    print(f"The maximum modularity score is {modularity_scores[max_index]}")
    print("max_index:", max_index)
