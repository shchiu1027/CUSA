import os
import sys
import random
import numpy as np
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community.quality import modularity
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from scipy.linalg import eig
from networkx.algorithms.community.quality import is_partition
import warnings
import argparse
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import member_change
from collections import defaultdict
import community

# 將分群結果轉換為list，每個元素是一個frozenset，代表對應的分群
def convert_to_partition_list(cluster_result):
    max_cluster_id = np.max(cluster_result)
    partition_sets = [set(str(i) for i in range(len(cluster_result)) if cluster_result[i] == j) for j in range(0, max_cluster_id + 1)]
    return list(partition_sets)

def handle_index(partition, removed_nodes):
    removed_nodes = sorted([int(item) for item in removed_nodes])
    for node in removed_nodes:
        partition = np.insert(partition, int(node), -1)
    return partition

def remove_all_ai_nodes(partition_list, G, num_ai_nodes):
    # 創建一個新的分群列表，其中不包含編號在 G.number_of_nodes() 到 G.number_of_nodes()+num_ai_nodes之間的節點
    new_partition_list = []
    for cluster in partition_list:
        new_cluster = {node for node in cluster if not (str(G.number_of_nodes()) <= node <= str(G.number_of_nodes()+num_ai_nodes))}
        if new_cluster:
            new_partition_list.append(new_cluster)
    return new_partition_list



def my_modularity(G, partition_list, num_ai_nodes):
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
        real_nodes_count = sum(1 for node in partition if int(node) < G.number_of_nodes() - num_ai_nodes)
        real_nodes_proportion = real_nodes_count / total_nodes_in_partition
        # 將模塊性和真實節點的比例相乘，得到加權模塊性
        weighted_modularity = real_nodes_proportion * modularity
        # 將加權模塊性加到總模塊性中
        total_modularity += weighted_modularity

    return total_modularity

def perform_sc(G, sc, n_clusters=20):
    # 使用Spectral Clustering進行分群
    adj_matrix = nx.to_numpy_array(G)
    partition = sc.fit_predict(adj_matrix)

    return partition

def perform_louvain(G):
    partition = community.best_partition(G)
    partition = list(partition.values())
    return partition

def remove_nodes_and_evaluate(G, sc, num_ai_nodes, n_clusters, scoring_of_AI):
    modularity_scores = []

    #partition = perform_sc(G, sc, n_clusters)
    #partition_list = convert_to_partition_list(partition)

    partition = perform_louvain(G)
    partition_list = convert_to_partition_list(partition)

    modularity = my_modularity(G, partition_list, num_ai_nodes)
    print("全沒刪時:", f"{modularity:.4f}")

    initial_modularity = modularity
    optimal_modularity = modularity
    modularity_scores = [modularity]
    best_partition_list = partition_list

    removed_nodes = []
    for i in range(1, num_ai_nodes + 1):
        k = num_ai_nodes - i + 1
        selected_nodes = list(G.nodes())[-k:]

        #用CC來決定
        if scoring_of_AI.upper() == 'CC':
            clustering_coefficients = nx.clustering(G)
            sorted_clustering_coefficients = sorted(selected_nodes, key=lambda x: clustering_coefficients[x], reverse=True)
            G.remove_node(sorted_clustering_coefficients[0])
            removed_nodes.append(sorted_clustering_coefficients[0])      

        #用EC來決定
        if scoring_of_AI.upper() == 'EC':
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            sorted_nodes_by_eigenvector = sorted(selected_nodes, key=lambda x: eigenvector_centrality[x], reverse=True)
            G.remove_node(sorted_nodes_by_eigenvector[-1])
            removed_nodes.append(sorted_nodes_by_eigenvector[-1]) 


        # 用快速近似的BC來決定
        if scoring_of_AI.upper() == 'BC':
            betweenness_centrality = nx.betweenness_centrality(G, endpoints=False, k=10, seed=None)
            sorted_nodes_by_betweenness = sorted(selected_nodes, key=lambda x: betweenness_centrality[x], reverse=True)
            G.remove_node(sorted_nodes_by_betweenness[-1])
            removed_nodes.append(sorted_nodes_by_betweenness[-1])

        # 用 EC+BC 來決定
        if scoring_of_AI.upper() == 'ECBC':
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            betweenness_centrality = nx.betweenness_centrality(G, endpoints=False, k=10, seed=None)

            max_ec = max(eigenvector_centrality.values())
            max_bc = max(betweenness_centrality.values())
            min_ec = min(eigenvector_centrality.values())
            min_bc = min(betweenness_centrality.values())
            normalized_ec = {node: (ec - min_ec) / (max_ec - min_ec) for node, ec in eigenvector_centrality.items()}
            normalized_bc = {node: (bc - min_bc) / (max_bc - min_bc) for node, bc in betweenness_centrality.items()}

            combined_centrality = {node: normalized_ec[node] + normalized_bc[node] for node in selected_nodes}
            min_combined_centrality_node = min(combined_centrality, key=combined_centrality.get)

            G.remove_node(min_combined_centrality_node)
            removed_nodes.append(min_combined_centrality_node)

        # 用BC+CC來決定
        if scoring_of_AI.upper() == 'BCCC':
            betweenness_centrality = nx.betweenness_centrality(G, endpoints=False, k=10, seed=None)
            clustering_coefficients = nx.clustering(G)

            reciprocal_cc = {node: 1 / cc if cc != 0 else 0 for node, cc in clustering_coefficients.items()}
            
            max_bc = max(betweenness_centrality.values())
            max_reciprocal_cc = max(reciprocal_cc.values())
            min_bc = min(betweenness_centrality.values())
            min_reciprocal_cc = min(reciprocal_cc.values())
            normalized_bc = {node: (bc - min_bc) / (max_bc - min_bc) for node, bc in betweenness_centrality.items()}
            normalized_reciprocal_cc = {node: (rc - min_reciprocal_cc) / (max_reciprocal_cc - min_reciprocal_cc) for node, rc in reciprocal_cc.items()}

            combined_value = {node: normalized_bc[node] + normalized_reciprocal_cc[node] for node in selected_nodes}
            min_combined_value_node = min(combined_value, key=combined_value.get)

            G.remove_node(min_combined_value_node)
            removed_nodes.append(min_combined_value_node)


        # 用EC+CC來決定
        if scoring_of_AI.upper() == 'ECCC':
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            clustering_coefficients = nx.clustering(G)

            reciprocal_cc = {node: 1 / cc if cc != 0 else 0 for node, cc in clustering_coefficients.items()}
            
            max_ec = max(eigenvector_centrality.values())
            max_reciprocal_cc = max(reciprocal_cc.values())
            min_ec = min(eigenvector_centrality.values())
            min_reciprocal_cc = min(reciprocal_cc.values())
            normalized_ec = {node: (ec - min_ec) / (max_ec - min_ec) for node, ec in eigenvector_centrality.items()}
            normalized_reciprocal_cc = {node: (rc - min_reciprocal_cc) / (max_reciprocal_cc - min_reciprocal_cc) for node, rc in reciprocal_cc.items()}

            combined_value = {node: normalized_ec[node] + normalized_reciprocal_cc[node] for node in selected_nodes}
            min_combined_value_node = min(combined_value, key=combined_value.get)

            G.remove_node(min_combined_value_node)
            removed_nodes.append(min_combined_value_node)

        # 用 EC+BC+CC 來決定
        if scoring_of_AI.upper() == 'ECBCCC':
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            betweenness_centrality = nx.betweenness_centrality(G, endpoints=False, k=10, seed=None)
            clustering_coefficients = nx.clustering(G)

            # 將 clustering coefficient 為零的節點倒數值設為 0
            reciprocal_cc = {node: 1 / cc if cc != 0 else 0 for node, cc in clustering_coefficients.items()}

            # 最小-最大歸一化 EC 和 BC 的值
            max_ec = max(eigenvector_centrality.values())
            max_bc = max(betweenness_centrality.values())
            max_reciprocal_cc = max(reciprocal_cc.values())
            min_ec = min(eigenvector_centrality.values())
            min_bc = min(betweenness_centrality.values())
            min_reciprocal_cc = min(reciprocal_cc.values())

            normalized_ec = {node: (ec - min_ec) / (max_ec - min_ec) for node, ec in eigenvector_centrality.items()}
            normalized_bc = {node: (bc - min_bc) / (max_bc - min_bc) for node, bc in betweenness_centrality.items()}
            normalized_reciprocal_cc = {node: (rc - min_reciprocal_cc) / (max_reciprocal_cc - min_reciprocal_cc) for node, rc in reciprocal_cc.items()}

            # 將三者相加並找出最小值
            combined_value = {node: normalized_ec[node] + normalized_bc[node] + normalized_reciprocal_cc[node] for node in selected_nodes}
            min_combined_value_node = min(combined_value, key=combined_value.get)

            # 移除具有最小組合值的節點
            G.remove_node(min_combined_value_node)
            removed_nodes.append(min_combined_value_node)


        if scoring_of_AI.upper() == 'RANDOM':
            node_to_remove = random.choice(selected_nodes)
            G.remove_node(node_to_remove)
            removed_nodes.append(node_to_remove)

        
        #partition = perform_sc(G, sc, n_clusters)
        partition = perform_louvain(G)
        partition = handle_index(partition, removed_nodes)
        partition_list = convert_to_partition_list(partition)
        modularity = my_modularity(G, partition_list, num_ai_nodes)

        if optimal_modularity < modularity:
            optimal_modularity = modularity
            best_partition_list = partition_list

        modularity_scores.append(modularity)
        print(f"刪第{i}顆:", f"{modularity:.4f}", len(partition_list))

        

    return G, modularity_scores, best_partition_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_graph', type=str, default='../../real_data/preprocessed_data/cora/cora_graph.gml', help='type of graph.')
    parser.add_argument('--input_graph', type=str, default='../../real_data/preprocessed_data/cora/evolve/k%/graph_1.gml', help='type of graph.')
    parser.add_argument('--method', type=str, default='CC', help='type of ai_scoring_method.')
    args = parser.parse_args()

    ori_G = nx.read_gml(args.ori_graph)
    print (ori_G, type(ori_G))
    num_ai_nodes = int(ori_G.number_of_nodes() * 0.1)
    print (num_ai_nodes)
    G = nx.read_gml(args.input_graph)
    scoring_of_AI = args.method
    print (G, type(G))
    all_G = G.copy()

    n_clusters = 20
    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    G, modularity_scores, best_partition_list = remove_nodes_and_evaluate(G, sc, num_ai_nodes, n_clusters, scoring_of_AI)
    max_index = np.argmax(modularity_scores)
    print ()
    print("全刪時 Modularity:", f"{modularity_scores[-1]:.4f}")
    print('-------')
    print(f"The maximum modularity score is {modularity_scores[max_index]:.4f}")
    print("max_index:", max_index)

    #計算跑群率，現在的'G'已經為全刪：
    #partition = perform_sc(G, sc, n_clusters)
    partition = perform_louvain(G)
    ori_partition_list = convert_to_partition_list(partition)

    new_partition_list = remove_all_ai_nodes(best_partition_list, G, num_ai_nodes)

    aligned_new_partition_list = member_change.align_clusters(ori_partition_list, new_partition_list)

    moves = member_change.compare_clusters(ori_partition_list, aligned_new_partition_list)
    print("節點移動的次數：", moves)
    migration_ratio_percentage = (moves / G.number_of_nodes()) * 100
    print(f"human migration ratio (HMR): {migration_ratio_percentage:.2f}%")
    print(f"Average AI-driven migration (AAM): {moves / num_ai_nodes:.2f}")


    print('-------')
    print("全不刪時 Modularity:", f"{modularity_scores[0]:.4f}")
    #partition = perform_clustering(all_G, sc, n_clusters)
    partition = perform_louvain(all_G)
    new_partition_list = convert_to_partition_list(partition)
    new_partition_list = remove_all_ai_nodes(new_partition_list, G, num_ai_nodes)
    aligned_new_partition_list = member_change.align_clusters(ori_partition_list, new_partition_list)
    moves = member_change.compare_clusters(ori_partition_list, aligned_new_partition_list)
    print("節點移動的次數：", moves)
    migration_ratio_percentage = (moves / G.number_of_nodes()) * 100
    print(f"human migration ratio (HMR): {migration_ratio_percentage:.2f}%")
    print(f"Average AI-driven migration (AAM): {moves / num_ai_nodes:.2f}")
    
