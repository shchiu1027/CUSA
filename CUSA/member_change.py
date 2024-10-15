from sklearn.cluster import SpectralClustering
from networkx.algorithms import community
import networkx as nx
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
'''
def align_clusters(ori_partition_list, new_partition_list):
    aligned_new_partition_list = []
    for ori_cluster in ori_partition_list:
        max_overlap_len = -1
        max_set = {}
        for new_cluster in new_partition_list:
            if len(ori_cluster.intersection(new_cluster)) > max_overlap_len:
                max_overlap_len = len(ori_cluster.intersection(new_cluster))
                max_set = new_cluster
        aligned_new_partition_list.append(max_set)
    return aligned_new_partition_list

'''
def intersection_matrix(cluster1, cluster2):
    matrix = []
    for group1 in cluster1:
        row = []
        for group2 in cluster2:
            intersection_size = len(group1.intersection(group2))
            row.append(intersection_size)
        matrix.append(row)
    return np.array(matrix)

def align_clusters(cluster1, cluster2):
    aligned_cluster2_list = [{}] * len(cluster1)
    intersection_mat = intersection_matrix(cluster1, cluster2)
    for row_i, row in enumerate(intersection_mat):
        sorted_indices = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
        for max_ind in sorted_indices:
            if cluster2[max_ind] not in aligned_cluster2_list:
                aligned_cluster2_list[row_i] = cluster2[max_ind]
                break
    return aligned_cluster2_list



def compare_clusters(cluster1, cluster2):
    # 創建字典來存儲每個節點的分群結果
    node_mapping = {}
    
    # 記錄節點移動的次數
    moves = 0
    
    # 遍歷第一種分群結果，將節點映射到其所在的分群
    for group, nodes in enumerate(cluster1):
        for node in nodes:
            node_mapping[node] = group
    
    # 遍歷第二種分群結果，比較與第一種分群的差異
    for group, nodes in enumerate(cluster2):
        for node in nodes:
            if node in node_mapping:
                # 如果節點在兩種分群中都存在，檢查其分群是否有變化
                if node_mapping[node] != group:
                    moves += 1
            else:
                # 如果節點只在第二種分群中出現，則視為移動
                moves += 1
    
    return moves


def convert_to_partition_list(cluster_result):
    max_cluster_id = np.max(cluster_result)
    partition_sets = [set(str(i) for i in range(len(cluster_result)) if cluster_result[i] == j) for j in range(0, max_cluster_id + 1)]
    return list(partition_sets)


def perform_clustering(G, n_clusters=20):
    # 使用SpectralClustering進行分群
    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    adj_matrix = nx.to_numpy_array(G)
    partition = sc.fit_predict(adj_matrix)

    return partition

def official_modularity(G, partition_list):

    modularity = community.modularity(G, partition_list)

    return modularity

def my_modularity(G, partition_list):
    total_modularity = 0

    for partition in partition_list:
        intra_edges = G.subgraph(partition).number_of_edges()
        intra_degree_sum = sum(G.degree(node, weight='weight') for node in partition)
        modularity = intra_edges / (len(G.edges())) - (intra_degree_sum / (2 * len(G.edges()))) ** 2
        total_modularity += modularity

    return total_modularity



if __name__ == "__main__":

    cluster1 = [{1, 2, 3}, {4, 5, 6, 7, 8, 9, 10}, {11,12}, {13,14}]
    cluster2 = [{1, 4, 5}, {3, 6}, {7, 8, 9, 10}, {11,12,13,14}]
    cluster2 = [{1, 4, 5}, {7, 8, 9, 10}, {11,12,13,14}, {3, 6}]
    moves = compare_clusters(cluster1, cluster2)
    print("節點移動的次數：", moves)


    cluster2 = align_clusters(cluster1, cluster2)
    print (cluster2)
    moves = compare_clusters(cluster1, cluster2)
    print("節點移動的次數：", moves)

    '''
    input_file = 'ori_football_graph.gml'
    G = nx.read_gml(input_file)

    partition = perform_clustering(G)
    partition_list = convert_to_partition_list(partition)
    modularity = my_modularity(G, partition_list)
    print (modularity)
    modularity = official_modularity(G, partition_list)
    print (modularity)

    G.remove_node('2')
    partition = perform_clustering(G)
    partition = np.insert(partition, 2, -1)
    partition_list = convert_to_partition_list(partition)
    modularity = my_modularity(G, partition_list)
    print (modularity)
    modularity = official_modularity(G, partition_list)
    print (modularity)
    '''