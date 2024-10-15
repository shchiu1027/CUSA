#!/bin/bash

# 初始化總和變數
total_modularity_delete_all_ai_nodes=0
total_modularity_all_nodes=0
total_max_modularity=0
total_max_index=0

# 迴圈執行 adaptive_clustering.py
for i in {1..10}; do

    input_file="data/evolve/half/60nodes_half_aug_football_graph_${i}_evolved.gml"
    echo "Running clustering with input file: $input_file"

    # 執行 adaptive_clustering.py，並提取輸出中的 Modularity 和 Max Modularity 和 Max Index
    output=$(python3 adaptive_clustering.py --input_graph $input_file)
    modularity_all_nodes=$(echo "$output" | awk '/全沒刪時 Modularity:/ {print $NF}')
    modularity_delete_all_ai_nodes=$(echo "$output" | awk '/全刪時 Modularity:/ {print $NF}')
    max_modularity=$(echo "$output" | awk '/The maximum modularity score is/ {print $NF}')
    max_index=$(echo "$output" | awk '/max_index:/ {print $NF}')

    echo "全沒刪時Modularity for run $i: $modularity_all_nodes"
    echo "全刪時Modularity for run $i: $modularity_delete_all_ai_nodes"
    echo "Max modularity for run $i: $max_modularity"
    echo "Max index for run $i: $max_index"

    # 累加 Modularity、Max Modularity 和 Max Index 值
    total_modularity_delete_all_ai_nodes=$(awk "BEGIN {print $total_modularity_delete_all_ai_nodes + $modularity_delete_all_ai_nodes}")
    total_modularity_all_nodes=$(awk "BEGIN {print $total_modularity_all_nodes + $modularity_all_nodes}")
    total_max_modularity=$(awk "BEGIN {print $total_max_modularity + $max_modularity}")
    total_max_index=$(awk "BEGIN {print $total_max_index + $max_index}")
done

# 計算平均值

average_modularity_delete_all_ai_nodes=$(awk "BEGIN {print $total_modularity_delete_all_ai_nodes/$i}")
average_modularity_all_nodes=$(awk "BEGIN {print $total_modularity_all_nodes/$i}")
average_max_modularity=$(awk "BEGIN {print $total_max_modularity/$i}")
average_max_index=$(awk "BEGIN {print $total_max_index/$i}")

# 輸出平均 Modularity、Max Modularity 和 Max Index
echo "Average Modularity for all nodes: $average_modularity_all_nodes"
echo "Average Modularity for deleting all AI nodes: $average_modularity_delete_all_ai_nodes"
echo "Average Max Modularity: $average_max_modularity"
echo "Average Max Index: $average_max_index"
