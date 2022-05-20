import utils
import nodeProps
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import collections
import heapq
import random
import queue
import time

# folder containing the work files
io_folder_path = utils.io_folder_path
network_app = utils.network_app
in1 = io_folder_path + network_app + \
    '_src_sink_low.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step17_low.json'
# 'part_8_1799_src_sink_nodes_levels.txt'
in3 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'
# 'rev_part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_' + network_app + '_src_sink_nodes_levels_low.txt'
in4_b = io_folder_path + 'rev_' + network_app + '_src_sink_low.dot'
in5 = io_folder_path + 'tensors_sz_32_low.txt'
in6 = io_folder_path + 'memory.txt'

# output file
out1 = io_folder_path + 'ver_grouper_placement_e_nc.place'

# grouper parameters
no_of_desired_groups = 4
memory_limit_per_group = 31 * 1024 * 1024 * 1024

comm_latency = 45
average_tensor_size_if_not_provided = 1
comm_transfer_rate = 1000000 / (140 * 1024 * 1024 * 1024)

reverse_levels = {}
# get nodes levels
with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            reverse_levels[node_and_level[0]] = node_and_level[1]

tensors_sizes = {}
edges_weights = {}
# get tensors sizes
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensor_size = int(splitted[1])
        tensor_name = splitted[0]
        tensors_sizes[tensor_name] = tensor_size
        edges_weights[tensor_name] = float(tensor_size) * comm_transfer_rate + comm_latency

# getting time (weight) info for nodes
analysis_graph = utils.read_profiling_file(in2, True)

# get_node_average_weiht
total_nodes_weight = 0
for node, node_props in analysis_graph.items():
    total_nodes_weight = total_nodes_weight + node_props.duration

average_node_weight = total_nodes_weight/len(analysis_graph)

# will contain the graph as an adgacency list
graph = {}
all_nodes = {}
sink_node_name = 'snk'
source_node_name = 'src'

# initializing the nodes and adjacencies from the dot file
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            if not splits[0] in all_nodes:
                all_nodes[splits[0]] = nodeProps.NodeProps()
            if not splits[1] in all_nodes:
                all_nodes[splits[1]] = nodeProps.NodeProps()

            all_nodes[splits[1]].parents.append(splits[0])
            all_nodes[splits[0]].children.append(splits[1])

            if splits[0] in graph.keys():
                graph[splits[0]].append(splits[1])
            else:
                graph[splits[0]] = [splits[1]]


for node, node_props in all_nodes.items():
    if node in analysis_graph:
        analysis_graph[node].parents = node_props.parents
        analysis_graph[node].children = node_props.children
    else:
        analysis_graph[node] = node_props


for node in all_nodes:
    if not node in tensors_sizes:
        tensors_sizes[node] = 0
        edges_weights[node] = float(comm_latency)

# constructing the graph and initializing the nodes levels from the dot file
rev_graph = {}
with open(in4_b, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            if nodes[0] in rev_graph:
                rev_graph[nodes[0]].append(nodes[1])
            else:
                rev_graph[nodes[0]] = [nodes[1]]

# get nodes in degrees for the topological sort
nodes_in_degrees = {}
for node in all_nodes:
    if node in rev_graph:
        nodes_in_degrees[node] = len(rev_graph[node])
    else:
        nodes_in_degrees[node] = 0
# get reverse nodes in degrees for the topological sort
rev_nodes_in_degrees = {}
for node in all_nodes:
    if node in graph:
        rev_nodes_in_degrees[node] = len(graph[node])
    else:
        rev_nodes_in_degrees[node] = 0
import time
def get_nodes_weighted_levels(graph, edges_weights, src_nodes = None, previosly_visited = []):
    # getting the sources of the graph to start the topological traversal from them
    graph_keys = {}
    nodes_weighted_levels={}
    tmp_nodes_in_degrees = copy.deepcopy(rev_nodes_in_degrees)
    traversal_queueu = queue.Queue()

    if src_nodes is None:
        for graph_key in graph.keys():
            graph_keys[graph_key] = 0

        for adj_nodes in graph.values():
            for node in adj_nodes:
                if node in graph_keys:
                    graph_keys[node] = 1
        src_nodes = {}
        for node, source_node in graph_keys.items():
            if source_node == 0:
                src_nodes[node] = 1

    for node in src_nodes:
        traversal_queueu.put(node)
    for node in graph.keys():
        nodes_weighted_levels[node] = 0  # analysis_graph[node].duration

    # start the traversal
    while not traversal_queueu.empty():
        current_node = traversal_queueu.get()
        adj_nodes = graph[current_node]
        current_node_level = nodes_weighted_levels[current_node]
        for adj_node in adj_nodes:
            if adj_node not in previosly_visited:
                new_level = current_node_level + edges_weights[adj_node] + analysis_graph[adj_node].duration
                tmp_nodes_in_degrees[adj_node] -= 1
                if nodes_weighted_levels[adj_node] < new_level:
                    nodes_weighted_levels[adj_node] = new_level
                if tmp_nodes_in_degrees[adj_node] == 0:
                    traversal_queueu.put(adj_node)
    return nodes_weighted_levels

levels_weights = {}
no_of_levels = 0
# get nodes levels
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            int_node_level = int(node_and_level[1])
            analysis_graph[node_and_level[0]].level = int_node_level
            if int_node_level in levels_weights.keys():
                levels_weights[int_node_level] = levels_weights[int_node_level] + \
                    analysis_graph[node_and_level[0]].duration
            else:
                levels_weights[int_node_level
                               ] = analysis_graph[node_and_level[0]].duration
                no_of_levels = no_of_levels + 1

# extracting all vertical paths in the graph
graph[sink_node_name] = []
rev_graph[source_node_name] = []
free_nodes = []
paths = []
current_path = []
visited = {}
src_nodes = {}
groups_weights = []
paths_lengths = []
current_path_weight = 0
current_path_weight_with_comm = 0
num_paths = 0
nodes_paths_mapping = {}
nodes_to_visit = list(all_nodes.keys())
tmp_rev_graph = copy.deepcopy(rev_graph)
tmp_nodes_in_degrees = copy.deepcopy(nodes_in_degrees)

nodes_weighted_levels = get_nodes_weighted_levels(rev_graph, edges_weights)
for node, weighted_level in nodes_weighted_levels.items():
    heapq.heappush(free_nodes, (-weighted_level, node))

while free_nodes:
    current_node = heapq.heappop(free_nodes)[1]
    while current_node in visited and free_nodes:
        current_node = heapq.heappop(free_nodes)[1]

    while current_node !='' and current_node not in visited:
        current_path.append(current_node)
        current_path_weight = current_path_weight + \
            analysis_graph[current_node].duration
        current_path_weight_with_comm = current_path_weight_with_comm + \
            analysis_graph[current_node].duration + edges_weights[current_node]
        visited[current_node] = 1
        src_nodes[current_node] = 1
        max_priority = -1
        next_node = ''
        for adj_node in graph[current_node]:
            if adj_node not in visited and nodes_weighted_levels[adj_node] > max_priority:
                max_priority = nodes_weighted_levels[adj_node]
                next_node = adj_node
        current_node = next_node

    if len(current_path) > 0:
        paths.append(current_path)
        groups_weights.append(current_path_weight)
        paths_lengths.append(len(current_path))
        if len(paths) <= no_of_desired_groups or current_path_weight_with_comm >= groups_weights[0] / 10:
            nodes_weighted_levels = get_nodes_weighted_levels(tmp_rev_graph, edges_weights, src_nodes, visited)
            free_nodes = []
            for node, weighted_level in nodes_weighted_levels.items():
                heapq.heappush(free_nodes, (-weighted_level, node))

        for node in current_path:
            del rev_nodes_in_degrees[node]
            for adj_node in graph[node]:
                tmp_nodes_in_degrees[adj_node] -= 1
                if adj_node in visited and tmp_nodes_in_degrees[adj_node] == 0:
                    del tmp_rev_graph[adj_node]
                    del src_nodes[adj_node]

        current_path = []
        current_path_weight = 0
        current_path_weight_with_comm = 0
        num_paths = num_paths + 1

# sort paths from shortest to longest
paths_lengths, groups_weights, paths = (list(t) for t in zip(
    *sorted(zip(paths_lengths, groups_weights, paths))))
print('num of paths: ' + str(len(paths)))
print(paths_lengths[-20:])

# which node is in which path
nodes_paths_mapping[source_node_name] = num_paths - 1
nodes_paths_mapping[sink_node_name] = num_paths - 1
for i in range(0, num_paths):
    for node in paths[i]:
        nodes_paths_mapping[node] = i

# get max potential of paths
groups_parents = {}
paths_max_potential = copy.deepcopy(groups_weights)

for i in range(0, len(paths)):
    current_path = paths[i]
    current_path_len = len(current_path) - 1
    parent_path_indx = -1
    found = False
    heaviest_parent_child_tensor = 0
    heaviest_parent_or_child_path = -1
    if current_path[0] != source_node_name and current_path[current_path_len] != sink_node_name:
        for src_node in analysis_graph[current_path[0]].parents:
            if tensors_sizes[src_node] > heaviest_parent_child_tensor:
                heaviest_parent_child_tensor = tensors_sizes[src_node]
                heaviest_parent_or_child_path = nodes_paths_mapping[src_node]
            for dst_node in analysis_graph[current_path[current_path_len]].children:
                if tensors_sizes[dst_node] > heaviest_parent_child_tensor:
                    heaviest_parent_child_tensor = tensors_sizes[dst_node]
                    heaviest_parent_or_child_path = nodes_paths_mapping[dst_node]
                if nodes_paths_mapping[src_node] == nodes_paths_mapping[dst_node]:
                    parent_path_indx = nodes_paths_mapping[src_node]
                    paths_max_potential[parent_path_indx] = paths_max_potential[parent_path_indx] + \
                        paths_max_potential[i]
                    found = True
                    break
                if found:
                    break
    if parent_path_indx == -1:
        parent_path_indx = heaviest_parent_or_child_path
    groups_parents[i] = parent_path_indx

#map, helpful to find nodes in a level in O(1)
levels_nodes = [None] * no_of_levels
for node, props in analysis_graph.items():
    if node in all_nodes:
        if levels_nodes[props.level] == None:
            levels_nodes[props.level] = []
        levels_nodes[props.level].append(node)

# get the average path length
after_heavy_paths_count = 0
after_heavy_paths_lengths = 0
for path in paths:
    after_heavy_paths_count = after_heavy_paths_count + 1
    after_heavy_paths_lengths = after_heavy_paths_lengths + len(path)

average_path_len = round(after_heavy_paths_lengths / after_heavy_paths_count)

print(average_path_len)

# getting initial groups
initial_groups = copy.deepcopy(paths)
initial_groups_indices = [1] * num_paths
path_joined_group = {}

for i in range(0, num_paths - 1):
    current_group = initial_groups[i]
    current_group_weight = groups_weights[i]
    group_comm_time = 0
    total_branching_potential = 0
    branch_start = ''
    branch_end = ''
    branching_main_path = groups_parents[i]
    sibling_from_branching_main_path = ''
    current_group_siblings_potentials = 0
    sibling_from_branching_main_path_weight = 0
    

    if (current_group_weight >= average_node_weight or len(current_group) >= average_path_len) and current_group_weight > 0:
        if current_group[0] != source_node_name and current_group[len(current_group) - 1] != sink_node_name:
            for src_node in analysis_graph[current_group[0]].parents:
                if nodes_paths_mapping[src_node] == branching_main_path:
                    branch_start = src_node
            min_sink_level = math.inf
            for dst_node in analysis_graph[current_group[len(current_group) - 1]].children:
                if nodes_paths_mapping[dst_node] == branching_main_path:
                    branch_end = dst_node
                if analysis_graph[dst_node].level < min_sink_level:
                    min_sink_level = analysis_graph[dst_node].level
            
            if min_sink_level - analysis_graph[current_group[0]].level > len(current_group) * average_path_len:
                continue

            if branch_start != '' and branch_end != '':
                current_group_siblings_heads = analysis_graph[branch_start].children

                for node in current_group_siblings_heads:
                    if node == current_group[0]:
                        continue
                    if nodes_paths_mapping[node] != branching_main_path:
                        current_group_siblings_potentials = current_group_siblings_potentials + \
                            paths_max_potential[nodes_paths_mapping[node]]
                    else:
                        sibling_from_branching_main_path = node
                        traversal_queue = [sibling_from_branching_main_path]
                        visited_nodes = {}
                        while len(traversal_queue) > 0:
                            current_node = traversal_queue.pop(0)
                            if current_node != branch_end and current_node not in visited_nodes and analysis_graph[current_node].level < analysis_graph[branch_end].level:
                                sibling_from_branching_main_path_weight = sibling_from_branching_main_path_weight + \
                                    analysis_graph[current_node].duration
                                traversal_queue = traversal_queue + \
                                    graph[current_node]
                            visited_nodes[current_node] = 1

                total_branching_potential = (
                    current_group_siblings_potentials) + sibling_from_branching_main_path_weight
                in_tensor_size = 0
                out_tensor_size = 0
                if branch_start in tensors_sizes:
                    in_tensor_size = tensors_sizes[branch_start]
                else:
                    in_tensor_size = average_tensor_size_if_not_provided
                if current_group[len(current_group) - 1] in tensors_sizes:
                    out_tensor_size = tensors_sizes[current_group[len(current_group) - 1]]
                else:
                    out_tensor_size = average_tensor_size_if_not_provided

                group_comm_time = comm_latency * 2 + \
                    (in_tensor_size + out_tensor_size) * comm_transfer_rate

                if group_comm_time >= total_branching_potential + current_group_weight:
                    while initial_groups_indices[branching_main_path] == 0:
                        # union find like stuff
                        branching_main_path = path_joined_group[branching_main_path]
                    path_joined_group[i] = branching_main_path
                    initial_groups_indices[i] = 0
                    groups_weights[branching_main_path] = groups_weights[branching_main_path] + \
                        groups_weights[i]
                    if len(initial_groups[branching_main_path]) > 1:
                        main_path_tail = initial_groups[branching_main_path].pop(
                            len(initial_groups[branching_main_path]) - 1)
                        initial_groups[branching_main_path] = initial_groups[branching_main_path] + \
                            initial_groups[i]
                        initial_groups[branching_main_path].append(
                            main_path_tail)
                    else:
                        initial_groups[branching_main_path] = initial_groups[branching_main_path] + \
                            initial_groups[i]
    else:
        if branching_main_path == -1:
            branching_main_path = nodes_paths_mapping[analysis_graph[current_group[0]].parents[0]]
        while initial_groups_indices[branching_main_path] == 0:
            branching_main_path = path_joined_group[branching_main_path]
        path_joined_group[i] = branching_main_path
        initial_groups_indices[i] = 0
        groups_weights[branching_main_path] = groups_weights[branching_main_path] + \
            groups_weights[i]
        if len(initial_groups[branching_main_path]) > 1:
            main_path_tail = initial_groups[branching_main_path].pop(
                len(initial_groups[branching_main_path]) - 1)
            initial_groups[branching_main_path] = initial_groups[branching_main_path] + \
                initial_groups[i]
            initial_groups[branching_main_path].append(main_path_tail)
        else:
            initial_groups[branching_main_path] = initial_groups[branching_main_path] + \
                initial_groups[i]
tmp_initial_groups = initial_groups
initial_groups = []
tmp_groups_weights = groups_weights
groups_weights = []
num_initial_groups = 0
for i in range(0, num_paths):
    if initial_groups_indices[i] == 1:
        initial_groups.append(tmp_initial_groups[i])
        groups_weights.append(tmp_groups_weights[i])
        num_initial_groups = num_initial_groups + 1

# parts work distribution over levels
tasks_per_levels = []
max_levels = [0]*len(initial_groups)
min_levels = [20000]*len(initial_groups)

for i in range(0, len(initial_groups)):
    tasks_per_levels.append(collections.OrderedDict())
    current_group = initial_groups[i]
    for node in current_group:
        node_props = analysis_graph[node]
        node_level = int(node_props.level)
        if node_level in tasks_per_levels[i].keys():
            tasks_per_levels[i][node_level] += node_props.duration
        else:
            tasks_per_levels[i][node_level] = node_props.duration

        if node_level < min_levels[i]:
            min_levels[i] = node_level
        if node_level > max_levels[i]:
            max_levels[i] = node_level

# getting main groups-------------------------------------------------
# Returns sum of arr[0..index]. This function assumes 
# that the array is preprocessed and partial sums of 
# array elements are stored in BITree[]. 
def getsum(BITTree,i): 
    s = 0 #initialize result 
  
    # index in BITree[] is 1 more than the index in arr[] 
    i = i+1
  
    # Traverse ancestors of BITree[index] 
    while i > 0: 
  
        # Add current element of BITree to sum 
        s += BITTree[i] 
  
        # Move index to parent node in getSum View 
        i -= i & (-i) 
    return s 
  
# Updates a node in Binary Index Tree (BITree) at given index 
# in BITree. The given value 'val' is added to BITree[i] and 
# all of its ancestors in tree. 
def updatebit(BITTree , n , i ,v): 
  
    # index in BITree[] is 1 more than the index in arr[] 
    i += 1
  
    # Traverse all ancestors and add 'val' 
    while i <= n: 
  
        # Add 'val' to current node of BI Tree 
        BITTree[i] += v 
  
        # Update index to that of parent in update View 
        i += i & (-i) 
  
  
# Constructs and returns a Binary Indexed Tree for given 
# array of size n. 
def construct(arr, n): 
  
    # Create and initialize BITree[] as 0 
    BITTree = [0]*(n+1) 
  
    # Store the actual values in BITree[] using update() 
    for i in range(n): 
        updatebit(BITTree, n, i, arr[i]) 
  
    # Uncomment below lines to see contents of BITree[] 
    #for i in range(1,n+1): 
    #     print BITTree[i], 
    return BITTree


final_groups = []
final_groups_weights = []
to_be_merged_groups = []
to_be_merged_groups_weights = []

for i in range(1, no_of_desired_groups + 1):
    final_groups.append(copy.deepcopy(initial_groups[-i]))
    final_groups_weights.append(groups_weights[-i])

final_groups_work_per_levels = []
work_trees = []
final_groups_max_levels = [0]*len(final_groups)
final_groups_min_levels = [math.inf]*len(final_groups)

for indx in range(0, no_of_desired_groups):
    final_groups_work_per_levels.append([])
    work_trees.append([])
    final_groups_work_per_levels[indx] = [0] * no_of_levels

for indx in range(0, no_of_desired_groups):
    current_group = final_groups[indx]
    for node in current_group:
        final_groups_work_per_levels[indx][analysis_graph[node].level] += analysis_graph[node].duration

for indx in range(0, no_of_desired_groups):
    work_trees[indx] = construct(final_groups_work_per_levels[indx],no_of_levels) 

nodes_groups = {}
for node in all_nodes:
    nodes_groups[node] = -1

for i in range(0, len(final_groups)):
    for node in final_groups[i]:
        nodes_groups[node] = i

for i in range(0, len(initial_groups) - no_of_desired_groups):
    #if i not in filling_groups:
    to_be_merged_groups.append(copy.deepcopy(initial_groups[i]))
    to_be_merged_groups_weights.append(groups_weights[i])

# parts work distribution over levels
to_be_merged_groups_tasks_per_levels = []
to_be_merged_groups_len = len(to_be_merged_groups)
to_be_merged_groups_max_levels = [0] * to_be_merged_groups_len
to_be_merged_groups_min_levels = [math.inf] * to_be_merged_groups_len
to_be_merged_groups_densities = [0] * to_be_merged_groups_len
to_be_merged_groups_lengths = [0] * to_be_merged_groups_len
to_be_merged_groups_empty_spots = [0] * to_be_merged_groups_len
to_be_merged_groups_sorting_criteria = [0] * to_be_merged_groups_len
penalize_small_paths = [0] * to_be_merged_groups_len

for i in range(0, to_be_merged_groups_len):
    to_be_merged_groups_tasks_per_levels.append(collections.OrderedDict())
    current_group = to_be_merged_groups[i]
    min_level = math.inf
    max_level = 0
    for node in current_group:
        node_props = analysis_graph[node]
        node_level = int(node_props.level)
        if node_level in to_be_merged_groups_tasks_per_levels[i].keys():
            to_be_merged_groups_tasks_per_levels[i][node_level] += node_props.duration
        else:
            to_be_merged_groups_tasks_per_levels[i][node_level] = node_props.duration

        if node_level < min_level:
            min_level = node_level
        if node_level > max_level:
            max_level = node_level

    to_be_merged_groups_min_levels[i] = min_level
    to_be_merged_groups_max_levels[i] = max_level
    
    sink_level = math.inf
    for snk_node in analysis_graph[current_group[-1]].children:
        if int(analysis_graph[snk_node].level) < sink_level:
            sink_level = int(analysis_graph[dst_node].level)

    spanning_over = sink_level - min_level
    to_be_merged_groups_lengths[i] = len(current_group)
    to_be_merged_groups_empty_spots[i] = max(spanning_over - len(current_group) - (sink_level - max_level), 0)
    if len(current_group) < average_path_len:
        penalize_small_paths[i] = 1

    if spanning_over <= 0:
        to_be_merged_groups_densities[i] = 0
    else:
        to_be_merged_groups_densities[i] = to_be_merged_groups_weights[i] / spanning_over

normalized_densities_den = max(to_be_merged_groups_densities) - min(to_be_merged_groups_densities) + 1
normalized_lengths_den = max(to_be_merged_groups_lengths) - min(to_be_merged_groups_lengths) + 1
normalized_empty_spots_den = max(to_be_merged_groups_empty_spots) - min(to_be_merged_groups_empty_spots) + 1
normalized_weights_den = max(to_be_merged_groups_weights) - min(to_be_merged_groups_weights) + 1
normalized_densities_sub = min(to_be_merged_groups_densities)
normalized_lengths_sub = min(to_be_merged_groups_lengths)
normalized_weights_sub = min(to_be_merged_groups_weights)
normalized_empty_spots_sub = min(to_be_merged_groups_empty_spots)

for i in range(0, to_be_merged_groups_len):
    to_be_merged_groups_sorting_criteria[i] = (to_be_merged_groups_weights[i] - normalized_weights_sub) / normalized_weights_den + \
    (to_be_merged_groups_densities[i] - normalized_densities_sub) / normalized_densities_den + \
        (to_be_merged_groups_lengths[i] - normalized_lengths_sub) / (normalized_lengths_den) \
        - (to_be_merged_groups_empty_spots[i] - normalized_empty_spots_sub) / normalized_empty_spots_den - penalize_small_paths[i]

total_gain = 0

to_be_merged_groups_sorting_criteria, to_be_merged_groups_weights, to_be_merged_groups_min_levels, to_be_merged_groups, to_be_merged_groups_max_levels, to_be_merged_groups_tasks_per_levels = \
    (list(t) for t in zip(*sorted(zip(to_be_merged_groups_sorting_criteria, to_be_merged_groups_weights, to_be_merged_groups_min_levels, to_be_merged_groups, to_be_merged_groups_max_levels, to_be_merged_groups_tasks_per_levels), reverse=True)))
cntt = 0
# merging the groups
print("hhhhhhhhh")
for to_be_merged_group_index in range(0, len(to_be_merged_groups)):
    to_be_merged_group = to_be_merged_groups[to_be_merged_group_index]
    branch_main_path_indx = -1
    src_min_level = -1
    branch_src_node = ''
    branch_snk_node = ''
    min_sink_level = math.inf

    to_be_merged_group_comms = [0] * no_of_desired_groups

    for node in to_be_merged_group:
        for parent_node in rev_graph[node]:
            if nodes_groups[parent_node] != -1:
                to_be_merged_group_comms[nodes_groups[parent_node]] += edges_weights[parent_node]
        
        for child_node in graph[node]:
            if nodes_groups[child_node] != -1:
                to_be_merged_group_comms[nodes_groups[child_node]] += edges_weights[node]

    src_min_level = int(analysis_graph[to_be_merged_group[0]].level)

    for dst_node in analysis_graph[to_be_merged_group[-1]].children:
        if int(analysis_graph[dst_node].level) < min_sink_level:
            min_sink_level = int(analysis_graph[dst_node].level)

    min_sum_in_targeted_levels = math.inf
    merge_destination_index = 0
    
    for i in range(0, no_of_desired_groups):
        sum_in_targeted_levels = 0
        sum_in_targeted_levels = getsum(work_trees[i], min_sink_level - 1) - getsum(work_trees[i], src_min_level)  

        for comm_i in range(0, no_of_desired_groups):
            if comm_i != i:
                sum_in_targeted_levels += to_be_merged_group_comms[comm_i] 

        if sum_in_targeted_levels < min_sum_in_targeted_levels:
            min_sum_in_targeted_levels = sum_in_targeted_levels
            merge_destination_index = i

    merge_min_level = min(
        to_be_merged_groups_min_levels[to_be_merged_group_index], final_groups_min_levels[merge_destination_index])
    merge_max_level = max(
        to_be_merged_groups_max_levels[to_be_merged_group_index], final_groups_max_levels[merge_destination_index])
    final_groups_weights[merge_destination_index] += to_be_merged_groups_weights[to_be_merged_group_index]
    final_groups[merge_destination_index] += to_be_merged_group
    final_groups_min_levels[merge_destination_index] = merge_min_level
    final_groups_max_levels[merge_destination_index] = merge_max_level
    merge_src_levels_tasks = to_be_merged_groups_tasks_per_levels[to_be_merged_group_index]

    for node in to_be_merged_group:
        nodes_groups[node] = merge_destination_index

    for level, tasks_sum in merge_src_levels_tasks.items():
        final_groups_work_per_levels[merge_destination_index][level] += tasks_sum
        updatebit(work_trees[merge_destination_index], no_of_levels, level, tasks_sum)

nodes_groups[sink_node_name] = 0
print("gggggggggg")       
#post processing paths switching:
# work destribution among levels:
total_swapping_gain = 0
initial_groups_no = len(initial_groups)
initial_groups_indices = []
initial_groups_latest_sorces_levels = []
initial_groups_earliest_sink_levels = []
containing_groups_indices = []
already_swapped = {}
swap_groups_sorting_criteria = []

initial_group_indx = 0
len_of_the_smallest_main_group_candidate = len(initial_groups[-no_of_desired_groups])
for initial_group in initial_groups:
    if len(initial_group) >= len_of_the_smallest_main_group_candidate:
        break
    totally_contained = True
    start_node = initial_group[0]
    end_node = initial_group[-1]
    end_node_children = graph[end_node]
    first_child_group = nodes_groups[end_node_children[0]]
    for child in end_node_children:
        if nodes_groups[child] != first_child_group:
            totally_contained = False

    for parent in rev_graph[start_node]:
        if nodes_groups[parent] != first_child_group:
            totally_contained = False
            break
    
    if first_child_group == nodes_groups[start_node]:
        totally_contained = False

    if totally_contained:
        initial_groups_indices.append(initial_group_indx)
        containing_groups_indices.append(first_child_group)

        min_child_end_level = math.inf
        for child in end_node_children:
            current_child_level = analysis_graph[child].level
            if current_child_level < min_child_end_level:
                min_child_end_level = current_child_level

        initial_groups_earliest_sink_levels.append(min_child_end_level)
        initial_groups_latest_sorces_levels.append(analysis_graph[initial_group[0]].level - 1)

        swap_groups_sorting_criteria.append((initial_groups_earliest_sink_levels[-1] - initial_groups_latest_sorces_levels[-1]) * -1)

    initial_group_indx += 1

swap_groups_sorting_criteria, initial_groups_latest_sorces_levels, initial_groups_earliest_sink_levels, initial_groups_indices, containing_groups_indices = (list(t) for t in zip(
        *sorted(zip(swap_groups_sorting_criteria, initial_groups_latest_sorces_levels, initial_groups_earliest_sink_levels, initial_groups_indices, containing_groups_indices))))

no_of_swap_groups = len(initial_groups_indices)
containing_group_levels_work_in_swap_levels = [0] * no_of_swap_groups
swap_groups_final_group_levels_work_in_swap_levels = [0] * no_of_swap_groups
comm_with_containing_groups = [0] * no_of_swap_groups
comm_with_its_groups = [0] * no_of_swap_groups
swap_groups_final_groups = [0] * no_of_swap_groups

for group_indx in range(no_of_swap_groups - 2, -1, -1):
    swap_group = initial_groups[group_indx]
    start_node = swap_group[0]
    swap_group_final_group = nodes_groups[start_node]
    swap_group_containing_group = containing_groups_indices[group_indx]
    swap_groups_final_groups[group_indx] = swap_group_final_group

    swap_group_containing_group_levels_work = getsum(work_trees[swap_group_containing_group], initial_groups_earliest_sink_levels[group_indx] - 1) - \
        getsum(work_trees[swap_group_containing_group], initial_groups_latest_sorces_levels[group_indx] + 1)

    swap_group_final_group_levels_work = getsum(work_trees[swap_group_final_group], initial_groups_earliest_sink_levels[group_indx] - 1) - \
        getsum(work_trees[swap_group_final_group], initial_groups_latest_sorces_levels[group_indx] + 1)

    containing_group_levels_work_in_swap_levels[group_indx] = swap_group_containing_group_levels_work
    swap_groups_final_group_levels_work_in_swap_levels[group_indx] = swap_group_final_group_levels_work

    comm_with_containing_group = 0
    comm_with_its_group = 0

    for node in swap_group:
        for parent in rev_graph[node]:
            if nodes_groups[parent] == swap_group_containing_group:
                comm_with_containing_group += edges_weights[parent]
            elif nodes_groups[parent] == swap_group_final_group:
                comm_with_its_group += edges_weights[parent]
        
        for child in graph[node]:
            if nodes_groups[child] == swap_group_containing_group:
                comm_with_containing_group += edges_weights[node]
            elif nodes_groups[child] == swap_group_final_group:
                comm_with_its_group += edges_weights[node]

    comm_with_containing_groups[group_indx] = comm_with_containing_group
    comm_with_its_groups[group_indx] = comm_with_its_group
    group_indx += 1

for to_be_swapped_group_indx in range(0, no_of_swap_groups - 1):
    to_be_swapped_group_end_level = initial_groups_earliest_sink_levels[to_be_swapped_group_indx]
    to_be_swapped_group_final_group_indx = swap_groups_final_groups[to_be_swapped_group_indx]
    to_be_swapped_group_containing_group_indx = containing_groups_indices[to_be_swapped_group_indx]
    to_be_swapped_group_work = groups_weights[initial_groups_indices[to_be_swapped_group_indx]]
    to_be_swapped_group_containing_group_work = containing_group_levels_work_in_swap_levels[to_be_swapped_group_indx]
    to_be_swapped_group_final_group_work = containing_group_levels_work_in_swap_levels[to_be_swapped_group_final_group_indx]

    max_swapping_gain = 0
    swapping_candidate_indx = -1
    swap_with_group_indx = to_be_swapped_group_indx + 1
    swap_with_group_end_level = initial_groups_earliest_sink_levels[swap_with_group_indx]
    while swap_with_group_end_level <= to_be_swapped_group_end_level and swap_with_group_indx < no_of_swap_groups:
        if swap_with_group_indx not in already_swapped:
            swap_with_group_final_group_indx = swap_groups_final_groups[swap_with_group_indx]
            swap_with_group_containing_group_indx = containing_groups_indices[swap_with_group_indx]
            if swap_with_group_containing_group_indx ==  to_be_swapped_group_final_group_indx and \
                swap_with_group_final_group_indx == to_be_swapped_group_containing_group_indx:

                swap_with_group_work = groups_weights[initial_groups_indices[swap_with_group_indx]]
                swap_with_group_containing_group_work = containing_group_levels_work_in_swap_levels[swap_with_group_indx]

                current_max_time = max(\
                    comm_with_containing_groups[to_be_swapped_group_indx] + swap_with_group_containing_group_work, \
                            comm_with_containing_groups[swap_with_group_indx] + to_be_swapped_group_containing_group_work)
                
                an_alternative_max_time = max(comm_with_its_groups[to_be_swapped_group_indx] + to_be_swapped_group_work + to_be_swapped_group_containing_group_work - swap_with_group_work, \
                    comm_with_its_groups[swap_with_group_indx] + swap_with_group_work + swap_with_group_containing_group_work - to_be_swapped_group_work)
                
                swapping_gain = current_max_time - an_alternative_max_time
                if swapping_gain > max_swapping_gain:
                    max_swapping_gain = swapping_gain
                    swapping_candidate_indx = swap_with_group_indx

        swap_with_group_indx += 1
        if swap_with_group_indx < no_of_swap_groups:
            swap_with_group_end_level = initial_groups_earliest_sink_levels[swap_with_group_indx]

    if swapping_candidate_indx == -1:
        current_time = max(comm_with_containing_groups[to_be_swapped_group_indx] + to_be_swapped_group_final_group_work, \
            to_be_swapped_group_containing_group_work)
        an_alternative_time = max(comm_with_its_groups[to_be_swapped_group_indx] + to_be_swapped_group_work + \
            to_be_swapped_group_containing_group_work,to_be_swapped_group_final_group_work - to_be_swapped_group_work)
        if an_alternative_time < current_time:
            to_be_swapped_group = initial_groups[initial_groups_indices[to_be_swapped_group_indx]]
            for node in to_be_swapped_group:
                nodes_groups[node] = to_be_swapped_group_containing_group_indx
                node_duration = analysis_graph[node].duration
                node_level = analysis_graph[node].level
                final_groups_work_per_levels[to_be_swapped_group_containing_group_indx][node_level] += node_duration
                updatebit(work_trees[to_be_swapped_group_containing_group_indx], no_of_levels, node_level, node_duration) 
                final_groups_work_per_levels[to_be_swapped_group_final_group_indx][node_level] -= node_duration 
                updatebit(work_trees[to_be_swapped_group_final_group_indx], no_of_levels, node_level, -node_duration)
            total_swapping_gain += current_time - an_alternative_time
 
    if swapping_candidate_indx != -1 and swapping_candidate_indx:
        already_swapped[swap_with_group_indx] = 1
        already_swapped[swapping_candidate_indx] = 1
        to_be_swapped_group = initial_groups[initial_groups_indices[to_be_swapped_group_indx]]
        swap_with_group = initial_groups[initial_groups_indices[swapping_candidate_indx]]

        for node in swap_with_group:
            nodes_groups[node] = to_be_swapped_group_final_group_indx
            node_duration = analysis_graph[node].duration
            node_level = analysis_graph[node].level
            final_groups_work_per_levels[to_be_swapped_group_final_group_indx][node_level] +=  node_duration
            updatebit(work_trees[to_be_swapped_group_final_group_indx], no_of_levels, node_level, node_duration)
            final_groups_work_per_levels[to_be_swapped_group_containing_group_indx][node_level] -= node_duration
            updatebit(work_trees[to_be_swapped_group_containing_group_indx], no_of_levels, node_level, -node_duration) 
        for node in to_be_swapped_group:
            nodes_groups[node] = to_be_swapped_group_containing_group_indx
            node_duration = analysis_graph[node].duration
            node_level = analysis_graph[node].level
            final_groups_work_per_levels[to_be_swapped_group_containing_group_indx][node_level] += node_duration
            updatebit(work_trees[to_be_swapped_group_containing_group_indx], no_of_levels, node_level, node_duration)
            final_groups_work_per_levels[to_be_swapped_group_final_group_indx][node_level] -= node_duration
            updatebit(work_trees[to_be_swapped_group_final_group_indx], no_of_levels, node_level, node_duration)

        total_swapping_gain += max_swapping_gain

print('total swapping gain = ' + str(total_swapping_gain))


#post processing, switching nodes placement modification:
total_switching_gain = 0
switching_nodes_pure_parents = []
switching_nodes_pure_children = []
#start from level 2, since level 0 contains src -added by me- and 1 contains nodes that are not children of any node in the original graph
#exclude the last level since it only contains the sink
for i in range(2, no_of_levels - 1):
    for node in levels_nodes[i]:
        all_chidren_in_one_group = True
        all_parents_in_one_group = True
        children = graph[node]
        first_child_group = nodes_groups[children[0]]
        parents = rev_graph[node]
        first_parent_group = nodes_groups[parents[0]]
        for child_node in children:
            if nodes_groups[child_node] != first_child_group:
                all_chidren_in_one_group = False
                break
        for parent_node in parents:
            if nodes_groups[parent_node] != first_parent_group:
                all_parents_in_one_group = False
                break
        #note: if all chidren are in the same group and all parents in the same group, then all of them will be in the same group and
        # there is no switching, this is because at least one path will be passing through the switching point.
        switching_node_group = nodes_groups[node]
        if all_chidren_in_one_group and (not all_parents_in_one_group) and switching_node_group != first_child_group:
            switching_nodes_pure_children.append(node)
        if all_parents_in_one_group and (not all_chidren_in_one_group) and switching_node_group != first_parent_group:
            switching_nodes_pure_parents.append(node)

nodes_initial_groups = {}
for i in range(0, len(initial_groups)):
    for node in initial_groups[i]:
        nodes_initial_groups[node] = i

for switching_node in switching_nodes_pure_children:
    children_final_group = nodes_groups[graph[switching_node][0]]
    switching_node_group = nodes_groups[switching_node]
    comm_from_its_group = 0
    comm_from_children_group = 0
    some_parent_initial_group = -1
    for parent in rev_graph[switching_node]:
        if nodes_groups[parent] == children_final_group:
            comm_from_children_group += edges_weights[parent]
            if some_parent_initial_group == -1:
                some_parent_initial_group = nodes_initial_groups[parent]
        elif nodes_groups[parent] == switching_node_group:
            comm_from_its_group += edges_weights[parent]
    
    comm_from_children_group += edges_weights[switching_node]
    movement_gain = comm_from_children_group - (comm_from_its_group + analysis_graph[switching_node].duration)

    if movement_gain > 0:
        total_switching_gain += movement_gain
        switching_node_weight = analysis_graph[switching_node].duration
        switching_node_level = analysis_graph[switching_node].level
        final_groups_work_per_levels[nodes_groups[switching_node]][switching_node_level] -= switching_node_weight
        updatebit(work_trees[nodes_groups[switching_node]], no_of_levels, switching_node_level, -switching_node_weight)
        final_groups_work_per_levels[children_final_group][switching_node_level] += switching_node_weight
        updatebit(work_trees[children_final_group], no_of_levels, switching_node_level, switching_node_weight)
        nodes_groups[switching_node] = children_final_group
        switching_node_group_indx = nodes_initial_groups[switching_node]
        
        if switching_node_level > analysis_graph[initial_groups[some_parent_initial_group][-1]].level:
            initial_groups[some_parent_initial_group].append(switching_node)
        else:
            tail_node = initial_groups[some_parent_initial_group].pop(-1)
            initial_groups[some_parent_initial_group].append(switching_node)
            initial_groups[some_parent_initial_group].append(tail_node)

        initial_groups[switching_node_group_indx].remove(switching_node)
        nodes_initial_groups[switching_node] = some_parent_initial_group

for switching_node in switching_nodes_pure_parents:
    parents_final_group = nodes_groups[rev_graph[switching_node][0]]
    switching_node_group = nodes_groups[switching_node]
    some_child_initial_group = -1
    for child in graph[switching_node]:
        if nodes_groups[child] == parents_final_group:
            if some_child_initial_group == -1:
                some_child_initial_group = nodes_initial_groups[child]
                break
    
    if analysis_graph[switching_node].duration <= comm_latency:
        switching_node_weight = analysis_graph[switching_node].duration
        switching_node_level = analysis_graph[switching_node].level
        final_groups_work_per_levels[nodes_groups[switching_node]][switching_node_level] -= switching_node_weight
        updatebit(work_trees[nodes_groups[switching_node]], no_of_levels, switching_node_level, -switching_node_weight)
        final_groups_work_per_levels[parents_final_group][switching_node_level] += switching_node_weight
        updatebit(work_trees[parents_final_group], no_of_levels, switching_node_level, switching_node_weight)
        nodes_groups[switching_node] = parents_final_group
        switching_node_group_indx = nodes_initial_groups[switching_node]
        initial_groups[some_child_initial_group].insert(0, switching_node)
        initial_groups[switching_node_group_indx].remove(switching_node)
        nodes_initial_groups[switching_node] = some_child_initial_group

tmp_initial_groups = initial_groups
initial_groups = []
for group in tmp_initial_groups:
    if len(group) != 0:
        initial_groups.append(group)

print('total_switching_gain = ' + str(total_switching_gain))

#memory----------------------------------------------------------------------------------------
nodes_memory = {}
additional_memory = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_memory[node_name] = int(splitted[1])
        #if '^' + node_name in all_nodes:
        #    nodes_memory['^' + node_name] = int(splitted[1])

for node in all_nodes:
    if node not in nodes_memory:
        nodes_memory[node] = 0

parents_last_active_levels = {}
parents_all_active_levels = {}
nodes_last_child_level = {}
for node in all_nodes.keys():
    max_child_level = 0
    nodes_last_child_level[node] = 0
    parents_last_active_levels[node] = [analysis_graph[node].level + 1] * no_of_desired_groups
    parents_all_active_levels[node] = []
    for group in range(0, no_of_desired_groups):
        parents_all_active_levels[node].append([])
    for child in graph[node]:
        child_level = analysis_graph[child].level
        parents_all_active_levels[node][nodes_groups[child]].append(child_level)
        if  child_level > max_child_level:
            max_child_level = child_level
        
    nodes_last_child_level[node] = max_child_level

print('hhhhhhhhhhhhhhhhhhhhhh')
additional_memory[source_node_name] = 0
for node, parents in rev_graph.items():
    node_additional_memory = 0
    if node != sink_node_name:
        for parent in parents:
            if analysis_graph[node].level >= parents_last_active_levels[parent][nodes_groups[node]]:
                node_additional_memory += nodes_memory[parent]
    
    additional_memory[node] = node_additional_memory

summ = 0
for node in graph:
    if node != sink_node_name and graph[node][0] == sink_node_name:
        summ += nodes_memory[node]
        
levels_additional_memory = []
levels_additional_memory_by_node = []
for i in range(0, no_of_desired_groups):
    levels_additional_memory.append([])
    levels_additional_memory_by_node.append([])

for i in range(0, no_of_desired_groups):
    for j in range(0, no_of_levels):
        levels_additional_memory_by_node[i].append([])

current_level = 0
for current_level_nodes in levels_nodes:
    max_addition = [0] * no_of_desired_groups
    for node in current_level_nodes:
        current_node_group = nodes_groups[node]
        max_addition[current_node_group] = max(
            additional_memory[node], max_addition[current_node_group])
        levels_additional_memory_by_node[current_node_group][current_level].append(additional_memory[node])
    current_level += 1

    for i in range(0, no_of_desired_groups):
        levels_additional_memory[i].append(max_addition[i])
print('nnnnnnnnnnnnnnn')
def get_levels_memory_consumption(graph, src_nodes=None):
    # getting the sources of the graph to start the topological traversal from them
    levels_memory_consumption = []
    for group in range(0, no_of_desired_groups):
        levels_memory_consumption.append([0] * no_of_levels)

    tmp_nodes_in_degrees = copy.deepcopy(nodes_in_degrees)

    traversal_queueu = queue.Queue()

    if src_nodes is None:
        src_nodes = [source_node_name]

    for node in src_nodes:
        traversal_queueu.put(node)

    # start the traversal
    while not traversal_queueu.empty():
        current_node = traversal_queueu.get()
        current_level = analysis_graph[current_node].level
        current_group = nodes_groups[current_node]

        if current_node in graph:
            adj_nodes = graph[current_node]
        else:
            adj_nodes = []
            
        levels_memory_consumption[current_group][current_level] += nodes_memory[current_node]

        if current_node != sink_node_name:
            for parent_node in rev_graph[current_node]:
                if nodes_memory[parent_node] > 0:
                    for level in range(parents_last_active_levels[parent_node][current_group], current_level):
                        levels_memory_consumption[current_group][level] += nodes_memory[parent_node]
                parents_last_active_levels[parent_node][current_group] = current_level
        else:
            for parent_node in rev_graph[current_node]:
                parent_group = nodes_groups[parent_node]
                if nodes_memory[parent_node] > 0:
                    for level in range(analysis_graph[parent].level + 1, current_level):
                        levels_memory_consumption[parent_group][level] += nodes_memory[parent_node]

        for adj_node in adj_nodes:
            tmp_nodes_in_degrees[adj_node] -= 1
            if tmp_nodes_in_degrees[adj_node] == 0:
                traversal_queueu.put(adj_node)

    """ for group in range(0, no_of_desired_groups):
        for level in range(0, no_of_levels):
            levels_memory_consumption[group][level] += levels_additional_memory[group][level] """
    
    return levels_memory_consumption

final_groups_memory_consumptions = get_levels_memory_consumption(graph)

max_mem = 0
for level in range(0, no_of_levels):
    _str = '' + str(level) + '::'
    sum_in_level = 0
    prntt = False
    for group in range(0, no_of_desired_groups):
        sum_in_level += final_groups_memory_consumptions[group][level]
        if final_groups_memory_consumptions[group][level] > 20.0 * (1024 * 1024 * 1024):
            prntt = True
        _str += str(final_groups_memory_consumptions[group][level] / (1024 * 1024 * 1024)) + ' '
    if sum_in_level > max_mem:
        max_mem = sum_in_level
    if prntt:
        print(_str)

print('max consumption in a level is: ' + str(max_mem / (1024 * 1024 * 1024)))

for i in range(0, no_of_desired_groups):
    for level in range(len(final_groups_memory_consumptions[i]) - 1, no_of_levels):
        final_groups_memory_consumptions[i][level] = final_groups_memory_consumptions[i][level - 1] 

memory_limit_is_exceeded = False
for i in range(0, no_of_desired_groups):
    for level in range(no_of_levels - 1, 0, -1):
        if final_groups_memory_consumptions[i][level] > memory_limit_per_group:
            memory_limit_is_exceeded = True
            break

if memory_limit_is_exceeded:
    print('limit is exceeded')

    # memory consumption of the initial groups
    initial_groups_mem_cons = []
    current_initial_group = 0
    for group in initial_groups:
        current_group_mem_cons = 0
        for node in group:
            current_group_mem_cons += nodes_memory[node]
        initial_groups_mem_cons.append(current_group_mem_cons)

    #communication and computation of the initial groups
    current_group_indx = 0
    initial_groups_comps_comms = []
    initial_groups_comms = []
    initial_groups_comm_comp_to_mem_ratio = []
    initial_groups_earliest_sink_levels = []
    initial_groups_lengths = []
    initial_groups_earliest_parents = []
    initial_groups_latest_childs = []
    com_comp_min = math.inf
    com_comp_max = 0
    ratio_min = math.inf
    ratio_max = 0
    for initial_group in initial_groups:
        end_node = initial_group[-1]
        initial_groups_lengths.append(len(initial_group))
        initial_groups_earliest_parents.append(analysis_graph[initial_group[0]].level)
        initial_groups_latest_childs.append(analysis_graph[end_node].level)
        comm_costs = [0] * no_of_desired_groups
        for node in initial_group:
            for parent_node in rev_graph[node]:
                comm_costs[nodes_groups[parent_node]] += edges_weights[parent_node]
                if parent != source_node_name and nodes_memory[parent] > 0:
                    parent_level = analysis_graph[parent_node].level
                    if parent_level < initial_groups_earliest_parents[-1]:
                        initial_groups_earliest_parents[-1] = parent_level
            
            node_memory = nodes_memory[node]
            for child_node in graph[node]:
                comm_costs[nodes_groups[child_node]] += edges_weights[node]
                if node_memory > 0:
                    child_level = analysis_graph[child_node].level
                    if child_level > initial_groups_latest_childs[-1]:
                        initial_groups_latest_childs[-1] = child_level

        initial_groups_comms.append(comm_costs)
        total_cost = sum(comm_costs) + groups_weights[current_group_indx]
        initial_groups_comps_comms.append(total_cost)
        ratio = total_cost / (initial_groups_mem_cons[current_group_indx] + 1) #negligable addition, just to avoid devision by zero without using if statements
        initial_groups_comm_comp_to_mem_ratio.append(ratio)

        min_child_end_level = math.inf
        for child in graph[end_node]:
            current_child_level = analysis_graph[child].level
            if current_child_level < min_child_end_level:
                min_child_end_level = current_child_level

        initial_groups_earliest_sink_levels.append(min_child_end_level)

        if total_cost > com_comp_max:
            com_comp_max = total_cost
        elif total_cost < com_comp_min:
            com_comp_min = total_cost

        if ratio > ratio_max:
            ratio_max = ratio
        elif ratio < ratio_min:
            ratio_min = ratio

        current_group_indx += 1

    normalized_ratio_den = ratio_max - ratio_min
    normalized_comm_comp_den = com_comp_max - com_comp_min
    group_len_min = min(initial_groups_lengths)
    normalized_group_len_den = max(initial_groups_lengths) - group_len_min + 1
    initial_groups_sorting_criteria = []
    for i in range(0, len(initial_groups)):
        initial_groups_sorting_criteria.append( (initial_groups_comps_comms[i] - com_comp_min) / normalized_comm_comp_den + \
            (initial_groups_comm_comp_to_mem_ratio[i] - ratio_min) / normalized_ratio_den + \
                (initial_groups_lengths[i] - group_len_min) / normalized_group_len_den )

    initial_groups_latest_childs, initial_groups_sorting_criteria, initial_groups, initial_groups_earliest_sink_levels, initial_groups_comms, initial_groups_earliest_parents = (list(t) for t in zip(
        *sorted(zip(initial_groups_latest_childs, initial_groups_sorting_criteria, initial_groups, initial_groups_earliest_sink_levels, initial_groups_comms, initial_groups_earliest_parents))))

    initial_final_group_mapping = {}
    current_initial_group_indx = 0
    for group in initial_groups:
        current_final_group = nodes_groups[group[0]]
        initial_final_group_mapping[current_initial_group_indx] = current_final_group
        current_initial_group_indx += 1

    for group_no in range(0, no_of_desired_groups):
        current_level = 0
        start_indx = 0
        while current_level < no_of_levels:
            if final_groups_memory_consumptions[group_no][current_level] > memory_limit_per_group:
                
                overflow = final_groups_memory_consumptions[group_no][current_level] - \
                    memory_limit_per_group
                candidate_groups_indices = []
                candidate_groups_sotring_weights = []
                current_indx = start_indx
                start_indx = -1
                while current_indx < len(initial_groups):
                    last_node_in_a_current_group = initial_groups[current_indx][-1]
                    if initial_final_group_mapping[current_indx] == group_no and (initial_groups_latest_childs[current_indx] \
                         >= current_level or sink_node_name in graph[last_node_in_a_current_group]):
                        if start_indx == -1:
                            start_indx = current_indx
                        if initial_groups_earliest_parents[current_indx] < current_level:
                            candidate_groups_indices.append(current_indx)
                            candidate_groups_sotring_weights.append(initial_groups_sorting_criteria[current_indx])
                    current_indx += 1

                if candidate_groups_indices:
                    candidate_groups_sotring_weights, candidate_groups_indices = (list(t) for t in zip(
                        *sorted(zip(candidate_groups_sotring_weights, candidate_groups_indices))))
                else:
                    print('cannot be addressed')
                    exit() 

                for sub_group_indx in candidate_groups_indices:
                    current_sub_group = initial_groups[sub_group_indx]
                    merge_with_index = -1
                    sub_group_start_level = analysis_graph[current_sub_group[0]].level

                    final_groups_indices = []
                    for i in range(0, no_of_desired_groups):
                        final_groups_indices.append(i)

                    comm_comp_with_final_groups = initial_groups_comms[sub_group_indx]
                    for i in range(0, no_of_desired_groups):
                        comm_comp_with_final_groups[i] = \
                            getsum(work_trees[i], initial_groups_earliest_sink_levels[sub_group_indx] - 1) - \
                                getsum(work_trees[i], sub_group_start_level - 1)                         
                    
                    comm_comp_with_final_groups, final_groups_indices = (list(t) for t in zip( *sorted(zip(comm_comp_with_final_groups, final_groups_indices))))

                    merged = False

                    for final_group_indx in final_groups_indices:
                        if final_group_indx == group_no:
                            continue
                        merged = True
                        tmp_groups_mem_consumption_dst = {}#copy.deepcopy(final_groups_memory_consumptions)
                        tmp_groups_mem_consumption_src = {}#copy.deepcopy(final_groups_memory_consumptions)
                        tmp_levels_additional_memory_dst = {}#copy.deepcopy(levels_additional_memory)
                        tmp_levels_additional_memory_src = {}#copy.deepcopy(levels_additional_memory)
                        tmp_levels_additional_memory_by_node_dst = {}#copy.deepcopy(levels_additional_memory_by_node)
                        tmp_levels_additional_memory_by_node_src = {}#copy.deepcopy(levels_additional_memory_by_node)
                        tmp_parents_all_active_levels_dst = {}#copy.deepcopy(parents_all_active_levels)
                        tmp_parents_all_active_levels_src = {}#copy.deepcopy(parents_all_active_levels)
                        tmp_parents_last_active_levels_dst = {}#copy.deepcopy(parents_last_active_levels)
                        tmp_parents_last_active_levels_src = {}#copy.deepcopy(parents_last_active_levels)
                        for node in current_sub_group:
                            node_level = analysis_graph[node].level
                            node_additional_memory = additional_memory[node]
                            node_mem_cons = nodes_memory[node]
                            
                            tmp_levels_additional_memory_by_node_src[node_level] = copy.deepcopy(levels_additional_memory_by_node[group_no][node_level])
                            tmp_levels_additional_memory_by_node_dst[node_level] = copy.deepcopy(levels_additional_memory_by_node[final_group_indx][node_level])
                            tmp_levels_additional_memory_src[node_level] = levels_additional_memory[group_no][node_level]
                            tmp_levels_additional_memory_dst[node_level] = levels_additional_memory[final_group_indx][node_level]
                            tmp_groups_mem_consumption_src[node_level] = final_groups_memory_consumptions[group_no][node_level]
                            tmp_groups_mem_consumption_dst[node_level] = final_groups_memory_consumptions[final_group_indx][node_level]

                            if tmp_levels_additional_memory_by_node_src[node_level]:
                                if node_additional_memory in tmp_levels_additional_memory_by_node_src[node_level]:
                                    tmp_levels_additional_memory_by_node_src[node_level].remove(node_additional_memory)
                            if tmp_levels_additional_memory_by_node_src[node_level] and tmp_levels_additional_memory_src[node_level] <= node_additional_memory:
                                tmp_levels_additional_memory_src[node_level] = \
                                    max(tmp_levels_additional_memory_by_node_src[node_level])
                                tmp_groups_mem_consumption_src[node_level] += tmp_levels_additional_memory_src[node_level]
                                tmp_groups_mem_consumption_src[node_level] -= node_additional_memory

                            tmp_levels_additional_memory_by_node_dst[node_level].append(node_additional_memory)
                            if tmp_levels_additional_memory_dst[node_level] < node_additional_memory:
                                tmp_levels_additional_memory_dst[node_level] = node_additional_memory
                                tmp_groups_mem_consumption_dst[node_level] += node_additional_memory
                                tmp_groups_mem_consumption_dst[node_level] -= \
                                    levels_additional_memory[final_group_indx][node_level]
                                if tmp_groups_mem_consumption_dst[node_level] > memory_limit_per_group:
                                    merged = False
                                    break
                
                            tmp_groups_mem_consumption_src[node_level] -= node_mem_cons
                            tmp_groups_mem_consumption_dst[node_level] += node_mem_cons
                            if tmp_groups_mem_consumption_dst[node_level] > memory_limit_per_group:
                                merged = False
                                break
                            parents = rev_graph[node]
                            if parents[0] != source_node_name:
                                for parent in parents:
                                    parent_mem_cons = nodes_memory[parent]
                                    if parent_mem_cons > 0:
                                        if parent not in tmp_parents_all_active_levels_src:
                                            tmp_parents_all_active_levels_src[parent] =copy.deepcopy(parents_all_active_levels[parent][group_no])
                                        if parent not in tmp_parents_all_active_levels_dst:
                                            tmp_parents_all_active_levels_dst[parent] = copy.deepcopy(parents_all_active_levels[parent][final_group_indx])

                                        tmp_parents_all_active_levels_src[parent].remove(node_level)

                                        tmp_parents_all_active_levels_dst[parent].append(node_level)
                                        tmp_parents_last_active_levels_src[parent] = parents_last_active_levels[parent][group_no]
                                        if node_level == tmp_parents_last_active_levels_src[parent]:
                                            if tmp_parents_all_active_levels_src[parent]:
                                                tmp_parents_last_active_levels_src[parent] = max(tmp_parents_all_active_levels_src[parent])
                                            else:
                                                tmp_parents_last_active_levels_src[parent] = analysis_graph[parent].level + 1
                                            for level in range(tmp_parents_last_active_levels_src[parent], node_level):
                                                if level not in tmp_groups_mem_consumption_src:
                                                    tmp_groups_mem_consumption_src[level] = final_groups_memory_consumptions[group_no][level]
                                                tmp_groups_mem_consumption_src[level] -= parent_mem_cons
                                        
                                        previous_last_level = parents_last_active_levels[parent][final_group_indx]
                                        if node_level > previous_last_level:
                                            for level in range(previous_last_level, node_level):
                                                if level not in tmp_groups_mem_consumption_dst:
                                                    tmp_groups_mem_consumption_dst[level] = final_groups_memory_consumptions[final_group_indx][level]
                                                tmp_groups_mem_consumption_dst[level] += parent_mem_cons
                                                if tmp_groups_mem_consumption_dst[level] > memory_limit_per_group:
                                                    merged = False
                                                    break
                                            tmp_parents_last_active_levels_dst[parent] = node_level
                            if not merged:
                                break
                        if merged:
                            for level, consump in tmp_groups_mem_consumption_src.items():
                                final_groups_memory_consumptions[group_no][level] = consump
                            for level, consump in tmp_groups_mem_consumption_dst.items():
                                final_groups_memory_consumptions[final_group_indx][level] = consump

                            for level, add_mem in tmp_levels_additional_memory_src.items():
                                levels_additional_memory[group_no][level] = add_mem
                            for level, add_mem in tmp_levels_additional_memory_dst.items():
                                levels_additional_memory[final_group_indx][level] = add_mem

                            for level, add_mem in tmp_levels_additional_memory_by_node_src.items():
                                levels_additional_memory_by_node[group_no][level] = add_mem
                            for level, add_mem in tmp_levels_additional_memory_by_node_dst.items():
                                levels_additional_memory_by_node[final_group_indx][level] = add_mem

                            for parent, active_levels in tmp_parents_all_active_levels_src.items():
                                parents_all_active_levels[parent][group_no] = copy.deepcopy(active_levels)

                            for parent, active_levels in tmp_parents_all_active_levels_dst.items():        
                                parents_all_active_levels[parent][final_group_indx] = copy.deepcopy(active_levels)

                            for parent, active_level in tmp_parents_last_active_levels_src.items():
                                parents_last_active_levels[parent][group_no] = active_level
                            for parent, active_level in tmp_parents_last_active_levels_dst.items():
                                parents_last_active_levels[parent][final_group_indx] = active_level

                            initial_final_group_mapping[sub_group_indx] = final_group_indx
                            
                            for node in current_sub_group:
                                nodes_groups[node] = final_group_indx
                                node_comp = analysis_graph[node].duration
                                node_level = analysis_graph[node].level
                                if node_comp > 0:
                                    final_groups_work_per_levels[group_no][node_level] -= node_comp
                                    updatebit(work_trees[group_no], no_of_levels, node_level, -node_comp)
                                    final_groups_work_per_levels[final_group_indx][node_level] += node_comp
                                    updatebit(work_trees[final_group_indx], no_of_levels, node_level, node_comp)

                            break

                    if final_groups_memory_consumptions[group_no][current_level] <= memory_limit_per_group:
                        break
                        
                if final_groups_memory_consumptions[group_no][current_level] > memory_limit_per_group:
                    print('cannot be addressed')
                    print(final_groups_memory_consumptions[group_no][current_level])
                    print(current_level)
                    exit()

            current_level += 1

with open(out1, 'w') as f:
    smm = [0] * (no_of_desired_groups + 1 )
    light_levels_sum = [0] * (no_of_desired_groups + 1)
    cntt = [0] * (no_of_desired_groups + 1)
    count = [0] * (no_of_desired_groups + 1)
    for node, group in nodes_groups.items():
        if not node.startswith("^"):
            f.write(node + ' ' + str(group) + '\n')
            smm[group] += analysis_graph[node].duration
            if analysis_graph[node].level >= 510 and analysis_graph[node].level <= 650:
                light_levels_sum[group] += analysis_graph[node].duration
                count[group] += 1
            cntt[group] += 1

    for i in range(0, no_of_desired_groups):
        print(str(i) + ': ' + str(cntt[i]) +
              ', ' + str(smm[i]) + ', ' + str(count[i]) + ', ' + str(light_levels_sum[i]))