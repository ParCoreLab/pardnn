import utils
import nodeProps
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import collections
import heapq

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/'

""" in1 = io_folder_path + 'inc_A_dot_src_sink.dot'#'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step303_n_c.json'
in3 = io_folder_path + 'src_snk_nodes_levels.txt'#'part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_src_snk_nodes_levels.txt'#'rev_part_8_1799_src_sink_nodes_levels.txt'
in5 = io_folder_path + 'tensors_sz.txt' """

in1 = io_folder_path + 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step303.json'
in3 = io_folder_path + 'part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_part_8_1799_src_sink_nodes_levels.txt'
in5 = io_folder_path + 'tensors_sz.txt'

# output file
out1 = io_folder_path + 'ver_grouper_placement_e.place'

# grouper parameters
path_length_threshold = 10
group_weight_threshold = 250
no_of_desired_groups = 2

comm_latency = 2 * 15
average_tensor_size_if_not_provided = 100000000
comm_transfer_rate = 1000000 / (24 * 1024 * 1024 * 1024)

reverse_levels = {}
# get nodes levels
with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            reverse_levels[node_and_level[0]] = node_and_level[1]

tensors_sizes = {}
# get tensors sizes
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensors_sizes[splitted[0]] = splitted[1]

# will contain the graph as an adgacency list
graph = {}
all_nodes = {}
sink_node_name = 'snk'
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
                heapq.heappush(graph[splits[0]], (int(
                    reverse_levels[splits[1]]), splits[1]))
            else:
                graph[splits[0]] = []
                heapq.heappush(graph[splits[0]], (int(
                    reverse_levels[splits[1]]), splits[1]))

graph[sink_node_name] = []


# change adjacents priority queues to lists
for node, adjacents in graph.items():
    ordered_adjacents_list = []
    while adjacents:
        ordered_adjacents_list.append(heapq.heappop(adjacents)[1])
    graph[node] = ordered_adjacents_list


# getting time (weight) info for nodes
analysis_graph = utils.read_profiling_file(in2, True)

for node, node_props in all_nodes.items():
    if node in analysis_graph:
        analysis_graph[node].parents = node_props.parents
        analysis_graph[node].children = node_props.children
    else:
        analysis_graph[node] = node_props


# extracting all vertical paths in the graph
source_node_name = 'src'
traversal_stack = [source_node_name]
paths = []
current_path = []
visited = {}
groups_weights = []
paths_lengths = []
current_path_weight = 0
num_paths = 0
nodes_paths_mapping = {}

while len(traversal_stack) > 0:
    current_node = traversal_stack.pop()
    if current_node not in visited:
        current_path.append(current_node)
        current_path_weight = current_path_weight + analysis_graph[current_node].duration
        adj_nodes = graph[current_node]
        visited[current_node] = 1
        all_neighbors_visited = True
        for adj_node in adj_nodes:
            if adj_node not in visited and adj_node != sink_node_name:
                all_neighbors_visited = False
                traversal_stack.append(adj_node)
        if all_neighbors_visited:
            paths.append(copy.deepcopy(current_path))
            groups_weights.append(current_path_weight)
            paths_lengths.append(len(current_path))
            current_path = []
            current_path_weight = 0
            num_paths = num_paths + 1

# sort paths from shortest to longest
paths_lengths, groups_weights, paths = (list(t) for t in zip(
    *sorted(zip(paths_lengths, groups_weights, paths))))

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
    if current_path[0] != source_node_name and current_path[current_path_len] != sink_node_name:
        for src_node in analysis_graph[current_path[0]].parents:
            for dst_node in analysis_graph[current_path[current_path_len]].children:
                if nodes_paths_mapping[src_node] == nodes_paths_mapping[dst_node]:
                    parent_path_indx = nodes_paths_mapping[src_node]
                    paths_max_potential[parent_path_indx] = paths_max_potential[parent_path_indx] + \
                        paths_max_potential[i]
                    found = True
                    break
                if found:
                    break
    groups_parents[i] = parent_path_indx

# get nodes levels
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            analysis_graph[node_and_level[0]].level = node_and_level[1]


# getting initial groups
initial_groups = copy.deepcopy(paths)
initial_groups_indices = [1] * num_paths

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

    if groups_weights[i] > comm_latency and len(current_group) > 4:
        if current_group[0] != source_node_name and current_group[len(current_group) - 1] != sink_node_name:
            for src_node in analysis_graph[current_group[0]].parents:
                if nodes_paths_mapping[src_node] == branching_main_path:
                    branch_start = src_node
            for dst_node in analysis_graph[current_group[len(current_group) - 1]].children:
                if nodes_paths_mapping[dst_node] == branching_main_path:
                    branch_end = dst_node
            
            if branch_start != '' and branch_end != '':
                current_group_siblings_heads = analysis_graph[branch_start].children
                current_group_siblings_heads.remove(current_group[0])

                for node in current_group_siblings_heads:
                    if nodes_paths_mapping[node] != branching_main_path:
                        current_group_siblings_potentials = current_group_siblings_potentials + paths_max_potential[nodes_paths_mapping[node]]
                    else:
                        sibling_from_branching_main_path = node
                        traversal_queue = [sibling_from_branching_main_path]
                        visited_nodes = {}
                        while len(traversal_queue) > 0:
                            current_node = traversal_queue.pop(0)
                            if current_node != branch_end and current_node not in visited_nodes and analysis_graph[current_node].level < analysis_graph[branch_end].level:
                                sibling_from_branching_main_path_weight = sibling_from_branching_main_path_weight + analysis_graph[current_node].duration
                                traversal_queue = traversal_queue + graph[current_node]
                            visited_nodes[current_node] =  1
                
                total_branching_potential = (current_group_siblings_potentials) + sibling_from_branching_main_path_weight
                in_tensor_size = 0
                out_tensor_size = 0
                if branch_start in tensors_sizes:
                    in_tensor_size = int(tensors_sizes[branch_start])
                else:
                    in_tensor_size = average_tensor_size_if_not_provided
                if current_group[len(current_group) - 1] in tensors_sizes:
                    out_tensor_size = int(tensors_sizes[current_group[len(current_group) - 1]])
                else:
                    out_tensor_size = average_tensor_size_if_not_provided

                group_comm_time = comm_latency + (in_tensor_size + out_tensor_size) * comm_transfer_rate


                if group_comm_time > total_branching_potential:
                    initial_groups_indices[i] = 0
                    groups_weights[branching_main_path] = groups_weights[branching_main_path] + groups_weights[i]
                    main_path_tail = initial_groups[branching_main_path].pop(len(initial_groups[branching_main_path]) - 1)
                    initial_groups[branching_main_path] = initial_groups[branching_main_path] + initial_groups[i]
                    initial_groups[branching_main_path].append(main_path_tail)
    else:
        #print(str(len(current_group)) + ': ' + str(groups_weights[i]))
        if branching_main_path == -1:
            branching_main_path = nodes_paths_mapping[analysis_graph[current_group[0]].parents[0]]
        initial_groups_indices[i] = 0
        groups_weights[branching_main_path] = groups_weights[branching_main_path] + groups_weights[i]
        main_path_tail = initial_groups[branching_main_path].pop(len(initial_groups[branching_main_path]) - 1)
        initial_groups[branching_main_path] = initial_groups[branching_main_path] + initial_groups[i]
        initial_groups[branching_main_path].append(main_path_tail)

    
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
            tasks_per_levels[i][node_level] = tasks_per_levels[i][node_level] + node_props.duration
        else:
            tasks_per_levels[i][node_level] = node_props.duration
        
        if node_level < min_levels[i]:
            min_levels[i] = node_level
        if node_level > max_levels[i]:
            max_levels[i] = node_level
    
# getting main groups
main_groups_indices = [-1] * no_of_desired_groups
for i in range(0, no_of_desired_groups):
    main_groups_indices[i] = num_initial_groups - i - 1


# merging the groups
while len(initial_groups) > no_of_desired_groups:
    to_be_merged_group_index = len(initial_groups) - no_of_desired_groups - 1
    to_be_merged_group = initial_groups[to_be_merged_group_index]    
    branch_main_path_indx = 0

    src_min_level = -1
    branch_src_node = ''
    src_max_level = 2000
    for src_node in analysis_graph[to_be_merged_group[0]].parents:
        if int(analysis_graph[src_node].level) > src_min_level:
            src_min_level = int(analysis_graph[src_node].level)
            branch_src_node = src_node
    for dst_node in analysis_graph[to_be_merged_group[len(to_be_merged_group) - 1]].children:
        if int(analysis_graph[dst_node].level) < src_max_level:
            src_max_level = int(analysis_graph[dst_node].level)

    if src_max_level == 2000:
        src_max_level = analysis_graph[to_be_merged_group[len(to_be_merged_group) - 1]].level
    if src_min_level == -1:
        src_min_level = analysis_graph[to_be_merged_group[0]].level

    for i in range(0, len(initial_groups)):
        if branch_src_node in initial_groups[i]:
            branch_main_path_indx = i

    in_tensor_size = 0
    out_tensor_size = 0
    if branch_src_node in tensors_sizes:
        in_tensor_size = int(tensors_sizes[branch_src_node])
    else:
        in_tensor_size = average_tensor_size_if_not_provided

    if to_be_merged_group[len(to_be_merged_group) - 1] in tensors_sizes:
        out_tensor_size = int(tensors_sizes[to_be_merged_group[len(to_be_merged_group) - 1]])
    else:
        out_tensor_size = average_tensor_size_if_not_provided

    group_comm_time = comm_latency + (in_tensor_size + out_tensor_size) * comm_transfer_rate

    min_sum_in_targeted_levels = 1000000000
    main_branch_sum_in_targeted_levels = 0
    merge_destination_index = 0

    for i in main_groups_indices:   
        sum_in_targeted_levels = 0
        current_min_level = max(src_min_level, min_levels[i])
        current_max_level = min(src_max_level, max_levels[i])

        for level in range(current_min_level, current_max_level):
            if level in tasks_per_levels[i].keys(): #and level in tasks_per_levels[to_be_merged_group_index].keys():
                sum_in_targeted_levels = sum_in_targeted_levels + tasks_per_levels[i][level]

        if sum_in_targeted_levels < min_sum_in_targeted_levels:
            min_sum_in_targeted_levels = sum_in_targeted_levels
            merge_destination_index = i
        if i == branch_main_path_indx:
            main_branch_sum_in_targeted_levels = sum_in_targeted_levels

    if merge_destination_index != branch_main_path_indx:
        current_branch_max_time = min_sum_in_targeted_levels + group_comm_time + groups_weights[to_be_merged_group_index]
        an_alternative_max_time = main_branch_sum_in_targeted_levels + groups_weights[to_be_merged_group_index]
        if int(analysis_graph[branch_src_node].level) >=650 and int(analysis_graph[branch_src_node].level) <=731:
            print(str(min_sum_in_targeted_levels) + ' ' + str(main_branch_sum_in_targeted_levels))
        if an_alternative_max_time < current_branch_max_time:
            merge_destination_index = branch_main_path_indx
        

    merge_min_level = min(min_levels[to_be_merged_group_index], min_levels[merge_destination_index])
    merge_max_level = max(max_levels[to_be_merged_group_index], max_levels[merge_destination_index])

    if int(analysis_graph[branch_src_node].level) >=650 and int(analysis_graph[branch_src_node].level) <=731:
        print('MMN: ' + str(min_levels[to_be_merged_group_index]) + ' : MMX:' + str(max_levels[to_be_merged_group_index]) + ', ' + str(groups_weights[to_be_merged_group_index]))
    groups_weights[merge_destination_index] = groups_weights[merge_destination_index]  + groups_weights[to_be_merged_group_index]
    groups_weights.pop(to_be_merged_group_index)
    initial_groups[merge_destination_index] = initial_groups[merge_destination_index] + to_be_merged_group
    initial_groups.pop(to_be_merged_group_index)
    min_levels[merge_destination_index] = merge_min_level
    max_levels[merge_destination_index] = merge_max_level
    min_levels.pop(to_be_merged_group_index)
    max_levels.pop(to_be_merged_group_index)
    merge_src_levels_tasks = tasks_per_levels[to_be_merged_group_index]

    for j in range(0, no_of_desired_groups):
        main_groups_indices[j] = main_groups_indices[j] - 1

    for level, tasks_sum in merge_src_levels_tasks.items():
        if level in tasks_per_levels[merge_destination_index].keys():
            tasks_per_levels[merge_destination_index][level] = tasks_per_levels[merge_destination_index][level] + tasks_sum
        else:
            tasks_per_levels[merge_destination_index][level] = tasks_sum
    tasks_per_levels.pop(to_be_merged_group_index)
    


count = 0
with open(out1, 'w') as f:
    for i in range(0, len(initial_groups)):
        smm = 0
        light_levels_sum = 0
        cntt = 0
        for node in initial_groups[i]:
            f.write(node + ' ' + str(no_of_desired_groups - i - 1) + '\n')
            smm = smm + analysis_graph[node].duration
            if int(analysis_graph[node].level) > 651 and int(analysis_graph[node].level) < 705:
                light_levels_sum = light_levels_sum + analysis_graph[node].duration
            count = count + 1
            cntt = cntt + 1
        print( str(no_of_desired_groups - i - 1) + ': ' + str(cntt) +', '+ str(smm) + ', ' + str(light_levels_sum))
