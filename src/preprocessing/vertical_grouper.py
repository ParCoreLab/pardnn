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

""" # input files
in1 = io_folder_path + 'part_65_103_n_src_sink.dot' #'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step303_dfs.json' #'timeline_step303_dfs.json'
in3 = io_folder_path + 'part_65_103_n_src_sink_nodes_levels.txt' #'part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_part_65_103_n_src_sink_nodes_levels.txt' #'rev_part_8_1799_src_sink_nodes_levels.txt' """

in1 = io_folder_path + 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step303_f_v.json'
in3 = io_folder_path + 'part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_part_8_1799_src_sink_nodes_levels.txt'

#output file
out1 = io_folder_path + 'ver_grouper_placement.place'

#grouper parameters
path_length_threshold = 8
group_weight_threshold = 250
no_of_desired_groups = 2

reverse_levels = {}
#get nodes levels
with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            reverse_levels[node_and_level[0]] = node_and_level[1]

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

            all_nodes[splits[1]].parents.append(all_nodes[splits[0]])
            all_nodes[splits[1]].children.append(all_nodes[splits[1]])
 
            if splits[0] in graph.keys():
                heapq.heappush(graph[splits[0]],( int(reverse_levels[splits[1]]), splits[1]))
            else:
                graph[splits[0]] = []
                heapq.heappush(graph[splits[0]],(int(reverse_levels[splits[1]]), splits[1]))
                
graph[sink_node_name] = []


#change adjacents priority queues to lists
for node, adjacents in graph.items():
    ordered_adjacents_list = []
    while adjacents:
        ordered_adjacents_list.append(heapq.heappop(adjacents)[1])
    graph[node] = ordered_adjacents_list


#getting time (weight) info for nodes
analysis_graph = utils.read_profiling_file(in2, True)

for node, node_props in all_nodes.items():
    if node in analysis_graph:
        analysis_graph[node].parents = node_props.parents
        analysis_graph[node].children = node_props.children
    else:
        analysis_graph[node] = node_props


#extracting all vertical paths in the graph
source_node_name = 'src'
traversal_stack = [source_node_name]
paths = []
current_path = []
visited = {}
paths_weights = []
current_path_weight = 0

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
            current_path.append(adj_node)
            paths.append(copy.deepcopy(current_path))
            poped_node = current_path.pop()
            current_path_weight = current_path_weight - analysis_graph[poped_node].duration
            paths_weights.append(current_path_weight)
            current_path = []
            current_path_weight = 0


#print(len(paths))
      

#get nodes levels
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            analysis_graph[node_and_level[0]].level = node_and_level[1]         


#getting initial groups
initial_groups = []
groups_weights = []
initial_groups.append(copy.deepcopy(paths[0])) 
groups_weights.append(paths_weights[0])
composite_groups_indices = {}

for i in range(1, len(paths)):
    current_path = paths[i]
    current_path_weight = paths_weights[i]
    this_path_joined = False
    common_node = current_path.pop()
    if current_path_weight <= group_weight_threshold and len(current_path) <= path_length_threshold:
        for j in range(0, len(initial_groups)):
            if common_node in initial_groups[j]:
                initial_groups[j] = initial_groups[j] + current_path
                groups_weights[j] = groups_weights[j] + current_path_weight
                this_path_joined = True
                composite_groups_indices[j] = j
                break
    else:
        this_path_joined = True
        initial_groups.append(copy.deepcopy(current_path))
        groups_weights.append(current_path_weight)
    if not this_path_joined:
        current_path.append(common_node)
        paths.pop(i)
        paths.append(current_path)
        i = i - 1



#print(len(initial_groups))

#parts work distribution over levels
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

#merging the groups
while len(initial_groups) > no_of_desired_groups:
    indx_min = 0
    min_group_weight = groups_weights[indx_min]
    for i in range(1, len(groups_weights)):
        if groups_weights[i] < min_group_weight:
            min_group_weight = groups_weights[i]
            indx_min = i
    
    min_weight_group = initial_groups[indx_min]
    src_min_level = min_levels[indx_min]
    src_max_level = max_levels[indx_min]
    min_sum_in_targeted_levels = 1000000000
    merge_destination_index = 0

    for i in range(0, len(initial_groups)):
        if i == indx_min:
            continue
        
        sum_in_targeted_levels = 0
        current_min_level = max(src_min_level, min_levels[i])
        current_max_level = min(src_max_level, max_levels[i])

        for level in range(current_min_level, current_max_level + 1):
            if level in tasks_per_levels[i].keys() and level in tasks_per_levels[indx_min].keys():
                sum_in_targeted_levels = sum_in_targeted_levels + tasks_per_levels[i][level]
        if sum_in_targeted_levels < min_sum_in_targeted_levels:
            min_sum_in_targeted_levels = sum_in_targeted_levels
            merge_destination_index = i
        #if src_min_level < 659 and src_max_level > 679:
         #   print(sum_in_targeted_levels)

    merge_min_level = min(min_levels[indx_min], min_levels[merge_destination_index])
    merge_max_level = max(max_levels[indx_min], max_levels[merge_destination_index])

    to_be_merged_group = initial_groups[indx_min]
    groups_weights[merge_destination_index] = groups_weights[merge_destination_index]  + groups_weights[indx_min]
    groups_weights.pop(indx_min)
    initial_groups[merge_destination_index] = initial_groups[merge_destination_index] + to_be_merged_group
    initial_groups.pop(indx_min)
    min_levels[merge_destination_index] = merge_min_level
    max_levels[merge_destination_index] = merge_max_level
    min_levels.pop(indx_min)
    max_levels.pop(indx_min)
    merge_src_levels_tasks = tasks_per_levels[indx_min]

    for level, tasks_sum in merge_src_levels_tasks.items():
        if level in tasks_per_levels[merge_destination_index].keys():
            tasks_per_levels[merge_destination_index][level] = tasks_per_levels[merge_destination_index][level] + tasks_sum
        else:
            tasks_per_levels[merge_destination_index][level] = tasks_sum
    tasks_per_levels.pop(indx_min)

count = 0
with open(out1, 'w') as f:
    for i in range(0, len(initial_groups)):
        for node in initial_groups[i]:
            f.write(node + ' ' + str(i) + '\n')
            count = count + 1