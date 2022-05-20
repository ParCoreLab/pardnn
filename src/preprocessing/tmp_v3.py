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
import numpy as np
import statistics

# folder containing the work files
io_folder_path = utils.io_folder_path
network_app = utils.network_app
in1 = io_folder_path + 'graph.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'weights.txt'
in3 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'
in5 = io_folder_path + 'costs.txt'
in9 = io_folder_path + 'no_ops.txt'
in10 = io_folder_path + 'ref_nodes.txt'
in11 = io_folder_path + 'var_nodes.txt'
in12 = io_folder_path + 'vanilla_cleaned.place'

# output file
out1 = io_folder_path + 'placement_8.place'

# grouper parameters
no_of_desired_groups = 8 
memory_limit_per_group = 30 * 1024 * 1024 * 1024

#tst
comm_latency = 45
average_tensor_size_if_not_provided = 1
comm_transfer_rate = 1.0 / (130000)

# will contain the graph as an adgacency list
graph = {}
rev_graph = {}
all_nodes = {}
sink_node_name = 'snk'
source_node_name = 'src'
graph[sink_node_name] = []
rev_graph[source_node_name] = []

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

            if splits[0] in graph.keys():
                graph[splits[0]].append(splits[1])
            else:
                graph[splits[0]] = [splits[1]]
            
            if splits[1] in rev_graph.keys():
                rev_graph[splits[1]].append(splits[0])
            else:
                rev_graph[splits[1]] = [splits[0]]

no_op_nodes = {}
with open(in9, 'r') as f:
    for line in f:
        no_op_nodes[utils.clean_line(line)] = 1

# getting time (weight) info for nodes
analysis_graph = utils.read_profiling_file_v2(in2)

sudo_nodes = {}
for node, node_props in all_nodes.items():
    if node not in analysis_graph:
        analysis_graph[node] = node_props
    if node.startswith('^'):
        sudo_nodes[node] = 1

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
        edge_weight = float(tensor_size) * comm_transfer_rate + comm_latency
        edges_weights[tensor_name] = {}
        if tensor_name in graph:
            for adj_node in graph[tensor_name]:
                if adj_node in no_op_nodes or adj_node in sudo_nodes or adj_node == sink_node_name:
                    edges_weights[tensor_name][adj_node] = comm_latency
                else:
                    edges_weights[tensor_name][adj_node] = edge_weight 

ref_nodes = {}
with open(in10, 'r') as f:
    for line in f:
        ref_nodes[utils.clean_line(line)] = 1

var_nodes = {}
with open(in11, 'r') as f:
    for line in f:
        var_nodes[utils.clean_line(line)] = 1

vanilla_placement = {}
with open(in12, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line).lower()
        splits = line.split(' ')
        vanilla_placement[splits[0]] = splits[1]

t0 = time.time()

# get_node_average_weight
total_nodes_weight = 0
for node, node_props in analysis_graph.items():
    total_nodes_weight = total_nodes_weight + node_props.duration

average_node_weight = total_nodes_weight/len(analysis_graph)

for node in all_nodes:
    if not node in tensors_sizes:
        tensors_sizes[node] = 0
        edges_weights[node] = {}
        for adj_node in graph[node]:
            edges_weights[node][adj_node] = float(comm_latency)

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

#nodes bottom levels
def get_nodes_weighted_levels(graph, edges_weights, src_nodes = None, previosly_visited = [], grouped = False, nodes_groups = {}, is_rev = True, _nodes_in_degrees = rev_nodes_in_degrees):
    # getting the sources of the graph to start the topological traversal from them
    graph_keys = {}
    nodes_weighted_levels={}
    tmp_nodes_in_degrees = copy.deepcopy(_nodes_in_degrees)
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
        current_node_duration = analysis_graph[current_node].duration
        for adj_node in adj_nodes:
            if adj_node not in previosly_visited:
                #this is correct, might seem confusing, remember we are working with the reversed graph
                if is_rev:
                    edge_weight = edges_weights[adj_node][current_node]
                else:
                    edge_weight = edges_weights[current_node][adj_node]
                if grouped:
                    if nodes_groups[adj_node] == nodes_groups[current_node]:
                        edge_weight = 0
                new_level = current_node_level + edge_weight + (analysis_graph[adj_node].duration if is_rev else current_node_duration)
                tmp_nodes_in_degrees[adj_node] -= 1
                if nodes_weighted_levels[adj_node] < new_level:
                    nodes_weighted_levels[adj_node] = new_level
                if tmp_nodes_in_degrees[adj_node] == 0:
                    traversal_queueu.put(adj_node)
    return nodes_weighted_levels

# extracting all vertical paths in the graph
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

nodes_weighted_levels = get_nodes_weighted_levels(rev_graph, edges_weights, src_nodes=[sink_node_name])


for node, weighted_level in nodes_weighted_levels.items():
    heapq.heappush(free_nodes, (-weighted_level, node))

tmp_rev_nodes_in_degrees = copy.deepcopy(rev_nodes_in_degrees)
while free_nodes:
    current_node = heapq.heappop(free_nodes)[1]
    while current_node in visited and free_nodes:
        current_node = heapq.heappop(free_nodes)[1]

    while current_node !='' and current_node not in visited:
        current_path.append(current_node)
        current_path_weight = current_path_weight + \
            analysis_graph[current_node].duration
        if len(current_path) > 1:
            current_path_weight_with_comm = current_path_weight_with_comm + \
                analysis_graph[current_node].duration + edges_weights[current_path[-2]][current_node]
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
        if current_node != '':
            current_path_weight_with_comm += edges_weights[current_path[-1]][current_node]
        paths.append(current_path)
        groups_weights.append(current_path_weight)
        paths_lengths.append(len(current_path))
        #if len(paths) <= no_of_desired_groups or current_path_weight_with_comm >= groups_weights[0]/200:
        nodes_weighted_levels = get_nodes_weighted_levels(graph = tmp_rev_graph, edges_weights = edges_weights, src_nodes= src_nodes, previosly_visited= visited)
        free_nodes = []
        for node, weighted_level in nodes_weighted_levels.items():
            heapq.heappush(free_nodes, (-weighted_level, node))

        for node in current_path:
            del tmp_rev_nodes_in_degrees[node]
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

print('paths obtained: ' + str( time.time() - t0 ))
t0 = time.time()
# which node is in which path
nodes_paths_mapping[source_node_name] = num_paths - 1
nodes_paths_mapping[sink_node_name] = num_paths - 1
for i in range(0, num_paths):
    for node in paths[i]:
        nodes_paths_mapping[node] = i

# get max potential of paths
groups_parents = {}
paths_max_potential = copy.deepcopy(groups_weights)
levels_work_sums = {}
nodes_weighted_levels = get_nodes_weighted_levels(graph= graph, grouped= True, edges_weights = edges_weights, nodes_groups= nodes_paths_mapping, \
    is_rev= False, _nodes_in_degrees= nodes_in_degrees, src_nodes= [source_node_name])
for node, level in nodes_weighted_levels.items():
    if level not in levels_work_sums:
        levels_work_sums[level] = 0
    levels_work_sums[level] += analysis_graph[node].duration

levels = levels_work_sums.keys() 
work_sums = levels_work_sums.values()

levels, work_sums = (list(t) for t in zip(
        *sorted(zip(levels, work_sums))))

for i in range(1, len(work_sums)):
    work_sums[i] += work_sums[i - 1]

levels_indices_map = {}
current_level = 0
for level in levels:
    if level not in levels_indices_map:
        levels_indices_map[level] = current_level 
        current_level += 1

paths_comms = []
paths_ranges = []
path_ranges_subtract = []
paths_parents = []
for i in range(0, len(paths)):
    path = paths[i]
    if path[0] == source_node_name:
        continue

    path_comm = 0
    path_last_src = 0
    path_first_snk = math.inf
    path_head = path[0]
    path_tail = path[-1]
    path_parents = {}

    first_child_comp = 0
    max_child_comm = 0
    for child in graph[path_tail]:
        child_path = nodes_paths_mapping[child]
        if child_path not in path_parents:
            path_parents[child_path] = 0
        path_parents[child_path] = max(path_parents[child_path], edges_weights[path_tail][child])
        max_child_comm = max(max_child_comm, edges_weights[path_tail][child])
        if nodes_weighted_levels[child] < path_first_snk:
            path_first_snk = nodes_weighted_levels[child]
            path_parents[child_path] = edges_weights[path_tail][child]
            first_child_comp = analysis_graph[child].duration

    path_comm += max_child_comm

    last_parent_comp = 0
    for parent in rev_graph[path_head]:
        path_comm += edges_weights[parent][path_head]
        if nodes_weighted_levels[parent] > path_last_src:
            path_last_src = nodes_weighted_levels[parent]
            parent_path = nodes_paths_mapping[parent]
            last_parent_comp = analysis_graph[parent].duration
            if parent_path not in path_parents:
                path_parents[parent_path] = 0
            path_parents[parent_path] += edges_weights[parent][path_head]

    path_ranges_subtract.append(last_parent_comp + first_child_comp)

    max_comm = 0
    max_comm_indx = 0
    for path_parent, comm in path_parents.items():
        if comm > max_comm:
            max_comm = comm
            max_comm_indx = path_parent

    paths_parents.append(max_comm_indx)

    paths_comms.append(path_comm)
    paths_ranges.append([path_last_src, path_first_snk])

# get the average path length
after_heavy_paths_count = 0
after_heavy_paths_lengths = 0
for path in paths:
    after_heavy_paths_count = after_heavy_paths_count + 1
    after_heavy_paths_lengths = after_heavy_paths_lengths + len(path)

average_path_len = round(after_heavy_paths_lengths / after_heavy_paths_count)

print('averge path len: ' + str(average_path_len))

# getting initial groups
initial_groups = copy.deepcopy(paths)
initial_groups_indices = [1] * num_paths
path_joined_group = {}
paths_become_groups = {}
for i in range(0, len(paths) - 1 ):
    if i in paths_become_groups:
        continue
    path = paths[i]
    path_parent = paths_parents[i]
    path_max_potential = ( work_sums[ levels_indices_map[paths_ranges[i][1]] ] - work_sums[ levels_indices_map[paths_ranges[i][0]] ] ) - ( groups_weights[i]  + path_ranges_subtract[i] )
    if paths_comms[i] >= path_max_potential:
        initial_groups_indices[i] = 0
        groups_weights[path_parent] += groups_weights[i]
        initial_groups[path_parent] += initial_groups[i]

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
min_levels = [1000000]*len(initial_groups)
nodes_initail_groups = {}
for i in range(0, len(initial_groups)):
    tasks_per_levels.append(collections.OrderedDict())
    current_group = initial_groups[i]
    for node in current_group:
        nodes_initail_groups[node] = i
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

print('Initial merging is done: ' + str( time.time() - t0 ))
t0 = time.time()
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

    return BITTree

# nodes levels calculation
def get_nodes_levels_dsc(_graph, edges_weights, bottom_levels, nodes_clusters = {}):
    # getting the sources of the graph to start the topological traversal from them
    nodes_levels = {}
    graph_keys = {}
    nodes_in_degrees = {}
    if bottom_levels:
        for node in graph:
            nodes_in_degrees[node] = len(graph[node])
    else:
        for node in rev_graph:
            nodes_in_degrees[node] = len(rev_graph[node])

    for graph_key in _graph.keys():
        graph_keys[graph_key] = 0

    for adj_nodes in _graph.values():
        for node in adj_nodes:
            if node in graph_keys:
                graph_keys[node] = 1

    traversal_queue = queue.Queue()
    for node, source_node in graph_keys.items():
        if source_node == 0:
            if bottom_levels:
                nodes_levels[node] = analysis_graph[node].duration
            else:
                nodes_levels[node] = 0
            traversal_queue.put(node)

    traversal_queue = []
    heapq.heappush(traversal_queue, (0, source_node_name))
    # start the traversal
    while traversal_queue:
        current_node = heapq.heappop(traversal_queue)
        current_node = current_node[1]
        if current_node in _graph:
            adj_nodes = _graph[current_node]
        else:
            adj_nodes = []
        current_node_weight = analysis_graph[current_node].duration
        current_node_level = nodes_levels[current_node]
        for adj_node in adj_nodes:
            edge_weight = edges_weights[current_node][adj_node]
            if len(nodes_clusters) > 0:
                if nodes_clusters[current_node] == nodes_clusters[adj_node]:
                    edge_weight = 0
            if bottom_levels:
                new_level = current_node_level + \
                    + edge_weight \
                    + analysis_graph[adj_node].duration
            else:
                new_level = current_node_level + \
                    + edge_weight \
                    + current_node_weight
            if adj_node not in nodes_levels or nodes_levels[adj_node] < new_level:
                nodes_levels[adj_node] = new_level
            nodes_in_degrees[adj_node] -= 1
            if nodes_in_degrees[adj_node] == 0:
                heapq.heappush(traversal_queue, (nodes_levels[current_node], adj_node))

    return nodes_levels

def calc_finish_time(no_of_clusters, nodes_clusters, nodes_levels, graph):
    clusters_ready_times = [0] * no_of_clusters
    nodes_ready_times = {}
    nodes = nodes_levels.keys()
    levels = nodes_levels.values()
    levels, nodes = (list(t) for t in zip(*sorted(zip(levels, nodes))))
    nodes_ready_times['src'] = 0
    for node in nodes:
        node_cluster = nodes_clusters[node]
        ready_time = clusters_ready_times[node_cluster]
        nodes_ready_times[node] = ready_time
        for parent in rev_graph[node]:
            if nodes_clusters[parent] != node_cluster:
                time_from_parent = nodes_ready_times[parent]
                if time_from_parent > nodes_ready_times[node]:
                    nodes_ready_times[node] = time_from_parent

        clusters_ready_times[node_cluster] = max(ready_time, nodes_ready_times[node] + analysis_graph[node].duration) 

    return max(clusters_ready_times)

nodes_levels = get_nodes_levels_dsc(graph, edges_weights, False, nodes_clusters= nodes_initail_groups)
nodes_final_clusters = {}
clusters_loads = [0] * no_of_desired_groups
clusters_start_times = {}
for node in all_nodes.keys():
    current_group = nodes_initail_groups[node]
    if current_group not in clusters_start_times:
        clusters_start_times[current_group] = math.inf
    clusters_start_times[current_group] = min(clusters_start_times[current_group], nodes_levels[node])

clusters = clusters_start_times.keys()
start_times = clusters_start_times.values()

start_times, groups_weights, clusters = (list(t) for t in zip(
    *sorted(zip(start_times, groups_weights, clusters))))

#glb
for cluster in clusters:
    group = initial_groups[cluster]
    min_load = math.inf
    min_loaded = -1
    for i in range(0, no_of_desired_groups):
        if clusters_loads[i] < min_load:
            min_load = clusters_loads[i]
            min_loaded = i
    
    for node in group:
        nodes_final_clusters[node] = min_loaded
        clusters_loads[min_loaded] += groups_weights[cluster]
#end glb

makespan = calc_finish_time(len(nodes_final_clusters), nodes_final_clusters, nodes_levels, graph)

print('finish time: ' + str(makespan))

print('Final merging is done: ' + str( time.time() - t0 ))
t0 = time.time()      

#nodes_levels = get_nodes_levels_dsc(graph,edges_weights, False, nodes_final_clusters)

# get nodes in degrees for the topological sort
nodes_in_degrees = {}
for node in all_nodes:
    if node in rev_graph:
        nodes_in_degrees[node] = len(rev_graph[node])
    else:
        nodes_in_degrees[node] = 0

traversal_queue = []
heapq.heappush(traversal_queue,(0, source_node_name))
groups_times_till_now = [0] * no_of_desired_groups
nodes_levels_scheduled = {}
for node in all_nodes:
  nodes_levels_scheduled[node] = 0
while traversal_queue:
  current_node = heapq.heappop(traversal_queue)
  node_level = current_node[0]
  node_name = current_node[1]
  current_node_group = nodes_final_clusters[node_name]
  groups_times_till_now[current_node_group] = max(groups_times_till_now[current_node_group], node_level + analysis_graph[node_name].duration)
  for adj in graph[node_name]:
    adj_node_group = nodes_final_clusters[adj]
    new_level = groups_times_till_now[adj_node_group]
    if adj_node_group != current_node_group:
      new_level = max(nodes_levels_scheduled[adj], \
        max(new_level, node_level + analysis_graph[node_name].duration + edges_weights[node_name][adj]))
    
    nodes_levels_scheduled[adj] = new_level
    nodes_in_degrees[adj] -= 1
    if nodes_in_degrees[adj] == 0:    
      heapq.heappush(traversal_queue, (new_level, adj))

#makespan = calc_finish_time(len(nodes_final_clusters), nodes_final_clusters, nodes_levels, graph)
        
print('finish time: ' + str(nodes_levels_scheduled[sink_node_name]))

changed = {}
for node_name in graph.keys():
  if node_name not in var_nodes or node_name in changed:
    continue
    
  to_visit = []
  node_part = nodes_final_clusters[node_name]
  to_visit.append(node_name)
  while len(to_visit) > 0:
    current_node_name = to_visit.pop(0)
    changed[current_node_name] = True
    for adj_node in graph[current_node_name]:
      #if adj_node == 'generator/e3d-lstm/e3d0/conv3d_1/kernel/Assign'.lower():
      if adj_node in ref_nodes:
        nodes_final_clusters[adj_node] = node_part
        for rev_adj_node in rev_graph[adj_node]:
          if (rev_adj_node not in changed and rev_adj_node in var_nodes):
            nodes_final_clusters[rev_adj_node] = node_part
            to_visit.append(rev_adj_node)
            changed[rev_adj_node] = True
            
with open(out1, 'w') as f:
    for node, cluster in nodes_final_clusters.items():
        if node in vanilla_placement and vanilla_placement[node] == '-1':
            f.write(node + ' ' + str(-1) + '\n')
        else:
            f.write(node + ' ' + str(cluster) + '\n')