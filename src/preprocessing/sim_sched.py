import utils
import nodeProps
import heapq
import copy
from random import shuffle

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/'

in1 = io_folder_path + 'part_1_39_src_sink.dot'
in2 = io_folder_path + 'timeline_step303.json'
in3 = io_folder_path + 'ver_grouper_placement_e_nc.place'
in4 = io_folder_path + 'tensors_sz.txt'

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
                graph[splits[0]].append(splits[1])
            else:
                graph[splits[0]] = [splits[1]]

graph[sink_node_name] = []

# getting time (weight) info for nodes
analysis_graph = utils.read_profiling_file(in2, True)

nodes_parts = {}
nodes_parts_list = []

with open(in3) as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        nodes_parts_list.append(line)

shuffle(nodes_parts_list)

for node_part in nodes_parts_list:
    splits = node_part.split(' ')
    nodes_parts[splits[0]] = int(splits[1])

all_tensors_sizes = {}
# get tensors sizes
with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        all_tensors_sizes[splitted[0]] = int(splitted[1])

tensors_sizes = {}

for node in nodes_parts.keys():
    if node in all_tensors_sizes:
        tensors_sizes[node] = all_tensors_sizes[node]
    else:
        tensors_sizes[node] = 1


for node, node_props in all_nodes.items():
    if node in analysis_graph:
        analysis_graph[node].parents = node_props.parents
        analysis_graph[node].children = node_props.children
    else:
        analysis_graph[node] = node_props
    analysis_graph[node].start_time = 0
    analysis_graph[node].end_time = node_props.duration

sink_node_name = 'snk'
source_node_name = 'src'
nodes_parts[sink_node_name] = 0
analysis_graph[sink_node_name].duration = 0
nodes_parts[source_node_name] = 0
comm_transfer_rate = 1000000 / (9 * 1024 * 1024 * 1024)
comm_latency = 20

def simulate_scheduling(nodes_parts):

    visited = {}
    max_end_times = [0, 0]
    analysis_graph[source_node_name].duration = 0
    traversal_queueu = []
    heapq.heappush(traversal_queueu, (0, source_node_name))

    tensor_communicated_to_part = {}

    for node in analysis_graph.keys():
        analysis_graph[node].start_time = 0
        analysis_graph[node].end_time = analysis_graph[node].duration

    while(traversal_queueu):
        current_node = heapq.heappop(traversal_queueu)[1]
        current_node_part = nodes_parts[current_node]
        current_node_end_time = analysis_graph[current_node].end_time
        if not current_node in visited:
            visited[current_node] = 1
            for adj_node in graph[current_node]:
                all_parents_visited = True
                last_finishing_parent_in_other_parts = 0
                for parent_node in analysis_graph[adj_node].parents:
                    if parent_node in graph.keys():
                        if parent_node not in visited:
                            all_parents_visited = False
                        if nodes_parts[parent_node] != nodes_parts[adj_node]:

                            if parent_node not in tensor_communicated_to_part.keys(
                            ) or nodes_parts[adj_node] not in tensor_communicated_to_part[parent_node]:
                                comm_time = (tensors_sizes[parent_node] * comm_transfer_rate + comm_latency)
                            else:
                                comm_time = 0
                            parent_finishing_time = analysis_graph[parent_node].end_time + comm_time
                            
                            if parent_finishing_time > last_finishing_parent_in_other_parts:
                                last_finishing_parent_in_other_parts = parent_finishing_time

                if all_parents_visited:
                    if nodes_parts[adj_node] == current_node_part:
                        analysis_graph[adj_node].start_time = max(max_end_times[current_node_part], last_finishing_parent_in_other_parts)
                        analysis_graph[adj_node].end_time = analysis_graph[adj_node].start_time + \
                            analysis_graph[adj_node].duration
                    else:
                        if current_node not in tensor_communicated_to_part.keys(
                        ) or nodes_parts[adj_node] not in tensor_communicated_to_part[current_node]:
                            comm_time = (tensors_sizes[current_node] * comm_transfer_rate + comm_latency)
                        else:
                            comm_time = 0

                        analysis_graph[adj_node].start_time = max(current_node_end_time + comm_time, max_end_times[nodes_parts[adj_node]])
                        analysis_graph[adj_node].end_time = analysis_graph[adj_node].start_time + \
                            analysis_graph[adj_node].duration

                        if current_node in tensor_communicated_to_part:
                            tensor_communicated_to_part[current_node].append(
                                nodes_parts[adj_node])
                        else:
                            tensor_communicated_to_part[current_node] = [
                                nodes_parts[adj_node]]

                    if analysis_graph[adj_node].end_time > max_end_times[nodes_parts[adj_node]]:
                        max_end_times[nodes_parts[adj_node]
                                        ] = analysis_graph[adj_node].end_time

                    heapq.heappush(
                        traversal_queueu, (analysis_graph[adj_node].start_time, adj_node))

    return (analysis_graph[sink_node_name].start_time)


def placement_best_first_search(nodes_parts, base_time):
    min_time = base_time
    best_place = []
    traversal_queueu = []
    indices_to_nodes = {}
    indx = 0
    for node in nodes_parts:
        indices_to_nodes[indx] = node
        indx = indx + 1
    search_tree_level = 0
    parts = list(nodes_parts.values())
    heapq.heappush(traversal_queueu, (base_time, [parts, search_tree_level] ) )
    iterations = 0
    search_tree_max_depth = len(nodes_parts)

    while(iterations < 1000):
        iterations = iterations + 1
        current_state = heapq.heappop(traversal_queueu)
        search_tree_level = (current_state[1])[1]
        if search_tree_level < search_tree_max_depth:
            if indices_to_nodes[search_tree_level] == 'src':
                continue 

            state_0_parts = copy.deepcopy((current_state[1])[0])
            state_0_parts[ search_tree_level ] = 0
            indx = 0
            for node in nodes_parts.keys():
                nodes_parts[node] = state_0_parts[indx]
                indx = indx + 1
            state_0_time = simulate_scheduling(nodes_parts)
            if state_0_time < min_time:
                min_time = state_0_time
                best_place = state_0_parts

            state_1_parts = copy.deepcopy((current_state[1])[0])
            state_0_parts[ search_tree_level ] = 1
            indx = 0
            for node in nodes_parts.keys():
                nodes_parts[node] = state_1_parts[indx]
                indx = indx + 1
            state_1_time = simulate_scheduling(nodes_parts)
            if state_1_time < min_time:
                min_time = state_1_time
                best_place = state_1_parts

            heapq.heappush(traversal_queueu, (state_0_time, (state_0_parts, search_tree_level + 1) ) )
            heapq.heappush(traversal_queueu, (state_1_time, (state_1_parts, search_tree_level + 1) ) )

    return [min_time, best_place]
            


print(simulate_scheduling(nodes_parts))
best_place = placement_best_first_search(nodes_parts, simulate_scheduling(nodes_parts) )
indx = 0
with open(io_folder_path + 'best_place.txt', 'w') as f:
    f.write(str(best_place[0]) + '\n')
    for node in nodes_parts.keys():
        f.write( node + ' ' + str(best_place[1][indx]) + '\n' )
        indx = indx + 1
#print( placement_best_first_search(nodes_parts, simulate_scheduling(nodes_parts) ) )

part_0_time = 0
part_1_time = 0
for node in graph:
    if nodes_parts[node] == 0:
        part_0_time = part_0_time + analysis_graph[node].duration
    else:
        part_1_time = part_1_time + analysis_graph[node].duration

print('part 0 total load is: ' + str(part_0_time))
print('part 1 total load is: ' + str(part_1_time))


""" sink_node_name = 'snk'
nodes_parts[sink_node_name] = 0
visited = {}
max_end_times = [0, 0] 
source_node_name = 'src'
nodes_parts[source_node_name] = 0
traversal_queueu = []
heapq.heappush(traversal_queueu, (0, source_node_name))

comm_transfer_rate = 1000000 / (9 * 1024 * 1024 * 1024)
cntt = 0
while(traversal_queueu):
    current_node = heapq.heappop(traversal_queueu)[1]
    current_node_part = nodes_parts[current_node]
    current_node_end_time = analysis_graph[current_node].end_time
    if not current_node in visited:
        visited[current_node] = 1
        for adj_node in graph[current_node]:
            all_parents_visited = True
            if current_node_end_time > max_end_times[current_node_part]:
                max_end_times[current_node_part] = current_node_end_time 
            for parent_node in analysis_graph[adj_node].parents:
                if parent_node not in visited and parent_node in graph.keys():
                    all_parents_visited = False
            if all_parents_visited:
                print(adj_node + ' ' + str(analysis_graph[adj_node].end_time))   
                heapq.heappush(traversal_queueu,(analysis_graph[adj_node].end_time, adj_node))
 """
