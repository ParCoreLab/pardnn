import utils

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/'

""" # input files
in1 = io_folder_path + 'inc_A_dot_low.dot'
in2 = io_folder_path + 'nodes_levels_low.txt'
in3 = io_folder_path + 'vanilla_cleaned_low.place'
in4 = io_folder_path + 'timeline_step303_25.json'
in5 = io_folder_path + 'colocation_low.txt'
in6 = io_folder_path + 'tensors_sz_32_low.txt' """

# input files
in1 = io_folder_path + 'resnet_src_sink_low.dot'
in2 = io_folder_path + 'resnet_src_sink_nodes_levels_low.txt'
in3 = io_folder_path + 'ver_grouper_placement_e_nc.place'  # 'mixed_placement_v_part_nc.place'
in4 = io_folder_path + 'timeline_step0_30_low.json'
in5 = io_folder_path + 'colocation_32_low.txt'
in6 = io_folder_path + 'tensors_sz_32_low.txt'


# output files

out1 = io_folder_path + 'first_layes.txt'
out2 = io_folder_path + 'the_rest_of_the_layes.txt'
out3 = io_folder_path + 'from_last_to_first.txt'
out4 = io_folder_path + 'in_blacklist.txt'
out5 = io_folder_path + 'nodes_start_times.txt'

graph = {}

# constructing the graph and initializing the nodes levels from the dot file
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            if nodes[0] in graph:
                graph[nodes[0]].append(nodes[1])
            else:
                graph[nodes[0]] = [nodes[1]]

nodes_levels = {}
# fill the analysis graph nodes levels
with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        node_and_level = line.split("::")
        nodes_levels[node_and_level[0]] = node_and_level[1]


first_layers_nodes = []
rest_layers_nodes = []
from_last_to_first = {}
total_edges = 0
for node, level in nodes_levels.items():
    level = int(level)
    if level <= 7:
        first_layers_nodes.append(node)
        if level == 7 and node in graph.keys():
            total_edges = total_edges + len(graph[node])
            from_last_to_first[node] = graph[node]
    else:
        rest_layers_nodes.append(node)
        if level == 68 and node in graph:
            print(graph[node])
            for adj in graph[node]:
                print(nodes_levels[adj])


nodes_devices = {}
blackList = {}
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line).strip()
        splts = line.split(" ")
        nodes_devices[splts[0]] = splts[1]
        if splts[0] in first_layers_nodes and int(splts[1]) == 4:
            blackList[splts[0]] = splts[1]

cntt = [0]*8

for node in first_layers_nodes:
    if node in nodes_devices and nodes_devices[node] == '0' and node in graph:
        for adj in graph[node]:
            if int(nodes_levels[adj]) < 7 and nodes_devices[node] == '0':
                cntt[int(nodes_levels[adj])] = cntt[int(nodes_levels[adj])] + 1


analysis_graph = utils.read_profiling_file(in4, False)

nodes = []
times = []

for node, node_properties in analysis_graph.items():
    nodes.append(node)
    times.append(int(node_properties.start_time))

times, nodes = (list(t) for t in zip(
    *sorted(zip(times, nodes))))


tensors_sizes = {}
# get tensors sizes
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensors_sizes[splitted[0]] = splitted[1]

print(len(tensors_sizes))
with open(out5, 'w') as f:
    for i in range(0, len(nodes)):
        if times[i] >= 0:
            curr_tensor_size = -1
            if nodes[i] in tensors_sizes:
                curr_tensor_size = tensors_sizes[nodes[i]]
            if nodes[i] in nodes_levels:
                f.write(nodes[i] + ':S:' + str(times[i]) + ':D:' +
                        str(analysis_graph[nodes[i]].duration) + ':L:' + nodes_levels[nodes[i]] + ':P:' + (str(nodes_devices[nodes[i]]) if nodes[i] in nodes_devices else '-1') + ':T:' + str(curr_tensor_size) + '\n')
            else:
                f.write(nodes[i] + ':S:' + str(times[i]) + ':D:' +
                        str(analysis_graph[nodes[i]].duration) + ':L:-1' + ':P:' + (str(nodes_devices[nodes[i]]) if nodes[i] in nodes_devices else '-1') + ':T:' + str(curr_tensor_size) + '\n')
        else:
            f.write(nodes[i] + ':L:' + nodes_levels[nodes[i]] + ':P:' + (str(nodes_devices[nodes[i]]) if nodes[i] in nodes_devices else '-1') + ':T:' + str(curr_tensor_size) + '\n')


    for i in range(0, len(nodes)):
        if times[i] >= 0:
            curr_tensor_size = -1
            if nodes[i] in tensors_sizes:
                curr_tensor_size = tensors_sizes[nodes[i]]
            if nodes[i] in nodes_levels:
                f.write(nodes[i] + ':S:' + str(times[i]) + ':D:' +
                        str(analysis_graph[nodes[i]].duration) + ':L:' + nodes_levels[nodes[i]] + ':T:' + str(curr_tensor_size) + '\n')
            else:
                f.write(nodes[i] + ':S:' + str(times[i]) + ':D:' +
                        str(analysis_graph[nodes[i]].duration) + ':L:-1:T:' + str(curr_tensor_size) + '\n')


sum = 0
for node in rest_layers_nodes:
    if node != 'src' and node != 'snk' and nodes_devices[node] == '4':
        #sum = sum + int(analysis_graph[node].duration)
        sum = sum + 1


print(sum)
print(sum/len(rest_layers_nodes))

distinct_nodes = {}
cpu_to_gpu = []
gpu_to_cpu = []
cpu_to_cpu = []
sum = 0
for node in first_layers_nodes:
    if node in graph:
        for adj_node in graph[node]:
            if not adj_node.startswith('^') and adj_node in rest_layers_nodes:
                if adj_node in analysis_graph and not adj_node in distinct_nodes:
                    sum = sum + int(analysis_graph[adj_node].duration)
                distinct_nodes[adj_node] = 1

#sum = len(distinct_nodes)
print(len(graph))
cntt = 0
for node, level in nodes_levels.items():
    #in graph and len(graph[node]) <= 100000 and node
    if int(level) > 8 and node in analysis_graph:
        cntt += analysis_graph[node].duration
print('multi child: ' + str(cntt))

cntt2 = 0
for node in analysis_graph:
    cntt2 += analysis_graph[node].duration
print('all nodes: ' + str(cntt2))
print('percentage:' + str(cntt / cntt2))

print(sum)


with open(out1, 'w') as f:
    for node in first_layers_nodes:
        if not node.startswith('^'):
            f.write(node + '\n')

with open(out2, 'w') as f:
    for node in rest_layers_nodes:
        if not node.startswith('^'):
            f.write(node + '\n')

with open(out3, 'w') as f:
    f.write('total edges from layer 7 to layer 8: ' +
            str(total_edges) + ' edges, they are:\n\n')
    for node, adj in from_last_to_first.items():
        f.write(node + ' -> ' + str(adj) + '\n')

with open(out4, 'w') as f:
    for node in blackList.keys():
        if not node.startswith('^'):
            f.write(node + '\n')
