import utils

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/'

# input files
in1 = io_folder_path + 'timeline_step303.json'
in2 = io_folder_path + 'part_8_1799_nodes_levels.txt'
in3 = io_folder_path + 'part_8_1799.dot'
in4 = io_folder_path + 'txt_part_8_1799mapping.txt'
in5 = io_folder_path + 'tensors_sz.txt'

# output files
out = io_folder_path + 'with_edges_metis.graph'

# node_name -> node properties
nodes_props_dict = utils.read_profiling_file(in1)

max_level = 0
no_of_nodes = 0
nodes_levels = {}
# fill the analysis graph nodes levels
with open(in2, 'r') as f:
    for line in f:
        no_of_nodes = no_of_nodes + 1
        line = utils.clean_line_keep_spaces(line)
        node_and_level = line.split("::")
        level = node_and_level[len(node_and_level) - 1]
        nodes_levels[node_and_level[0]] = int(level)
        if node_and_level[0] in nodes_props_dict:
            nodes_props_dict[node_and_level[0]].level = level
        if int(level) > int(max_level):
            max_level = level

max_level = int(max_level) + 1

# fill tensors sizes
tensors_sizes = {}
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensors_sizes[splitted[0]] = splitted[1]


# will contain the graph as an adgacency list
graph = {}
graph_2 = {}
no_of_edges = 0
# initializing the nodes and adjacencies from the dot file
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            no_of_edges = no_of_edges + 1
            if nodes[0] in graph.keys():
                graph[nodes[0]].append(nodes[1])
            else:
                graph[nodes[0]] = [nodes[1]]

            if nodes[1] in graph_2.keys():
                graph_2[nodes[1]].append(nodes[0])
            else:
                graph_2[nodes[1]] = [nodes[0]]


# nodes names mapped to numbers
nodes_mapping = {}
rev_node_mapping = {}
with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        nodes_mapping[splitted[0]] = splitted[1]
        rev_node_mapping[splitted[1]] = splitted[0]

with open(out, 'w') as f:
    f.write( str(no_of_nodes) + " " + str(no_of_edges) + " " + "011 " + "\n")
    for i in range(1, no_of_nodes + 1):
        node = rev_node_mapping[str(i)]
        node_weight = 0
        tensor_size = 1
        src_node = False
        if node in nodes_props_dict:
                node_weight = nodes_props_dict[node].duration

        if node in graph.keys():
            src_node = True
            adj = graph[node]
            if node in tensors_sizes:
                tensor_size = tensors_sizes[node]
            f.write(str(node_weight) + ' ')
            for adj_node in adj:
                f.write(str(nodes_mapping[adj_node]) + ' ' +  str(tensor_size) + ' ')

        if node in graph_2.keys():
            adj = graph_2[node]
            if not src_node:
                f.write(str(node_weight) + ' ')
            for adj_node in adj:
                if adj_node in tensors_sizes:
                    tensor_size = tensors_sizes[adj_node]
                f.write(str(nodes_mapping[adj_node]) + ' ' +  str(tensor_size) + ' ')
        f.write('\n')
