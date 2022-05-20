import utils

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/'

# input files
in1 = io_folder_path + 'timeline_step303_vanilla.json'
in2 = io_folder_path + 'txt_part_20_1799mapping.txt'
in3 = io_folder_path + 'vanilla_Cleaned.place'
in4 = io_folder_path + 'txt_part_20_1799.dot'
in5 = io_folder_path + 'tensors_sz.txt'

out1 = io_folder_path + 'txt_part_20_1799_dagp.dot'
nodes_mapping = {}

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        nodes_mapping[splits[0]] = splits[1]

analysis_graph = utils.read_profiling_file(in1)


graph = {}
# initializing the nodes and adjacencies from the dot file
with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            if nodes[0] in graph:
                graph[nodes[0]].append(nodes[1])
            else:
                graph[nodes[0]] = [nodes[1]]


tensors_sizes = {}
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensors_sizes[splitted[0]] = splitted[1]


#0[weight="1"];
#0->1[weight="1"];

with open(out1, 'w') as f:
    f.write('digraph{\n')
    for node in nodes_mapping.keys():
        f.write(nodes_mapping[node] + '[weight="' + ( str(analysis_graph[node].duration) if node in analysis_graph.keys() else '0') + '"];\n')
    for src, dst_list in graph.items():
        for dst in dst_list:
            f.write(nodes_mapping[src] + '->' + nodes_mapping[dst] + '[weight="' + ( str(tensors_sizes[src]) if src in tensors_sizes.keys() else '0')  + '"];\n')
    f.write('}')