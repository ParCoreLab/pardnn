import utils

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/'

# input files
in1 = io_folder_path + 'timeline_step303_dfs.json'
in3 = io_folder_path + 'txt_part_20_1799.dot'
in4 = io_folder_path + 'txt_part_20_1799mapping.txt'

# output files
out = io_folder_path + 'h_edges_metis.graph'

# node_name -> node properties
nodes_props_dict = utils.read_profiling_file(in1)

# will contain the graph as an adgacency list
graph = {}
graph_2 = {}
no_of_edges = 0
no_of_nodes = 0
# initializing the nodes and adjacencies from the dot file
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            no_of_edges = no_of_edges + 1

            if not nodes[0] in graph.keys() and not nodes[0] in graph_2.keys():
                no_of_nodes = no_of_nodes + 1
            if not nodes[1] in graph.keys() and not nodes[1] in graph_2.keys():
                no_of_nodes = no_of_nodes + 1

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
    f.write( str(no_of_nodes) + " " + str(no_of_nodes) + " " + "10 " + "\n")
    for i in range(1, no_of_nodes + 1):
        node = rev_node_mapping[str(i)]

        if node in graph.keys():
            adj = graph[node]

        if node in graph_2.keys():
            adj = adj + graph_2[node]
        
        for adj_node in adj:        
            f.write(nodes_mapping[adj_node] + ' ')
        f.write('\n')

    for i in range(1, no_of_nodes + 1):
        node_weight = 0
        node =  rev_node_mapping[str(i)]
        if node in nodes_props_dict:
            node_weight = nodes_props_dict[node].duration
        f.write(str(node_weight) + '\n')