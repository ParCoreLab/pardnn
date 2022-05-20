import utils

# folder containing the work files
#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_4800/'

io_folder_path = utils.io_folder_path

# input files
in1 = io_folder_path + utils.network_app + 't_low.dot'#'inc_A_dot.dot'

#output filse
out1 = io_folder_path + utils.network_app + '_src_sink_low.dot'#'inc_A_dot_src_sink.dot'
sink_node_name = 'snk'
src_node_name = 'src'

# will contain the graph as an adgacency list
src_nodes = {}
dst_nodes = {}
srcs_to_write = []
dsts_to_write = []

# initializing the nodes and adjacencies from the dot file
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            srcs_to_write.append(nodes[0])
            dsts_to_write.append(nodes[1])
            src_nodes[nodes[0]] = 1
            dst_nodes[nodes[1]] = 1


for src_node in src_nodes.keys():
    if src_node not in dst_nodes:
        srcs_to_write.append(src_node_name)
        dsts_to_write.append(src_node)


for dst_node in dst_nodes.keys():
    if dst_node not in src_nodes:
        srcs_to_write.append(dst_node)
        dsts_to_write.append(sink_node_name)

with open(out1, 'w') as f:
    f.write('digraph{\n')
    for i in range(0, len(srcs_to_write)):
        f.write('"' + srcs_to_write[i] + '"->"' + dsts_to_write[i] + '"\n')
    f.write('}')
