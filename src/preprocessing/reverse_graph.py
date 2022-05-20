import utils

#folder containing the work files
#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_4800/'

io_folder_path = utils.io_folder_path

#input files
in1 = io_folder_path + utils.network_app + '_src_sink_low.dot' #'inc_A_dot_src_sink.dot'

#output files
out1 = io_folder_path + 'rev_' + utils.network_app + '_src_sink_low.dot' #'rev_inc_A_dot_src_sink.dot'


rev_graph_src = []
rev_graph_dst = []
#constructing the graph and initializing the nodes levels from the dot file
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes)  > 1:
            rev_graph_src.append(nodes[1])
            rev_graph_dst.append(nodes[0])

with open(out1, 'w') as f:
    f.write('digraph{\n')
    for i in range(0, len(rev_graph_src)):
        f.write('"' + rev_graph_src[i] + '"->"' + rev_graph_dst[i] + '"\n')
    f.write('}')