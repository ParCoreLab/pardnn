import utils
import re
import logging
import shutil

#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_480/'
# input files

logger = logging.getLogger()

io_folder_path = utils.io_folder_path

network_app = utils.network_app

in1 = io_folder_path + network_app + '.dot'

# TODO: temp patch, should fix
shutil.copyfile(io_folder_path + 'mtf_dot.dot', in1)

in2 = io_folder_path + 'tensors_sz_32.txt'

out = io_folder_path + network_app + 't_low.dot'
out2 = io_folder_path + 'tensors_sz_32_low.txt'

all_nodes = {}
graph = {}
# constructing the graph and initializing the nodes levels from the dot file
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line).lower()
        nodes = line.split("->")
        if len(nodes) > 1:
            all_nodes[nodes[0]] = "1"
            all_nodes[nodes[1]] = "1"
            if nodes[0] in graph:
                graph[nodes[0]].append(nodes[1])
            else:
                graph[nodes[0]] = [nodes[1]]

for node in all_nodes.keys():
    if node.startswith('^'):
        normal_node = node[1:]
        if normal_node in all_nodes:
            if normal_node in graph.keys():
                graph[normal_node].append(node)
            else:
                graph[normal_node] = [node]
    sub_ten = re.search(r'\d+$', node)
    if sub_ten is not None and node[sub_ten.span()[0] - 1] == ':':
        normal_node = node[0:sub_ten.span()[0] - 1]
        if normal_node in graph.keys():
            graph[normal_node] += graph[node]
        else:
             graph[normal_node] = graph[node]
        del graph[node]

with open(out, 'w') as f:
    f.write('digraph{\n')
    for node,adjs in graph.items():
        for adj in adjs:
            f.write('"' + node.lower() + '"->"' + adj.lower() + '"\n')
    f.write('}')

tensors_sizes = {}
# get tensors sizes
with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        line=line.lower()
        splitted = line.split('::')
        tensors_sizes[splitted[0]] = splitted[1]

# print(len(tensors_sizes))
logger.debug(len(tensors_sizes))

with open(out2, 'w') as f:
    for tensor, size in tensors_sizes.items():
        f.write(tensor+"::"+size+"\n")

