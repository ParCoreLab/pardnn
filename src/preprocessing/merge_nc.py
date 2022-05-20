import utils
import to_lower
# folder containing the work files
#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_4800/'

io_folder_path = utils.io_folder_path

# input files
network_app = utils.network_app
in1 = io_folder_path + 'vanilla_cleaned_low.place'
in2 = io_folder_path + 'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e.place'
in5 = io_folder_path + network_app + '_src_sink_low.dot'#'resnet_src_sink_low.dot'
in5_b = io_folder_path + 'rev_' + network_app + '_src_sink_low.dot'#'resnet_src_sink_low.dot'
in6 = io_folder_path + 'colocation_32_low.txt'
in7 = io_folder_path + 'timeline_step17_low.json'


out1 = io_folder_path + 'placement.place'

nodes_levels = {}

analysis_graph = utils.read_profiling_file(in7, True)

vanilla_placement = {}
placer_placement = {}

with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' ')
        vanilla_placement[splits[0]] = splits[1]

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' ')
        placer_placement[splits[0]] = splits[1]


graph = {}
rev_graph = {}
#constructing the graph and initializing the nodes levels from the dot file
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes)  > 1:
            if nodes[0] in graph:
                graph[nodes[0]].append(nodes[1])
            else:
                graph[nodes[0]] = [nodes[1]]

with open(in5_b, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes)  > 1:
            if nodes[0] in rev_graph:
                rev_graph[nodes[0]].append(nodes[1])
            else:
                rev_graph[nodes[0]] = [nodes[1]]

applies = ['/applymomentum', '/applyadam']
assigns = ['/assign', '/assignsub']

for i in range(0, len(assigns)):
    current_assign = assigns[i]
    for j in range(1, 100):
        assigns.append(current_assign + '_' +str(j))

def place_collocated_nodes(rev_graph):
    for node in rev_graph.keys():
        if ( node.endswith(tuple(applies)) or node.endswith(tuple(assigns)) ) and not node.startswith('^'):
            orig_node = node[:node.rfind('/')]
            read_node = node[:node.rfind('/')] + '/read'
            if node.endswith(tuple(applies)):
                plcmnt = placer_placement[node]
                placer_placement[orig_node] = plcmnt
                placer_placement[read_node] = plcmnt

                for assign_node in assigns:
                    if (orig_node + assign_node) in graph:
                        placer_placement[assign_node] = plcmnt

            elif node.endswith(tuple(assigns)):
                plcmnt = placer_placement[node]
                placer_placement[orig_node] = plcmnt
                placer_placement[read_node] = plcmnt

place_collocated_nodes(rev_graph)              

collocated = {}
rev_collocated = {}
with open(in6) as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line).lower()
        splits = line.split(' vs ')
        if splits[0] not in collocated:
            collocated[splits[0]] = [splits[1]]
        else:
            collocated[splits[0]].append(splits[1])
        if splits[1] not in rev_collocated:
            rev_collocated[splits[1]] = [splits[0]]
        else:
            rev_collocated[splits[1]].append(splits[0])

for collocation_dst in rev_collocated.keys():
    #if collocation_src in graph:
    for adj_node in rev_collocated[collocation_dst]:#graph[collocation_src]:
        placer_placement[adj_node] = placer_placement[collocation_dst]
        for another_dst in collocated[adj_node]:
            placer_placement[another_dst] = placer_placement[adj_node]

parts_weights = {}
with open(out1, 'w') as f:
    for node, part in vanilla_placement.items():
        if not node.startswith('^') and node in placer_placement:
            if node in placer_placement:
                if part == '-1':
                    f.write(node + ' ' + part + '\n')
                else:
                    f.write(node + ' ' + placer_placement[node] + '\n')
                    if placer_placement[node] not in parts_weights:
                        parts_weights[placer_placement[node]] = analysis_graph[node].duration if node in analysis_graph else 1
                    else:
                        parts_weights[placer_placement[node]] += analysis_graph[node].duration if node in analysis_graph else 1
            else:
                f.write(node + ' ' + part + '\n')

print(parts_weights)
