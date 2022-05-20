import utils

io_folder_path = utils.io_folder_path
network_app = utils.network_app

in1 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'
in2 = io_folder_path + 'rev_' + network_app + '_src_sink_nodes_levels_low.txt'

# output file
out1 = io_folder_path + 'priorities.txt'

reverse_levels = {}
levels = {}

# get nodes levels
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            levels[node_and_level[0]] = node_and_level[1]

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            reverse_levels[node_and_level[0]] = node_and_level[1]

with open(out1, 'w') as f:
  for node, level in levels.items():
    f.write(node + ' ' + str(int(level) + int(reverse_levels[node])) + '\n')
