import utils
from os import walk

io_folder_path = utils.io_folder_path
#in1 = io_folder_path + 'mem.txt'
in2 = io_folder_path + utils.network_app + '_src_sink_low.dot'
in4 = io_folder_path + 'no_ops.txt'
in5 = io_folder_path + 'operations_attributes.txt'
out1 = io_folder_path + 'mems.txt'
out1_1 = io_folder_path + 'nf_memory.txt'
out2 = io_folder_path + 'res_memory.txt'
out2_1 = io_folder_path + 'nf_res_memory.txt'

all_nodes = {}

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            all_nodes[splits[0]] = 1
            all_nodes[splits[1]] = 1

no_op_nodes = {}
with open(in4, 'r') as f:
    for line in f:
        no_op_nodes[utils.clean_line(line)] = 1

do_not_check_ops = {}
with open(in5, 'r') as f:
    for line in f:
        splits = utils.clean_line(line).lower().split('::')
        if splits[1] == 'switch' or splits[1] == 'identity' or (len(splits) > 2 and splits[2] == 'true' ):
            do_not_check_ops[splits[0]] = 1

sum_inits = 0

def text_to_bytes(mem_cons):
    node_mem_cons = 0 
    if mem_cons.endswith('GB'):
        node_mem_cons = float(mem_cons[:-2]) * 1024 * 1024 * 1024
    elif mem_cons.endswith('MB'):
        node_mem_cons = float(mem_cons[:-2]) * 1024 * 1024
    elif mem_cons.endswith('KB'):
        node_mem_cons = float(mem_cons[:-2]) * 1024
    elif mem_cons.endswith('B'):
        node_mem_cons = float(mem_cons[:-1])

    return node_mem_cons

nodes_memory = {}
nf_nodes_memory = {}
additional_memory = {}
res_memory = {}
nf_res_memory = {}

files = []
for (dirpath, dirnames, filenames) in walk(io_folder_path):
    files.extend(filenames)
    break

# getting time (weight) info for nodes
nodes_durations = {}
for file in files:
    if 'mem_' in file:
        print(file)
        with open(io_folder_path + file, 'r') as f:
            for line in f:
                if not '_TFProfRoot' in line:
                    line = utils.clean_line_keep_spaces(line)
                    splits = line.split('::(')
                    if len(splits) < 2:
                        splits = line.split(' (')

                    if len(splits) > 1:
                        node_name = splits[0].lower()
                        node_name = utils.clean_line(node_name)

                        mem_cons = utils.clean_line(splits[1]).split(',')

                        if len(mem_cons) > 2:
                            res_cons = text_to_bytes(mem_cons[2].split('/')[0])
                            if res_cons > 0:
                                if node_name in all_nodes:
                                    res_memory[node_name] = res_cons
                                else:
                                    nf_res_memory[node_name] = res_cons

                        mem_cons = mem_cons[-1]
                        mem_cons = mem_cons.split('/')[0]

                        node_mem_cons = text_to_bytes(mem_cons)

                        if node_name in all_nodes:
                            if node_name not in nodes_memory:
                                nodes_memory[node_name] = []
                            nodes_memory[node_name].append(node_mem_cons)
                        elif node_mem_cons > 0:
                            nf_nodes_memory[node_name] = node_mem_cons

with open(out1, 'w') as f:
    for key, val in nodes_memory.items():
        f.write(key + '::' + str(val) + '\n')
