import utils
import logging
from os import walk

logger = logging.getLogger()

io_folder_path = utils.io_folder_path
in1 = io_folder_path + 'mem.txt'
in2 = io_folder_path + utils.network_app + '_src_sink_low.dot'
in4 = io_folder_path + 'no_ops.txt'
in5 = io_folder_path + 'operations_attributes.txt'
in6 = io_folder_path + 'tensors_sz_32_low.txt'
in7 = io_folder_path + 'var_nodes.txt'
out1 = io_folder_path + 'memory.txt'
out2 = io_folder_path + 'res_memory.txt'

in10 = io_folder_path + 'ref_nodes.txt'

ref_nodes = {}
with open(in10, 'r') as f:
    for line in f:
        ref_nodes[utils.clean_line(line)] = 1

var_nodes = {}
with open(in7, 'r') as f:
    for line in f:
        var_nodes[utils.clean_line(line)] = 1
        
in11 = io_folder_path + 'var_nodes.txt'
var_nodes = {}
with open(in11, 'r') as f:
    for line in f:
        var_nodes[utils.clean_line(line)] = 1

in12 = io_folder_path + 'collocations.txt'
collocations = {}
with open(in12, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("::")
        for node in splits:
            collocations[node] = 1

all_nodes = {}

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            all_nodes[splits[0]] = 1
            all_nodes[splits[1]] = 1

tensors_sizes = {}
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensor_size = int(splitted[1])
        tensor_name = splitted[0]
        tensors_sizes[tensor_name] = tensor_size

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

nodes_memories = {}
nodes_res_memory = {}
files = []
for (dirpath, dirnames, filenames) in walk(io_folder_path):
    files.extend(filenames)
    break

for _file in files:
  if 'mem_' in _file:
    with open(io_folder_path + _file, 'r') as f:
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
                    if(len(mem_cons) != 4):
                      continue

                    mem_cons = mem_cons[2]
                    mem_cons = mem_cons.split('/')[0]

                    node_mem_cons = text_to_bytes(mem_cons)
          
                    if node_name in all_nodes:
                      if node_name not in nodes_res_memory or nodes_res_memory[node_name] < node_mem_cons:
                        nodes_res_memory[node_name] = node_mem_cons

                    mem_cons = utils.clean_line(splits[1]).split(',')
                    mem_cons = mem_cons[-1]
                    mem_cons = mem_cons.split('/')[0]

                    node_mem_cons = text_to_bytes(mem_cons)
          
                    if node_name in all_nodes:
                      if node_name not in nodes_memories:
                        nodes_memories[node_name] = []
                      nodes_memories[node_name].append(node_mem_cons)
                        
                    if node_name in all_nodes:
                        all_nodes[node_name] = 0
                        nodes_res_memory[node_name] = max(nodes_res_memory[node_name], max(nodes_memories[node_name]))

for node, val in all_nodes.items():
    if val == 1:        
        nodes_res_memory[node] = 0
        if node in var_nodes:
          # print(node, ' is a var node without memory!')
          logger.debug(node, ' is a var node without memory!')
          nodes_res_memory[node] = tensors_sizes[node]
    #elif node in var_nodes:
    #    print(node, nodes_res_memory[node])

summ = 0
with open(out1, 'w') as f:
    for key, val in nodes_res_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')
        summ += val
        
# print(summ/1000000000)
logger.debug(summ/1000000000)

frequent_memory_vals = {}
for node, memories in nodes_memories.items():
  freqs = {}
  for memory in memories:
    if memory not in freqs:
      freqs[memory] = 0
    freqs[memory] += 1
    
  max_freq = 0
  most_frequent_mem = 0
  for memory, freq in freqs.items():
    if freq > max_freq:
      max_freq = freq
      most_frequent_mem = memory
  
  frequent_memory_vals[node] = most_frequent_mem 

with open(in6, 'w') as f:
  for tensor, size in tensors_sizes.items():
    if tensor in frequent_memory_vals and frequent_memory_vals[tensor] > 10 * size:
      f.write(tensor + '::' + str(int(max(frequent_memory_vals[tensor], size) )) + "\n")
    else:
      f.write(tensor + '::' + str(size) + "\n")
