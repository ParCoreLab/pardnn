import utils

in1 = utils.io_folder_path + 'memory.txt'
in2 = utils.io_folder_path + 'tensors_sz_32_low.txt'
in3 = utils.io_folder_path + 'res_nodes_cleaned.txt'
in4 = utils.io_folder_path + 'ref_nodes.txt'
in5 = utils.io_folder_path + 'all_tensors.txt'

nodes_mem = {}
with open(in1, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    nodes_mem[splits[0]] = int(splits[1])

tensors_sizes = {}
with open(in2, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    tensors_sizes[splits[0]] = int(splits[1])

res_mem = {}
with open(in3, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    res_mem[splits[0]] = int(splits[1])

ref_nodes = {}
with open(in4, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    ref_nodes[line] = 1

all_tensors = {}
with open(in5, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    splits = line.split('::')
    if len(splits) > 1:
      all_tensors[splits[0]] = int(splits[1])

for node, sz in tensors_sizes.items():
  comp_with = 0
  if node in nodes_mem:
    comp_with = nodes_mem[node]
  if node in res_mem:
    comp_with = max(comp_with, res_mem[node])
  if sz > comp_with + 1000000 and node not in ref_nodes:
    print(node + '::' + str(sz - comp_with))