import utils

network_app = utils.network_app
io_folder_path = utils.io_folder_path
# input files
#in1 = io_folder_path + 'op_mem_for_indices.txt'
in2 = io_folder_path + 'op_mem_for.txt'
in3 = io_folder_path + 'res_nodes.txt'
in4 = io_folder_path + 'placement.place'
in5 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'
in6 = io_folder_path + 'memory.txt'

#out1 = io_folder_path + 'forwarding.txt'
out2 = io_folder_path + 'forwarding_paths.txt'

#forwarding_parent_indices = {}
forwarding_parent = {}

""" with open(in1, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) > 1:
      forwarding_parent_indices[splits[1]] = int(splits[0])
 """
with open(in2, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    splits = line.split('::')
    if len(splits) > 1:
      forwarding_parent[splits[0]] = splits[1] 

res_nodes = {}
with open(in3, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    splits = line.split('::')
    if len(splits) > 2:
      res_nodes[splits[1]] = int(splits[2])

mappings = {}
forwarding_paths = {}
for node in res_nodes:
  src = node
  forwarding_paths[src] = []
  while src in forwarding_parent:
    forwarding_paths[node].append(forwarding_parent[src])
    src = forwarding_parent[src]

""" with open(out1, 'w') as f:
  for key, val in forwarding_parent.items():
    f.write(key + '::' + val + '\n')
 """
placement = {}
with open(in4, 'r') as f:
  for line in f:
    line = utils.clean_line_keep_spaces(line)
    splits = line.split(' ')
    if len(splits) > 1:
      #print(splits[0])
      placement[splits[0]] = splits[1]

nodes_levels = {}
with open(in5, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) > 1:
      nodes_levels[splits[0]] = int(splits[1])

nodes_memory = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_memory[node_name] = int(splitted[1])

with open(out2, 'w') as f:
  for key, val in forwarding_paths.items():
    if len(val) == 0:
      continue
    if key in placement or key in nodes_memory:
      f.write(key)
    elif key.split('-')[0] in placement or key.split('-')[0] in nodes_memory:
      f.write(key.split('-')[0])
    f.write('::')
    iii = 0
    for val_ in val:
      if val_ in placement or val_ in nodes_memory:
        f.write(val_)
      elif val_.split('-')[0] in placement or val_.split('-')[0] in nodes_memory:
        f.write(val_.split('-')[0])
      else:
        f.write(val_)
        print(val_)
      iii += 1
      if iii < len(val):
        f.write('::')
    f.write('\n')

forwarding_paths = {}
with open(out2, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) > 1:
      belongs = True
      for split in splits:
        if split not in placement and split not in nodes_memory:
          belongs = False
          break
      if belongs:
        forwarding_paths[splits[0]] = splits[1:]

with open(out2, 'w') as f:
  for key, val in forwarding_paths.items():
    f.write(key + '::')
    iii = 0
    for val_ in val:
      iii += 1
      f.write(val_)
      if iii < len(val):
        f.write('::') 

    f.write('\n')

forwarding_paths = {}
with open(out2, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) > 1:
      forwarding_paths[splits[0]] = splits[1:]

smm = 0
for node, path in forwarding_paths.items():
  #print(path)
  node = node.lower()
  #if len(path) > 0 and node in nodes_levels and path[-1] in nodes_levels and nodes_levels[path[-1]] - nodes_levels[node] > 1 and \
    #node in placement and placement[node] == '0' and nodes_levels[node] > 3500 and nodes_levels[node] < 5000:
  if node in placement and placement[node] == '0':  
    smm += nodes_memory[node]
    print(node)
  #else:
  #  print(node)

print(smm/(1000*1000*1000))