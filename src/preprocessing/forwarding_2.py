import utils

network_app = utils.network_app
io_folder_path = utils.io_folder_path
# input files
in1 = io_folder_path + utils.network_app + '_src_sink_low.dot'
in2 = io_folder_path + 'op_mem_for.txt'
in3 = io_folder_path + 'res_nodes.txt'
in4 = io_folder_path + 'placement.place'
in5 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'
in6 = io_folder_path + 'memory.txt'
in7 = io_folder_path + 'ref_nodes.txt'
in8 = io_folder_path + 'var_nodes.txt'

out1 = io_folder_path + 'res_nodes_cleaned.txt'
out2 = io_folder_path + 'forwarding_paths.txt'
out3 = io_folder_path + 'long_living.txt'

#forwarding_parent_indices = {}
forwarding_parent = {}

all_nodes = {}
graph = {}
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            all_nodes[splits[0]] = 1
            all_nodes[splits[1]] = 1

            if splits[0] in graph.keys():
                graph[splits[0]].append(splits[1])
            else:
                graph[splits[0]] = [splits[1]]

nodes_levels = {}
with open(in5, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) > 1:
      nodes_levels[splits[0]] = int(splits[1])

with open(in2, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    splits = line.split('::')
    exists =True
    if len(splits) > 1:
      parent = ''
      child = ''
      if splits[0] in all_nodes:
        parent = splits[0]
      elif splits[0].split('-')[0] in all_nodes:
        parent = splits[0].split('-')[0]
      else:
        exists = False

      if splits[1] in all_nodes:
        child = splits[1]
      elif splits[1].split('-')[0] in all_nodes:
        child = splits[1].split('-')[0]
      else:
        exists = False
    
      if exists:
        if parent != child:
          forwarding_parent[parent] = child

res_nodes = {}
smm = 0
overall = 0
wasted_nodes = {}
allocated_nodes = {}
nodes_allocation_deallocation_span = {}
max_allocated_level = 0
deallocated_nodes = {}
tr_deallocated_nodes = {}
with open(in3, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    splits = line.split('::')
    node = ''
    if len(splits) > 2:
      if splits[1] == '':
        continue
      if splits[0] == 'allocated':
        allocated_nodes[splits[1]] = int(splits[2])
        if splits[1] in nodes_levels:
          max_allocated_level = nodes_levels[splits[1]]
      elif splits[0] == 'deallocated':
        deallocated_nodes[splits[1]] = int(splits[2])
        if splits[1] in nodes_levels and (node not in nodes_allocation_deallocation_span or nodes_allocation_deallocation_span[node] < \
          max_allocated_level - nodes_levels[splits[1]]):  
          nodes_allocation_deallocation_span[splits[1]] = max_allocated_level - nodes_levels[splits[1]]
      elif splits[0] == 'tr_deallocated':
        tr_deallocated_nodes[splits[1]] = int(splits[2])
        if splits[1] in nodes_levels and (node not in nodes_allocation_deallocation_span or nodes_allocation_deallocation_span[node] < \
          max_allocated_level - nodes_levels[splits[1]]):  
          nodes_allocation_deallocation_span[splits[1]] = max_allocated_level - nodes_levels[splits[1]]
      if splits[1] in all_nodes:
        if (splits[1] not in res_nodes or res_nodes[splits[1]] < int(splits[2])):
          overall += int(splits[2])
          if splits[1] in res_nodes:
            overall -= res_nodes[splits[1]]
          res_nodes[splits[1]] = int(splits[2])
      elif splits[1] not in wasted_nodes:
        wasted_nodes[splits[1]] = 1
        smm += int(splits[2])

print('wasted mem::' + str(smm/(1024*1024*1024)))
print('overall mem::' + str(overall/(1024*1024*1024)))

mappings = {}
included = {}
forwarding_paths = {}
for node in res_nodes:
  src = node
  forwarding_paths[src] = []
  included[src] = 1
  while src in forwarding_parent:
    if forwarding_parent[src] not in res_nodes or res_nodes[forwarding_parent[src]] <= res_nodes[node] and src not in included:
      if forwarding_parent[src] not in included:
        forwarding_paths[node].append(forwarding_parent[src])
        included[forwarding_parent[src]] = 1
        src = forwarding_parent[src]
      else:
        break
    else:
      break
  
  if len(forwarding_paths[node]) == 0:
    del forwarding_paths[node]

with open(out1, 'w') as f:
  for key, val in res_nodes.items():
    f.write(key + '::' + str(val) + '\n')

placement = {}
with open(in4, 'r') as f:
  for line in f:
    line = utils.clean_line_keep_spaces(line)
    splits = line.split(' ')
    if len(splits) > 1:
      placement[splits[0]] = splits[1]

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
    #if nodes_levels[key] < 4:
    #  continue
    f.write(key + '::')
    iii = 0
    for val_ in val:
      iii += 1
      f.write(val_)
      if iii < len(val):
        f.write('::') 

    f.write('\n')

smm = 0

for path in forwarding_paths.keys():
  smm += res_nodes[path]
print(smm/1000000000)
print('%%%%%%%%%%')

ref_nodes= {}
with open(in7, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    ref_nodes[line] = 1

var_nodes= {}
with open(in8, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    var_nodes[line] = 1

cntt = 0
smm = 0
cntt2 = 0
smm2 = 0
grp1 = {}

all_forwarded_nodes = {}
for node, path in forwarding_paths.items():
  all_forwarded_nodes[node] = 1
  for in_path in path:
    all_forwarded_nodes[in_path] = 1

smm3 = 0
for node, mem in res_nodes.items():
  smm3 += mem
  max_dist = 0
  #if node not in ref_nodes and node not in var_nodes and node not in deallocated_nodes and node not in forwarding_paths and nodes_levels[node] >= 4:
  if node in nodes_allocation_deallocation_span and nodes_allocation_deallocation_span[node] > 1000 and node not in ref_nodes and node not in var_nodes \
    and node not in forwarding_paths:
    grp1[node] = 1
    cntt2 += 1
    smm2 += nodes_memory[node]
    #if nodes_memory[node] - mem > 1000000:
    print(node + '::' + str(nodes_memory[node]))

    """ if (node in nodes_allocation_deallocation_span and node not in all_forwarded_nodes and node not in ref_nodes\
    and node in placement and placement[node] == '0' and nodes_allocation_deallocation_span[node] > 1000):
      print('=='+node)
    else:
      print('!='+node) """

    #print(node + '::' +str(placement[node]))
  """ if node not in all_forwarded_nodes and node not in ref_nodes and node in placement and placement[node] == '0': 

    for child in graph[node]:
      max_dist = max(max_dist, nodes_levels[child] - nodes_levels[node])

    if node in nodes_allocation_deallocation_span and nodes_allocation_deallocation_span[node] > max(1000, max_dist):
      #print(node + '::' + str(nodes_levels[node]) + '::' + str(nodes_allocation_deallocation_span[node]) )
      if node not in grp1:
        print('xx'+node)
      cntt += 1
      smm += mem """
  #if node in all_nodes and node in allocated_nodes and node not in deallocated_nodes:
  #  cntt2 += 1
print(smm3)

smm = 0
with open(out3, 'w') as f:
  for node in grp1:
    smm += nodes_memory[node]
    f.write(node + '\n')

print('long living:' + str(smm))
smmm = 0
print("###################")
for node in all_nodes:
  printt = True
  if node == 'snk':
    continue
  for child in graph[node]:
    if nodes_levels[child] < nodes_levels[node] + 100:
      printt = False
    if printt and node not in ref_nodes and node not in var_nodes and node not in grp1:
      if node in res_nodes:
        smmm += res_nodes[node]
        print(node + str(res_nodes[node]))
      elif node in nodes_memory:
        smm += nodes_memory[node]
print(smmm/1000000000)
print('=====================')
print(cntt)
print(smm/(1000000000))
print(cntt2)
print(smm2/(1000000000))

print('********************')

smm = 0
for node, span in nodes_allocation_deallocation_span.items():
  if span > 300 and node not in ref_nodes and node not in var_nodes and nodes_levels[node] >= 4 and node not in forwarding_paths: 
    smm += res_nodes[node]
    print(node)

print(smm/1000000000)
print('^^^^^^^^^^^^^^^^^^^^')
smm = 0
for node in res_nodes:
  #if allocated_nodes[node] - max(deallocated_nodes[node] if node in deallocated_nodes else 0, tr_deallocated_nodes[node] if node in tr_deallocated_nodes else 0) > 0:\
  act_allocated = allocated_nodes[node] if node in allocated_nodes else 0 #if node in forwarding_paths else allocated_nodes[node] - \
    #max(deallocated_nodes[node] if node in deallocated_nodes else 0, tr_deallocated_nodes[node] if node in tr_deallocated_nodes else 0)
  if  act_allocated > nodes_memory[node]:# - max(deallocated_nodes[node] if node in deallocated_nodes else 0, 0) - \
    #nodes_memory[node] > 0:
    smm += allocated_nodes[node]
    nodes_memory[node] = allocated_nodes[node]
    print(node)
    #and node not in forwarding_paths:
    #if node in nodes_allocation_deallocation_span and nodes_allocation_deallocation_span[node] - max([nodes_levels[adj] for adj in graph[node]]) >= min(1000,\
      #nodes_levels['snk'] - nodes_levels[node] - 100) and node not in forwarding_paths:
      #smm += allocated_nodes[node]# - max(deallocated_nodes[node] if node in deallocated_nodes else 0, tr_deallocated_nodes[node] if node in tr_deallocated_nodes else 0)

print(smm/1000000000)

with open(in6, 'w') as f:
    for key, val in nodes_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
count = 0
for node in allocated_nodes.keys():
    if node not in deallocated_nodes and node in nodes_memory and node not in var_nodes and nodes_levels[node] > 4 and node not in ref_nodes:
      strr = ""
      for adj in graph[node]:
        strr += str(adj) + "::" + str(nodes_levels[adj])
      print(node + "::" + str(nodes_memory[node])+"::"+str(nodes_levels[node])+"::"+strr)
      count += 1

print(count)

count = 0
smm = 0
smm_all = 0
print('-------------------newnewnew--------------------------')
analysis_graph = utils.read_profiling_file_v2(in2)
for node in all_nodes:
  if node in analysis_graph:
    smm_all += analysis_graph[node].duration
  cand = node not in ref_nodes and node not in var_nodes and 'read' not in node
  if node == 'snk':
    continue
  for child in graph[node]:
    if nodes_levels[child] - nodes_levels[node] < 100:
      cand = False
  if cand and node in analysis_graph:
    count += 1
    smm += analysis_graph[node].duration
    print(node)

print(count)
print(smm / smm_all)