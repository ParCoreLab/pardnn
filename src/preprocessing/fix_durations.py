import utils
import logging

logger = logging.getLogger()

io_folder_path = utils.io_folder_path

in1 = io_folder_path + 'nodes_average_durations.txt'
in1_b = io_folder_path + 'nodes_average_durations.txt'
in2 = io_folder_path + utils.network_app + '_src_sink_low.dot'
in3 = io_folder_path + 'operations_attributes.txt'
in4 = io_folder_path + 'tensors_sz_32_low.txt'
out1 = io_folder_path + 'nodes_average_durations_fixed.txt'

tensors_sizes = {}
# get tensors sizes
with open(in4, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splitted = line.split('::')
    tensors_sizes[splitted[0]] = int(splitted[1])
        
graph = {}
rev_graph = {}
# initializing the nodes and adjacencies from the dot file
with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            if splits[0] in graph.keys():
                graph[splits[0]].append(splits[1])
            else:
                graph[splits[0]] = [splits[1]]
            
            if splits[1] in rev_graph.keys():
                rev_graph[splits[1]].append(splits[0])
            else:
                rev_graph[splits[1]] = [splits[0]]
 
ops_types = {}
with open(in3, 'r') as f:
  for line in f:
    splits = utils.clean_line(line).lower().split('::')
    ops_types[splits[0]] = splits[1]

single_nodes_durations = {}
with open(in1, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) == 2:
      single_nodes_durations[splits[0]] = int(splits[1])
                             
nodes_durations = {}
with open(in1_b, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) == 2:
      nodes_durations[splits[0]] = int(splits[1])

nodes_keys_mapping = {}
keys_durations_lists = {}
for node, rev_adjs in rev_graph.items():
  if node not in nodes_durations:
    continue
  time_key = ops_types[node] if node in ops_types else node
  for rev_adj in rev_adjs:
    time_key += '_' + (str(tensors_sizes[rev_adj]) if rev_adj in tensors_sizes else '1')
  if time_key not in keys_durations_lists:
    keys_durations_lists[time_key] = []
  
  nodes_keys_mapping[node] = time_key
  
  node_duration = 1  
  if node in single_nodes_durations:
    node_duration = single_nodes_durations[node]
  elif node in nodes_durations:
    node_duration = nodes_durations[node]
    
  keys_durations_lists[time_key].append(node_duration)

keys_durations = {}
for time_key, times_list in keys_durations_lists.items():
  keys_durations[time_key] = int( ( sum(times_list) / len(times_list) + times_list[int(len(times_list)/2)] ) / 2 + 1 ) #averaging mean and median

# print(len(nodes_durations))
# print(len(nodes_keys_mapping))
logger.debug(len(nodes_durations))
logger.debug(len(nodes_keys_mapping))

for node in nodes_durations:
  if node not in nodes_keys_mapping and node in graph:
    # print(node, nodes_durations[node])
    logger.debug(node, nodes_durations[node])
with open(out1, 'w') as f:
  for node, duration in nodes_durations.items():
    if node not in graph:
      continue
    f.write(node + '::' + (str(keys_durations[nodes_keys_mapping[node]]) if node in nodes_keys_mapping else str(nodes_durations[node]) ) + "\n")
    
      
