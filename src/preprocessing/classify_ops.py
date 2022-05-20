import utils
import logging

logger = logging.getLogger()

# folder containing the work files
io_folder_path = utils.io_folder_path
network_app = utils.network_app
in1 = io_folder_path + network_app + \
    '_src_sink_low.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'operations_attributes.txt'
in3 = io_folder_path + 'vanilla_cleaned.place'

out1 = io_folder_path + 'var_nodes.txt'
out2 = io_folder_path + 'ref_nodes.txt'
out3 = io_folder_path + 'no_ops.txt'
out4 = io_folder_path + 'collocations.txt'

graph = {}
rev_graph = {}
# initializing the nodes and adjacencies from the dot file
with open(in1, 'r') as f:
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

no_ops = {}
ref_ops = {}
var_ops = {}
ops_types = {}
const_ops = {}
with open(in2, 'r') as f:
    for line in f:
        splits = utils.clean_line(line).lower().split('::')
        ops_types[splits[0]] = splits[1]
        if splits[0].lower() not in graph:# fareed recheck
          continue
        if splits[1] == 'noop' or 'control_dependency' in splits[0]:
            no_ops[splits[0]] = 1
        elif splits[1] in ['variablev2', 'variable', 'varhandleop']: 
          var_ops[splits[0]] = 1
        elif splits[1] == 'const':
          const_ops[splits[0]] = 1
        if (len(splits) > 2 and splits[2] == 'true') or 'isvariableinitialized' in splits[0] or \
          'assignvariableop' in splits[1] or 'readvariableop' in splits[1] or 'resourceapplyadam' in splits[1] \
            or 'varisinitializedop' in splits[1]:  
            if splits[0] not in var_ops:
                ref_ops[splits[0]] = 1

vanilla_placement = {}
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' ')
        vanilla_placement[splits[0]] = splits[1]

"""for node in graph:
  if node.startswith("^"):
    no_ops[node]=1"""
""" for node in graph['rnnlm/softmax_w']:
    if vanilla_placement[node] != '-1':
        print(node)
        for rev_adj in rev_graph[node]:
            print(rev_adj)
print('--------------------')
for node in graph['batch_time']:
    if vanilla_placement[node] != '-1':
        print(node)
        for rev_adj in rev_graph[node]:
            print(rev_adj) """

collocations = {}
collocated = {}
for node in graph:
  if node.startswith('^'):
    no_ops[node] = 1
    continue
  if node.endswith(('applyadam','applymomentum')) and node in ref_ops:
    collocated[node] = rev_graph[node][0]
    if node not in collocations:
      collocations[rev_graph[node][0]] = [node]
    for rev_adj in rev_graph[node]:
      if rev_adj == rev_graph[node][0] or (rev_adj not in ref_ops and rev_adj not in var_ops):
        continue
      collocated[rev_adj] = rev_graph[node][0]
      collocations[rev_graph[node][0]].append(rev_adj)
      for adj in graph[rev_adj]:
        if adj == node or (adj not in ref_ops and adj not in var_ops):
          continue
        collocated[adj] = rev_graph[node][0]
        collocations[rev_graph[node][0]].append(adj)

for node in graph:
  if node.startswith('^'):
    continue
  if node.endswith('assign') and node in ref_ops:
    if node in collocated:
      continue
    ref_node = rev_graph[node][0]
    if 'unit_3_99/bn_1/beta/assign' in node:
        # print(ref_node)
        logger.debug(ref_node)
    if ref_node not in collocations:
      collocations[ref_node] = [node]
    else:
      collocations[ref_node].append(node)
    collocated[node] = ref_node

    for rev_adj in rev_graph[node]:
      if rev_adj not in ref_ops or rev_adj not in var_ops:
        continue
      collocations[ref_node].append(rev_adj)
      collocated[rev_adj] = ref_node

for node in ref_ops.keys():
  if node not in collocated:
    for rev_adj in rev_graph[node]:
      if rev_adj in collocations:
        collocations[rev_adj].append(node)
        collocated[node] = rev_adj

for node in var_ops.keys():
    if node + '/read' in graph:
      if node in collocations:
          collocations[node].append(node + '/read')
      else:
          collocations[collocated[node]].append(node + '/read')

""" for node in var_ops.keys():
    in_collocations = False
    if node not in vanilla_placement or vanilla_placement[node] != '-1':
        if node not in collocated:
            collocated[node] = node
            collocations[node] = []
        else:
            in_collocations = True
        for adj in graph[node]:
            if (adj in ref_ops or adj in var_ops) and adj not in collocated and (adj not in vanilla_placement or vanilla_placement[adj] != '-1'):
                if in_collocations:
                  collocations[collocated[node]].append(adj)
                else:
                  collocations[node].append(adj)
                collocated[adj] = node
                for rev_adj in rev_graph[adj]:
                    if adj == 'unit_3_100/bn_1/gamma/momentum/assign':
                      print(rev_adj)
                    if rev_adj != node and (rev_adj in ref_ops or rev_adj in var_ops) and rev_adj not in collocated and (rev_adj not in vanilla_placement or vanilla_placement[rev_adj] != '-1'):
                        collocated[rev_adj] = node
                        if adj == 'unit_3_100/bn_1/gamma/momentum/assign':
                          print(rev_adj)
                        if in_collocations:
                          collocations[collocated[node]].append(rev_adj)
                        else:
                          collocations[node].append(rev_adj) """

with open(out1, 'w') as f:
    for var_node in var_ops.keys():
        f.write(var_node + '\n')

    #for var_node in const_ops.keys():
    #    f.write(var_node + '\n')

with open(out2, 'w') as f:
    for ref_node in ref_ops.keys():
        f.write(ref_node + '\n')

with open(out3, 'w') as f:
    for no_op_node in no_ops.keys():
        f.write(no_op_node + '\n')

# print(len(collocations))
logger.debug(len(collocations))

with open(out4, 'w') as f:
    for coll_key, collocated in collocations.items():
        _str = coll_key
        for collocated_node in collocated:
            _str += '::' + collocated_node
        _str += '\n'
        f.write(_str)
