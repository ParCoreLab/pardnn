import utils
import nodeProps
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import collections
import heapq
import random
import queue

# folder containing the work files
io_folder_path = utils.io_folder_path
network_app = utils.network_app
in1 = io_folder_path + network_app + \
    '_src_sink_low.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'operations_attributes.txt'
in3 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'
in6 = io_folder_path + 'memory.txt'
in6_b = io_folder_path + 'res_nodes_cleaned.txt'
in7 = io_folder_path + 'vanilla_cleaned.place'

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
ops_types = {}
var_ops = {}
with open(in2, 'r') as f:
    for line in f:
        splits = utils.clean_line(line).lower().split('::')
        ops_types[splits[0]] = splits[1]
        if splits[1] == 'noop':
            no_ops[splits[0]] = 1
        elif len(splits) > 2 and splits[2] == 'true':
            ref_ops[splits[0]] = 1 
            if splits[1].startswith('variable'):
                var_ops[splits[0]] = 1

nodes_levels = {}
# get nodes levels
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            int_node_level = int(node_and_level[1])
            nodes_levels[node_and_level[0]] = int_node_level

nodes_memory = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_memory[node_name] = int(splitted[1])

nodes_res_memory = {}
# get memory consumption
with open(in6_b, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_res_memory[node_name] = int(splitted[1])





no_of_groups = 0
nodes_groups = {}
with open(in7, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splitted = line.split(' ')
        node_name = splitted[0].lower()
        nodes_groups[node_name] = int(splitted[1])
        if int(splitted[1]) > no_of_groups:
            no_of_groups = int(splitted[1])
        if int(splitted[1]) == -1:
            nodes_groups[node_name] = 0

no_of_groups += 1

collocations = {}
for node in ref_ops.keys():
    for rev_adj in rev_graph[node]:
        if rev_adj in ref_ops:
            if node not in collocations.keys():
                collocations[node] = []
            if not ops_types[rev_adj].endswith(('variable','variablev2')):
                collocations[node].append(str(rev_adj) + ':' + str(ops_types[rev_adj]))

for node, adjs in collocations.items():
    if adjs:
        print(node + '::' + str(adjs))

for g_no in range(0, 4):
  var_count = 0
  ref_count = 0
  res_count = 0
  norm_count = 0

  var_sum = 0
  ref_sum = 0
  res_sum = 0
  norm_sum = 0
  for node, mem in nodes_memory.items():
      if mem > 0 and node in nodes_groups and nodes_groups[node] == g_no:# and nodes_levels[node] < 40000:
          if node in var_ops:
              var_count += 1
              var_sum += mem
          elif node in ref_ops:
              ref_count += 1
              ref_sum += mem
          elif node in nodes_res_memory:
              res_count += 1
              res_sum += mem
              #if mem > 100000000:
                  #print(mem)
          else:
              norm_count += 1
              norm_sum += mem

  print('-----------------------')
  print('var_count: ' + str(var_count)) 
  print('ref_count: ' + str(ref_count)) 
  print('res_count: ' + str(res_count)) 
  print('norm_count: ' + str(norm_count)) 

  print('var_sum: ' + str( var_sum / (1024 * 1024 * 1024) )) 
  print('ref_sum: ' + str(ref_sum / (1024 * 1024 * 1024) )) 
  print('res_sum: ' + str(res_sum / (1024 * 1024 * 1024) )) 
  print('norm_sum: ' + str(norm_sum / (1024 * 1024 * 1024) )) 
  print('-----------------------')
