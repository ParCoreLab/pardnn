import csv
import utils
import nodeProps

# folder containing the work files
io_folder_path = utils.io_folder_path
in2 = io_folder_path + 'durations.csv'
in3 = io_folder_path + 'timeline_step6.json'
#in4 = io_folder_path + 'operations_attributes.txt'

train_steps = 8
wanted_step = 6
#just getting num of lines to skip
all_nodes = []
pattern_len = 500
last_node_pattern = [''] * pattern_len

with open(in2, newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    all_nodes.append(row['Name'])
    
for i in range(0, pattern_len):
  last_node_pattern[pattern_len - i - 1] = all_nodes[-i - 1].lower()

unwanted_nodes = []#['[CUDA memcpy HtoD]','[CUDA memset]','[CUDA memcpy DtoH]', '[CUDA memcpy DtoD]']
common_types = ['gemm', 'TensorShufflingOp', 'tensorbroadcastingop', 'devicesegmentedreducekernel', 'swapdimension', 'columnreducekernel',\
  'devicereducesingletilekernel', 'devicereducekernel']
for i in range(0, len(common_types)): 
  common_types[i] = common_types[i].lower()
for i in range(0, len(unwanted_nodes)): 
  unwanted_nodes[i] = unwanted_nodes[i].lower()
types_durations = {}
nodes_durations = {}
types_counts = {}
count = 0
match = 0
iteration = 0
summ = 0
summ_gemm = 0
summ_gemm_40 = 0
with open(in2, newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    node_name = row['Name'].lower()
    if node_name == last_node_pattern[match]:
      match += 1
    else:
      match = 0
    if match > pattern_len - 10:
      iteration += 1
      match = 0
      print('iter', iteration)
    for common_type in common_types:
      if common_type in node_name:
        node_name = common_type
        break
    if node_name not in unwanted_nodes: #and count > num_rows
      if 1==1:#iteration == wanted_step:
        nodes_durations[node_name + '_' + str(count)] = int(float(row['Duration']) * 1000)
        summ +=  int(float(row['Duration']) * 1000)
        if 'gemm' in node_name:
          summ_gemm +=  int(float(row['Duration']) * 1000)
          if int(float(row['Duration'])) >= 12:
            summ_gemm_40 +=  int(float(row['Duration']))
        if node_name not in types_durations:
          types_durations[node_name] = 0
          types_counts[node_name] = 0
        types_durations[node_name] += int(float(row['Duration']) * 1000)
        types_counts[node_name] += 1
    
    #if node_name == last_node:
    #  iteration += 1
    count += 1

nodes_durations_tf = utils.read_profiling_file(in3)

nodes_nvprof = list(nodes_durations.keys())
nodes_tf = list(nodes_durations_tf.keys())

print("Ratio of Matmul:", summ_gemm / summ)
print(summ_gemm_40 / summ)

gemms = []
matmuls = []
for i in range(len(nodes_nvprof)):
  if 'gem' in nodes_nvprof[i]:
    gemms.append(nodes_nvprof[i])
    
for i in range(len(nodes_tf)):
  if len(nodes_tf[i].split('/')) > 1 and 'matmul' in nodes_tf[i].split('/')[-1]:
    matmuls.append(nodes_tf[i])

types = types_durations.keys()
durations = types_durations.values()

durations, types = (list(t) for t in zip(
    *sorted(zip(durations, types), reverse=True)))

summ_durations = sum(durations)
print('nvprof node types and durations:')
for i in range(0, len(types)):
  print(durations[i] / summ_durations, durations[i], types[i])

print('summ durations: ', summ_durations/1000000)

types_durations_tf = {}
''' with open(in4, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    splits = line.split('::')
    if len(splits) > 1:
      node_name = splits[0]
      node_type = splits[1]
      if node_type not in types_durations_tf:
        types_durations_tf[node_type] = 0
      if node_name in nodes_durations_tf:
        types_durations_tf[node_type] += nodes_durations_tf[node_name].duration '''

types_counts_tf = {}
letters = 'abcdefghijklmnopqrstuvwxyz'
for node_name in nodes_tf:
  if nodes_durations_tf[node_name].duration <= 1:
    continue
  node_type = node_name.split('/')[-1]
  i = len(node_type) - 1
  while i > 0:
    if node_type[i] in letters:
      break
    i -= 1
  node_type = node_type[0: i + 1]
  if node_type not in types_durations_tf:
    types_durations_tf[node_type] = 0
    types_counts_tf[node_type] = 0
  types_durations_tf[node_type] += nodes_durations_tf[node_name].duration
  types_counts_tf[node_type] += 1

types = types_durations_tf.keys()
durations = types_durations_tf.values()

durations, types = (list(t) for t in zip(
    *sorted(zip(durations, types), reverse=True)))

summ_durations = sum(durations)
print('tf summ durations: ', summ_durations/1000000)

print('tf node types and durations:')
for i in range(0, len(types)):
  print(durations[i] / summ_durations, durations[i], types[i])
  
print('summ durations: ', summ_durations/1000000)

print('number of gemms', len(gemms) / 56)
print('number of matmuls', len(matmuls))

print('frequencies: ')

types = types_counts.keys()
counts = types_counts.values()

types_tf = types_counts_tf.keys()
counts_tf = types_counts_tf.values()

counts_tf, types_tf = (list(t) for t in zip(
    *sorted(zip(counts_tf, types_tf), reverse=True)))

counts, types = (list(t) for t in zip(
    *sorted(zip(counts, types), reverse=True)))

for i in range(0, min(len(types), len(types_tf))):
  print(counts_tf[i], counts[i] / 8, types_tf[i], types[i])