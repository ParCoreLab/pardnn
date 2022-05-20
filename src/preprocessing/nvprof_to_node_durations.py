import utils
import csv
import math

io_folder_path = utils.io_folder_path
in1 = io_folder_path + 'nodes_average_durations.txt'
in2 = io_folder_path + 'durations.csv'
in3 = io_folder_path + 'actual_nodes.txt'
in4 = io_folder_path + 'vanilla_cleaned.place'
in5 = io_folder_path + 'var_nodes.txt'
in6 = io_folder_path + utils.network_app + '_src_sink_nodes_levels_low.txt'

cpu_nodes = {}
with open(in4, 'r') as f:
  for line in f:
    line = utils.clean_line_keep_spaces(line)
    splits = line.split(' ')
    if splits[1] == '-1':
      cpu_nodes[splits[0]] = 1

var_nodes = {}
with open(in5 , 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    var_nodes[line] = splits[1]

nodes = {}
with open(in6 , 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) == 2:
      nodes[splits[0]] = splits[1]

nodes_durations_tf = {}
nodes_tf = []
matmul_cntt = 0
start =False
with open(in1 , 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) == 2 and splits[0] not in cpu_nodes and splits[0] not in var_nodes and splits[0] in nodes:
      nodes_tf.append(splits[0])
      if 'matmul' in splits[0].split('/')[-1]:
        matmul_cntt += 1
        start = True
      #if start:  
      nodes_durations_tf[splits[0]] = splits[1]

nodes_durations_nvprof = []
artificial_delay_between_nodes = 5
artificial_delay_between_iterations = 5000
gemm_cntt = 0
start = False
cntt = 0
with open(in2, newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  previous_start = math.inf
  set_of_kernels = []
  set_of_kernels_duration = 0
  effective_iterations_started = 0
  row_num = 0
  for row in reader:
    row_num += 1
    node_start = float(row['Start']) * 1000 #convert from ms to s
    if node_start >= previous_start + artificial_delay_between_iterations:
      effective_iterations_started += 1
      print(row_num, row['Name'])
    if effective_iterations_started == 4:
      if start and  '[CUDA memcpy DtoH]' not in row['Name'] and '[CUDA memcpy HtoD]' not in row['Name']:
        cntt += 1
      if 'gemm' in row['Name'].lower():
        gemm_cntt+=1
        start = True
      if node_start >= previous_start + artificial_delay_between_nodes:
        nodes_durations_nvprof.append([set_of_kernels_duration, set_of_kernels])
        set_of_kernels = []
        set_of_kernels_duration = 0
      set_of_kernels.append(row['Name'])
      set_of_kernels_duration += float(row['Duration']) * 1000 #convert ms to us
    previous_dur =  float(row['Duration'])
    previous_start = node_start
          
actual_nodes = {}
start_reading = 0
with open(in3, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    if '********' in line: 
      start_reading += 1
      continue
    if start_reading == 3 and line not in cpu_nodes:# and line in nodes_durations_tf:
      actual_nodes[line] = 1
  
print(len(nodes_durations_nvprof), len(nodes_durations_tf), len(actual_nodes), cntt, gemm_cntt, matmul_cntt)

''' for i in range(0, 1500):
  print(nodes_tf[i], nodes_durations_nvprof[i][1]) '''