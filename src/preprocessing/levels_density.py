import utils

io_folder_path= 'C:/Users/fareed/PycharmProjects/tf_project/inc/e3d/'

in1 = io_folder_path + 'e3d_src_sink_nodes_levels_low.txt'
in2 = io_folder_path + 'memory.txt'
in3 = io_folder_path + 'timeline_step17_low.json'

out1 = io_folder_path + 'levels_densities_6.txt'

analysis_graph = utils.read_profiling_file_v2(in3)

mem_sum = 0
nodes_memory = {}
mem_hist = {}
# get memory consumption
with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        nodes_memory[splitted[0]] = int(splitted[1])
        if float(splitted[1]) / (1024 * 1024) not in mem_hist:
            mem_hist[float(splitted[1]) / (1024 * 1024)] = 0
        mem_hist[float(splitted[1]) / (1024 * 1024)] += 1

lst1 = mem_hist.keys()
lst2 = mem_hist.values()
lst1, lst2 = (list(t) for t in zip(
                *sorted(zip( lst1, lst2), reverse=True)))
""" for i in range(0, len(lst1)):
    print(str(lst1[i]) + ' : ' + str(lst2[i])) """

all_work = 0
levels_density = {}
levels_density_memory = {}
levels_densities_weight = {}
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        if len(splits) > 1:
            if int(splits[1]) in levels_density:
                levels_density[int(splits[1])] += 1
                levels_density_memory[int(splits[1])] += int(nodes_memory[splits[0]]) if splits[0] in nodes_memory else 0
                levels_densities_weight[int(splits[1])] += int(analysis_graph[splits[0]].duration) if splits[0] in analysis_graph else 0
            else:
                levels_density[int(splits[1])] = 1
                levels_density_memory[int(splits[1])] = int(nodes_memory[splits[0]]) if splits[0] in nodes_memory else 0
                levels_densities_weight[int(splits[1])] = int(analysis_graph[splits[0]].duration) if splits[0] in analysis_graph else 0
            all_work += analysis_graph[splits[0]].duration if splits[0] in analysis_graph else 0
        #mem_sum += int(nodes_memory[splits[0]])

print("total memory consumption of the model is: " + str(mem_sum))

no_levels = len(levels_density)
densities_hist = [0] * 33
for level, density in levels_density.items():
    for i in range(min(32, density), 0, -1):
        densities_hist[i] += 1

""" for i in range(1, 6):
    print(str(pow(2, i)) + ' : ' + str(densities_hist[pow(2, i)] / no_levels)) """

""" for i in range(1, 32):
  print(str(i) + '::' + str(densities_hist[i] / no_levels)) """

for level, density in levels_density.items():
  print(level, density)

levels = []
densities = []
weights = []
densities_memory = []
weight_ratios = []
for level in levels_density.keys():
    levels.append(level)
    densities.append(levels_density[level])
    densities_memory.append(levels_density_memory[level])
    weight_ratios.append((levels_densities_weight[level] / all_work) if level in levels_densities_weight else 0)
    weights.append(levels_densities_weight[level] if level in levels_densities_weight else 0)

densities, weights, weight_ratios, levels, densities_memory = (list(t) for t in zip(
                *sorted(zip( densities, weights, weight_ratios, levels, densities_memory), reverse=True)))

with open(out1, 'w') as f:
    for i in range(0, len(levels)):
        f.write('-' + str(levels[i]) +'::\t' + str(densities[i]) + '::\t' + str(densities_memory[i]/1000000000 if i in levels_density_memory else 0) + \
          '::\t' + str(weights[i]/1000 if i in levels_densities_weight else 0) + '::\t' + str(weight_ratios[i]) + '\n')


densities_memory, levels = (list(t) for t in zip(
                *sorted(zip( densities_memory, levels), reverse=True)))

''' for i in range(10):
  print(str(densities_memory[i]/1000000000) + '::' + str(levels[i])) '''