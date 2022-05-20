import utils
import logging
from os import walk

logger = logging.getLogger()

# folder containing the work files
#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_4800/'

io_folder_path = utils.io_folder_path

# output file
out1 = io_folder_path + 'nodes_average_durations.txt'

files = []
for (dirpath, dirnames, filenames) in walk(io_folder_path):
    files.extend(filenames)
    break

# getting time (weight) info for nodes
nodes_durations = {}
for file in files:
    if 'json' in file and 'low' not in file and not 'tensor' in file and not 'mem' in file:
        analysis_graph = utils.read_profiling_file(io_folder_path + file)
        for node in analysis_graph:
            if node in nodes_durations:
                nodes_durations[node].append(analysis_graph[node].duration)
            else:
                nodes_durations[node] = [analysis_graph[node].duration]

for node, running_times in nodes_durations.items():
    nodes_durations[node].sort()
    if len(nodes_durations[node]) > 4:
        nodes_durations[node] = nodes_durations[node][2:len(
            nodes_durations[node]) - 2]

with open(out1, 'w+') as f:
    for node, running_times in nodes_durations.items():
        mean = int(sum(nodes_durations[node]) / len(nodes_durations[node]))
        median = int(nodes_durations[node]
                     [int(len(nodes_durations[node]) / 2)])
        to_write = 0
        if mean >= 1.5 * median or mean <= median / 1.5:
            to_write = int((median + int(nodes_durations[node][int(len(nodes_durations[node]) / 4)]) + int(
                nodes_durations[node][int(3 * len(nodes_durations[node]) / 4)])) / 3)
            if len(nodes_durations[node]) >= 4 and nodes_durations[node][int(2 * len(nodes_durations[node]) / 3)] >= 2 * median:
                # print(node + ', ' + str(nodes_durations[node]) + ', The mean is: ' + str(
                #     mean) + ', The median is:' + str(median) + ' ' + str(to_write))
                logger.debug(node + ', ' + str(nodes_durations[node]) + ', The mean is: ' + str(
                    mean) + ', The median is:' + str(median) + ' ' + str(to_write))
        else:
            to_write = int((mean + median) / 2)
        f.write(node.lower() + '::' + str(to_write) + '\n')
# print(len(nodes_durations))
logger.debug(len(nodes_durations))
