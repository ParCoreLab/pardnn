import utils

# folder containing the work files
io_folder_path = utils.io_folder_path
in2 = io_folder_path + 'operations_attributes.txt'
in3 = io_folder_path + 'nodes_average_durations.txt'
in4 = io_folder_path + 'operations_attributes.txt'
in6 = io_folder_path + 'memory.txt'

nodes_memory = {}
nodes_durations = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_memory[node_name] = int(splitted[1])
        # if '^' + node_name in all_nodes:
        #    nodes_memory['^' + node_name] = int(splitted[1])

with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_durations[node_name] = int(splitted[1])

no_ops = {}
ref_ops = {}
var_ops = {}
ops_types = {}
with open(in2, 'r') as f:
    for line in f:
        splits = utils.clean_line(line).lower().split('::')
        ops_types[splits[0]] = splits[1]
        if splits[1] == 'noop':
            no_ops[splits[0]] = 1
        elif splits[1] in ['variablev2', 'variable']:
            var_ops[splits[0]] = 1
        if len(splits) > 2 and splits[2] == 'true' or 'isvariableinitialized' in splits[0]:
            if splits[0] not in var_ops:
                ref_ops[splits[0]] = 1

types_consumptions = {}
for node, mem in nodes_memory.items():
    if node in ops_types:
        if ops_types[node] not in types_consumptions:
            types_consumptions[ops_types[node]] = 0
        types_consumptions[ops_types[node]] += mem

types_durations = {}
for node, mem in nodes_durations.items():
    if node in ops_types:
        if ops_types[node] not in types_durations:
            types_durations[ops_types[node]] = 0
        types_durations[ops_types[node]] += mem

types = types_consumptions.keys()
consumptions = types_consumptions.values()

consumptions, types = (list(t) for t in zip(
    *sorted(zip(consumptions, types), reverse=True)))

total_cons = sum(consumptions)
print('running time breakdown: ')
for i in range(0, len(types)):
    print(types[i] + '::' + str(consumptions[i] / (1024 * 1024 * 1024)
                                ) + '::' + str(consumptions[i] / total_cons))

types = types_durations.keys()
durations = types_durations.values()


durations, types = (list(t) for t in zip(
    *sorted(zip(durations, types), reverse=True)))

total_time = sum(durations)

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
for i in range(0, len(types)):
    print(types[i] + '::' + str(durations[i] / (1000)
                                ) + '::' + str(durations[i] / total_time))

types_counts = {}
# get memory consumption
with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        if len(splits) > 1:
            if splits[1] not in types_counts:
                types_counts[splits[1]] = 0
            types_counts[splits[1]] += 1

print("#####################")

for type, count in types_counts.items():
  print(type + "::" + str(count))