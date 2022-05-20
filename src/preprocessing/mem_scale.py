import utils

io_folder_path= utils.io_folder_path

in1 = io_folder_path + 'memory_4.txt'
in2 = io_folder_path + 'memory_2.txt'

out1 = io_folder_path + 'mem_ratios.txt'

mem1 = {}
mem2 = {}
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        if len(splits) > 1:
            mem1[splits[0]] = float(splits[1])

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        if len(splits) > 1:
            mem2[splits[0]] = float(splits[1])

with open(out1, 'w') as f:
    for key, val in mem1.items():
        if mem2[key] > 0:
            f.write( key + '::' + str(val / mem2[key]) + '\n')