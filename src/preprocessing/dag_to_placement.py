
import utils

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/'

# input files
in1 = io_folder_path + 'txt_part_20_1799mapping.txt'
in2 = io_folder_path + 'txt_part_20_1799_dagp.dot.partsfile.part_4.seed_0.txt'

out1 = io_folder_path + '20_1799_dagp.place'

nodes_mapping = {}
colors_parts = {'blue': '0', 'red': '1', 'purple': '2', 'green': '3'}
placement = {}


with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        if len(splits) > 1:
            nodes_mapping[splits[1]] = splits[0]

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split(',')
        if(len(splits) > 1):
            fill_color = splits[1].split('=')[1]
            node_indx = line.split('[')[0]
            placement[nodes_mapping[node_indx]] = colors_parts[fill_color]


with open(out1, "w") as f2:
	for key, val in placement.items():
		f2.write(str(key) + " " + str(val) + "\n")