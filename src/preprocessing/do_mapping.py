import utils

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/'

# input files
in1 = io_folder_path + 'part_8_1799.dot'

#output
out1 = io_folder_path + 'txt_part_8_1799mapping.txt'

mapping = utils.dot_to_mapping(in1, False)

with open(out1, 'w') as f:
    for key, val in mapping.items():
        f.write(key + '::' + str(val) + '\n')