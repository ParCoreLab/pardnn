import utils

# folder containing the work files
#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_4800/'

io_folder_path = utils.io_folder_path

ins =  ['timeline_step17.json']#,'tensors_sz_32.txt', 'timeline_step17.json', utils.network_app+'.dot']

for in_i in ins:
    in_i = io_folder_path + in_i
    tmp = []
    with open(in_i, 'r') as f:
        for line in f:
            tmp.append(line.lower())

    out = in_i.split('.')[0] + '_low.' + in_i.split('.')[1]

    with open(out, 'w') as f:
        for line in tmp:
            f.write(line)