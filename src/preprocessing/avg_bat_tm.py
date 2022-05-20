import utils

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/batch_32/'

#in1 = io_folder_path + 'sec_batch_vp_v.txt'
#in2 = io_folder_path + 'sec_batch_vp_whole.txt'
in3 = io_folder_path + 'sec_batch_vp_gpu.txt'
in4 = io_folder_path + 'sec_batch_gpu.txt'
in5 = io_folder_path + 'dfs_batch_gpu.txt'


time = 0.0
count = 0

with open(in4) as f:
    for line in f:
        line = utils.clean_line(line)
        time = time + float(line.split('examples/sec;')[1].split('sec/batch')[0])
        count = count + 1
    print('**gpu running time is: ' + str(time/count))

time = 0.0
count = 0

with open(in3) as f:
    for line in f:
        line = utils.clean_line(line)
        time = time + float(line.split('examples/sec;')[1].split('sec/batch')[0])
        count = count + 1
    print('**vertical partitioner gpus running time is: ' + str(time/count))

time = 0.0
count = 0

time = 0.0
count = 0

with open(in5) as f:
    for line in f:
        line = utils.clean_line(line)
        time = time + float(line.split('examples/sec;')[1].split('sec/batch')[0])
        count = count + 1
    print('dfs running time is: ' + str(time/count))

"""with open(in2) as f:
    for line in f:
        line = utils.clean_line(line)
        time = time + float(line.split('examples/sec;')[1].split('sec/batch')[0])
        count = count + 1
    print('vertical partitioner whole running time is: ' + str(time/count))

time = 0.0
count = 0

with open(in1) as f:
    for line in f:
        line = utils.clean_line(line)
        time = time + float(line.split('examples/sec;')[1].split('sec/batch')[0])
        count = count + 1
    print('vertical partitioner 30: ' + str(time/count)) """