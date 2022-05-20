import utils
import sys
import logging

#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/'

io_folder_path = utils.io_folder_path

in1 = io_folder_path + 'act_place.place'
#in2 = io_folder_path + 'nodes_devices.txt'

out = io_folder_path + 'vanilla_cleaned.place'
#out2 = io_folder_path + 'cpu_nodes.txt'

def clean_line(node_string):
    return (node_string.strip(';\n')).replace('"', '').replace('\t', '')

actual_placement = {}
# print(in1)

logger = logging.getLogger()
logger.debug(in1)

with open(in1, 'r', encoding="utf8") as f:
    for line in f:
        line = clean_line(line)
        line = line.lower()
        splits = line.split(" ")
        device_n = ''
        if len(splits) > 2:
            device = splits[2]
            if device.endswith('cpu:0'):
                device_n = '-1'
            elif device.endswith('ipu:0'):
                device_n = '0'
            elif device.endswith('ipu:1'):
                device_n = '1'
            elif device.endswith('ipu:2'):
                device_n = '2'
            elif device.endswith('ipu:3'):
                device_n = '3'

            elif device.endswith('gpu:0'):
                device_n = '0'
            elif device.endswith('gpu:1'):
                device_n = '1'
            elif device.endswith('gpu:2'):
                device_n = '2'
            elif device.endswith('gpu:3'):
                device_n = '3'
            elif device.endswith('gpu:4'):
                device_n = '4'
            elif device.endswith('gpu:5'):
                device_n = '5'
            elif device.endswith('gpu:6'):
                device_n = '6'
            elif device.endswith('gpu:7'):
                device_n = '7'
            elif device.endswith('gpu:8'):
                device_n = '8'
            elif device.endswith('gpu:9'):
                device_n = '9'
            elif device.endswith('gpu:10'):
                device_n = '10'
            elif device.endswith('gpu:11'):
                device_n = '11'
            elif device.endswith('gpu:12'):
                device_n = '12'
            elif device.endswith('gpu:13'):
                device_n = '13'
            elif device.endswith('gpu:14'):
                device_n = '14'
            elif device.endswith('gpu:15'):
                device_n = '15'
            
            if device_n != '' and (splits[0][:-1] not in actual_placement or actual_placement[splits[0][:-1]] == '-1'):
                actual_placement[splits[0][:-1]] = device_n

cpu_nodes = []
""" with open(in2, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split("::")
    if(len(splits) > 1):
      if len(splits) == 3 and "cpu" in splits[1].lower():
        cpu_nodes.append(splits[0].lower()) """
        
with open(out, 'w') as f:
    for key, val in actual_placement.items():
        f.write(key + ' ' + val + '\n')
        #if val == '-1' and key not in cpu_nodes:
        #  print(key) 

""" with open(out2, 'w') as f:
    for key, val in actual_placement.items():
        if val == '-1':
            f.write(key + ' ' + val + '\n') """
