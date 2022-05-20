import utils

io_folder_path = utils.io_folder_path

in1 = io_folder_path + 'nodes_devices.txt'

out1 = io_folder_path + 'cpu_nodes.txt'

cpu_nodes = []
with open(in1, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    splits = line.split("::")
    if(len(splits) > 1):
      if (len(splits) == 2 and "cpu" in splits[1]) or splits[0].startswith('add'):
        cpu_nodes.append(splits[0])
        
with open(out1, 'w') as f:
  for node in cpu_nodes:
    f.write(node + "\n")