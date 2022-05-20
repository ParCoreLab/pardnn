import utils
import matplotlib.pyplot as plt
import numpy as np

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/'

# input files
in1 = io_folder_path + 'tensors_sz.txt'

tensors_map = {}

summ = 0
cntt = 0

with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        tensors_map[splits[0]] = int(splits[1])
        summ = summ + int(splits[1])
        cntt = cntt + 1

print((summ * 1000000 / (9 * 1024 * 1024 * 1024)) / cntt)

tensors_statistics = {}


for tensor, size in tensors_map.items():
    if size in tensors_statistics.keys():
        tensors_statistics[size] = tensors_statistics[size] + 1
    else:
        tensors_statistics[size] = 1

print(tensors_statistics[177020928])
print(tensors_statistics[7102464])

c = plt.figure(3)
y_pos = list(tensors_statistics.keys())
vals = tensors_statistics.values()
y_pos, vals = (list(t) for t in zip(*sorted(zip(y_pos, vals))))

#running times
y_pos = [int(x * 1000000 / (9 * 1024 * 1024 * 1024)) for x in y_pos]

y_pos = list(map(str, y_pos))

#large tensors
y_pos = y_pos[int(len(y_pos)/2):len(y_pos)]
vals = vals[int(len(vals)/2):len(vals)]

plt.bar(y_pos, vals, align='center',  width=0.9)
#plt.yscale("log")
plt.grid()
plt.xlabel('Comm time in us', fontsize=18)
plt.ylabel('Count', fontsize=16)
plt.xticks(rotation=90)
c.show()

for i, v in enumerate(vals):
    plt.text(y= v + 3,x= i - 0.5, s= str(v), color='blue', fontweight='bold')

input()
