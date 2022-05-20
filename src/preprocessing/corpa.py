import utils

in1 = utils.io_folder_path + 'orig_train.en.bpe'
in2 = utils.io_folder_path + 'orig_train.de.bpe'

out1 = utils.io_folder_path + 'train.en.bpe'
out2 = utils.io_folder_path + 'train.de.bpe'

arr = []
lens = []
indices = []
file_len = 0
with open(in1, encoding='utf-8', mode='r') as f:
    for line in f:
        if line != '\n':
            indices.append(file_len)
            file_len += 1
            arr.append(line)
            lens.append(len(line))

lens, arr, indices = (list(t) for t in zip(
    *sorted(zip(lens, arr, indices), reverse=True)))

with open(out1, encoding='utf-8', mode='w') as f:
    for i in range(0, file_len):
        f.write(arr[i])

arr = []
with open(in2, encoding='utf-8', mode='r') as f:
    for line in f:
        if line != '\n':
            arr.append(line)

with open(out2, encoding='utf-8', mode='w') as f:
    for i in range(0, file_len):
        f.write(arr[indices[i]])