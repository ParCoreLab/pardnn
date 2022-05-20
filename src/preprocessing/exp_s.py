import time
import heapq


def fnd_perms(string, length, all_permutations_count):
    if length == 0:
        all_permutations_count[0] = all_permutations_count[0] + 1
    else:
        fnd_perms(string + '0', length - 1, all_permutations_count)
        fnd_perms(string + '1', length - 1, all_permutations_count)


def fnd_perms_stk(length, all_permutations_count):
    stk = []
    heapq.heappush(stk, (0, '') )
    while len(stk) > 0:
        poped_string = heapq.heappop(stk)[1]
        poped_string_0 = poped_string + '0'
        poped_string_1 = poped_string + '1'
        if len(poped_string_0) == length:
            all_permutations_count[0] = all_permutations_count[0] + 2
        else:
            heapq.heappush(stk, (len(poped_string_0), poped_string_0) )
            heapq.heappush(stk, (len(poped_string_1), poped_string_1) )

all_permutations_count = [0]
start_time = time.time()
fnd_perms('', 25, all_permutations_count)

print('Time consumed is: ' + str(time.time() - start_time))
print('We have: ' + str(all_permutations_count[0]) + ' permutations.')

all_permutations_count = [0]
start_time = time.time()
fnd_perms_stk(25, all_permutations_count)

print('Time consumed is: ' + str(time.time() - start_time))
print('We have: ' + str(all_permutations_count[0]) + ' permutations.')


""" with open('permutations.txt', 'w') as f:
    for permutation in all_permutations:
        f.write(permutation + '\n') """
