io_folder_path = './trn_2_2048_4096_12_16_128_bsz_8/'

graph_file_name = io_folder_path + 'graph.dot'
weights_file_name = io_folder_path + 'weights.txt'
costs_file_name = io_folder_path + 'costs.txt'
var_nodes_file_name = io_folder_path + 'var_nodes.txt'
ref_nodes_file_name = io_folder_path + 'ref_nodes.txt'
no_ops_file_name = io_folder_path + 'no_ops.txt'
metis_input_file_name = io_folder_path + 'metis_graph.txt'
metis_parts_file_name = io_folder_path + 'metis_graph.txt.part.4'
vanilla_file_name = io_folder_path + 'vanilla_cleaned.place'
output_file_name = '/home/nahmad/placement.place'

comm_latency = 25
comm_transfer_rate_rec = 1.0 / (130000)


def clean_line(node_string):
    return (node_string.replace('\n', '')).replace('\r', '').replace(
            '"', '').replace('\t', '').replace(' ', '')


def clean_line_keep_spaces(node_string):
    return node_string.replace('\n',
            '').replace('\r',
                    '').replace('"',
                            '').replace('\t', '')


number_of_edgse = 0
graph = {}
nodes_ranks_map = {}
rev_nodes_ranks_map = {}
number_of_nodes = 0
nodes_weights = {}
edges_costs = {}

with open(graph_file_name, 'r') as f:
    for line in f:
        line = clean_line(line)
        splits = line.split('->')
        if len(splits) > 1:
            number_of_edgse += 1
            if splits[0] not in graph:
                graph[splits[0]] = []
                nodes_ranks_map[splits[0]] = number_of_nodes + 1
                rev_nodes_ranks_map[number_of_nodes + 1] = splits[0]
                number_of_nodes += 1
            if splits[1] not in graph:
                graph[splits[1]] = []
                nodes_ranks_map[splits[1]] = number_of_nodes + 1
                rev_nodes_ranks_map[number_of_nodes + 1] = splits[1]
                number_of_nodes += 1
            graph[splits[0]].append(splits[1])
            graph[splits[1]].append(splits[0])

with open(weights_file_name, 'r') as f:
    for line in f:
        line = clean_line(line)
        splits = line.split('::')
        if (len(splits) > 1):
            nodes_weights[splits[0]] = splits[1]

ref_nodes = {}
with open(var_nodes_file_name, 'r') as f:
    for line in f:
        ref_nodes[clean_line(line)] = 1

var_nodes = {}
with open(ref_nodes_file_name, 'r') as f:
    for line in f:
        var_nodes[clean_line(line)] = 1

no_op_nodes = {}
with open(no_ops_file_name, 'r') as f:
    for line in f:
        no_op_nodes[clean_line(line)] = 1

cntt = 0
with open(costs_file_name, 'r') as f:
    for line in f:
        line = clean_line(line)
        splits = line.split('::')
        if (len(splits) > 1):
            node_name = splits[0]
            cntt += 1
            if node_name in graph:
                if node_name not in edges_costs:
                    edges_costs[node_name] = {}
                for adj_node in graph[node_name]:
                    if adj_node not in edges_costs:
                        edges_costs[adj_node] = {}
                    if adj_node in no_op_nodes:
                        edges_costs[node_name][adj_node] = comm_latency
                        edges_costs[adj_node][node_name] = comm_latency
                    else:
                        edges_costs[node_name][adj_node] = int(
                                float(splits[1]) * comm_transfer_rate_rec +
                                comm_latency)
                        edges_costs[adj_node][node_name] = comm_latency

with open(metis_input_file_name, 'w') as f:
    f.write(str(number_of_nodes) + ' ' + str(number_of_edgse) + ' 11\n')
    for node in graph.keys():
        line_to_write = str(
                nodes_weights[node]) if node in nodes_weights else '1'
        for adj_node in graph[node]:
            line_to_write += ' ' + str(nodes_ranks_map[adj_node])
            line_to_write += ' ' + str(edges_costs[node][adj_node]) if node in edges_costs and adj_node \
                    in edges_costs[node] else str(comm_latency)

        line_to_write += '\n'
        f.write(line_to_write)

nodes_parts = {}
i = 0
with open(metis_parts_file_name, 'r') as f:
    for line in f:
        nodes_parts[rev_nodes_ranks_map[i + 1]] = int(clean_line(line))
        i += 1

changed = {}
for node_name in graph.keys():
    if node_name not in var_nodes or node_name in changed:
        continue

    to_visit = []
    node_part = nodes_parts[node_name]
    to_visit.append(node_name)
    while len(to_visit) > 0:
        current_node_name = to_visit.pop(0)
        changed[current_node_name] = True
        for adj_node in graph[current_node_name]:
            if adj_node in ref_nodes:
                nodes_parts[adj_node] = node_part
                for rev_adj_node in graph[adj_node]:
                    if (rev_adj_node not in changed
                            and rev_adj_node in var_nodes):
                        nodes_parts[rev_adj_node] = node_part
                        to_visit.append(rev_adj_node)
                        changed[rev_adj_node] = True

vanilla_placement = {}
with open(vanilla_file_name, 'r') as f:
    for line in f:
        line = clean_line_keep_spaces(line).lower()
        splits = line.split(' ')
        vanilla_placement[splits[0]] = int(splits[1])

with open(output_file_name, 'w') as f:
    for node, part in nodes_parts.items():
        if node in vanilla_placement and vanilla_placement[node] == -1:
            f.write(node + ' ' + str(-1) + '\n')
        else:
            f.write(node + ' ' + str(part) + '\n')
