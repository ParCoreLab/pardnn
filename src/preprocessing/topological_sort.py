import json
import utils
import queue
import logging

logger = logging.getLogger()

# folder containing the work files
#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_4800/'

io_folder_path = utils.io_folder_path

# input files
ins = [io_folder_path + utils.network_app + '_src_sink_low.dot', io_folder_path + 'rev_' + utils.network_app + '_src_sink_low.dot']

# output files
outs = [io_folder_path + utils.network_app + '_src_sink_nodes_levels_low.txt',
        io_folder_path + 'rev_' + utils.network_app + '_src_sink_nodes_levels_low.txt']
for i in range(0, len(outs)):
    # will contain the graph as an adgacency list
    graph = {}
    # will contain the nodes and their levels
    nodes_levels = {}

    # constructing the graph and initializing the nodes levels from the dot file
    with open(ins[i], 'r') as f:
        for line in f:
            line = utils.clean_line(line)
            nodes = line.split("->")
            if len(nodes) > 1:
                """ if not nodes[0] in nodes_levels:
                    nodes_levels[nodes[0]] = -1
                if not nodes[1] in nodes_levels:
                    nodes_levels[nodes[1]] = -1 """
                if nodes[0] in graph:
                    graph[nodes[0]].append(nodes[1])
                else:
                    graph[nodes[0]] = [nodes[1]]

    src_nodes_map = {}
    src_nodes = queue.Queue()
    in_degrees = {}
    for node_name in graph.keys():
        src_nodes_map[node_name] = 1
    for adjs in graph.values():
        for adj in adjs:
            nodes_levels[adj] = 0
            if adj in src_nodes_map:
                src_nodes_map[adj] = 0
            if adj in in_degrees:
                in_degrees[adj] += 1
            else:
                in_degrees[adj] = 1

    for key, val in src_nodes_map.items():
        if val == 1:
            # print(key)
            logger.debug(key)
            src_nodes.put(key)
            nodes_levels[key] = 0
    # print('---------')
    logger.debug('---------')
    # topological sort
    while not src_nodes.empty():
        current_node = src_nodes.get()
        current_level = nodes_levels[current_node]
        if current_node == "birnn/stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/bw/while/identity_1":
                # print(graph[current_node]) 
                logger.debug(graph[current_node])
        if current_node in graph.keys():
            for adj in graph[current_node]:
                if adj == "birnn/stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/bw/while/merge_1":
                    # print(current_node)
                    # print(in_degrees[adj])
                    logger.debug(current_node)
                    logger.debug(in_degrees[adj])
                in_degrees[adj] -= 1
                if in_degrees[adj] == 0:
                    src_nodes.put(adj)
                if current_level >= nodes_levels[adj]:
                    nodes_levels[adj] = current_level + 1

    """ for node_name in graph.keys():
        if nodes_levels[node_name] == -1:
            nodes_levels[node_name] = 0
            visit.put(node_name)
            while not visit.empty():
                curr_node = visit.get()
                curr_level = nodes_levels[curr_node]
                for adj in graph[curr_node]:
                    if curr_level >= nodes_levels[adj]:
                        nodes_levels[adj] = curr_level + 1
                        if adj in graph:
                            visit.put(adj) """

    # writing results to file
    with open(outs[i], 'w') as f:
        for node_name, node_level in nodes_levels.items():
            f.write(node_name + "::" + str(node_level) + "\n")
