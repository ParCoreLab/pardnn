import json
import nodeProps
import random
import logging
import os.path
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
loggingLevel = logging.DEBUG if '-v' in sys.argv else logging.INFO
logger.setLevel(loggingLevel)

# folder containing the work files
#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_4800/'

io_folder_path = sys.argv[1] + '/'
# io_folder_path = '/home/endim/pardnn/test/f/' #'/home/nahmad/models/f/abc/inc/wrn/'
network_app = 'crn'

# output file
in1 = io_folder_path + 'nodes_average_durations.txt'

def clean_line_keep_spaces(node_string):
    return (node_string.strip(';\n')).replace('"', '').replace('\t', '')


def clean_line(node_string):
    return (node_string.strip(';\n')).replace('"', '').replace('\t', '').replace(' ', '')


nodes_durations = {}
def read_nodes_durations(filename = in1):
    if (not os.path.isfile(filename)):
        with open(filename, 'w') as f:
            pass
    with open(filename) as f:
        for line in f:
            line = clean_line(line)
            splits = line.split('::')
            nodes_durations[splits[0]] = splits[-1]


def read_profiling_file_v2(_filename):
    profiling_dict = {}
    read_nodes_durations(filename=_filename)
    for node, duration in nodes_durations.items():
        node_properties = nodeProps.NodeProps()
        node_properties.duration = int( round(float(duration)) )
        profiling_dict[node] = node_properties
    return profiling_dict

def read_profiling_file(file_name, averages = False):
    read_nodes_durations()
    profiling_dict = {}
    with open(file_name, 'r') as f:
        profiling_data = json.load(f)
        for tracing_entry in profiling_data['traceEvents']:
            if 'args' in tracing_entry and 'dur' in tracing_entry:
                try:
                    node_name = tracing_entry['args']['name'].lower()
                except:
                    node_name = str(random.randint(1, 50000) +
                                    random.randint(1100, 5000))
                node_properties = nodeProps.NodeProps()
                profiling_dict[node_name] = node_properties
                if 'dur' in tracing_entry:
                    if averages and node_name in nodes_durations:
                        duration = int( round(float(nodes_durations[node_name])) )
                    else:
                        duration = tracing_entry['dur']
                    node_properties.duration = duration
                    start_time = tracing_entry['ts']
                    node_properties.start_time = start_time
                    node_properties.end_time = start_time + duration
    return profiling_dict


def dot_to_mapping(file_name, zeor_based):
    mapping = {}
    with open(file_name, 'r') as f:
        for line in f:
            if '->' in line:
                line = clean_line(line)
                nodes = line.split('->')
                if nodes[0] not in mapping.keys():
                    if zeor_based:
                        mapping[nodes[0]] = len(mapping)
                    else:
                        mapping[nodes[0]] = len(mapping) + 1
                if nodes[1] not in mapping.keys():
                    if zeor_based:
                        mapping[nodes[1]] = len(mapping)
                    else:
                        mapping[nodes[1]] = len(mapping) + 1
    return mapping


def write_dot_of_nodes_at_levels(start_from_level, end_at_level, nodes_levels, dot_file, output_file):
    lines_to_write = []
    with open(dot_file, 'r') as f:
        for line in f:
            if '->' in line:
                line = clean_line(line)
                nodes = line.split('->')
                if nodes_levels[nodes[0]] >= start_from_level and nodes_levels[nodes[0]] < end_at_level and nodes_levels[nodes[1]] >= start_from_level and nodes_levels[nodes[1]] < end_at_level:
                    lines_to_write.append('"' + nodes[0] + '"->' + '"' + nodes[1] + '"')

    with open(output_file, 'w') as f:
        f.write("digraph{\n")
        for line in lines_to_write:
            f.write(line + "\n")
        f.write("\n}")
