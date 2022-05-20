import utils
import nodeProps
from graphics import *
import math
import matplotlib.pyplot as plt
import numpy as np

network_app = utils.network_app
io_folder_path = utils.io_folder_path
# input files
in1 = io_folder_path + 'nodes_average_durations_fixed.txt'
# 'nodes_levels.txt'#'part_8_1799_src_sink_nodes_levels.txt'
in2 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'#'part_1_39_src_sink_nodes_levels.txt'
#in3 = io_folder_path + 'fareed/mixed_h_zoltan/mixed_h_zoltan_2_cleaned.place'
in3 = io_folder_path + 'placement.place'#'mixed_placement_v_part_nc.place'#'vanilla_cleaned_low.place'#'mixed_placement_v_part_nc.place'
in4 = io_folder_path + network_app + '_src_sink_low.dot' #'part_1_39_src_sink.dot'  #part_8_1799
in5 = io_folder_path + 'tensors_sz_32_low.txt'
in6 = io_folder_path + 'memory.txt'
in6_b = io_folder_path + 'res_memory.txt'

analysis_graph = utils.read_profiling_file_v2(in1)


class LevelProps:
    def __init__(self, duration=0, level=-1, start_time=-1, end_time=-1, dense_duration=0, num_nodes=0, nodes={}):
        self.duration = duration
        self.level = level
        self.start_time = start_time
        self.end_time = end_time
        self.dense_duration = dense_duration
        self.num_nodes = num_nodes
        self.nodes = {}

    def representation(self):
        return "[ duration: " + str(self.duration) + ', level: ' + str(self.level) + ', start time: ' + str(self.start_time) + ', end_time: ' + str(self.end_time) + "]"


max_level = 0
nodes_levels = {}
# fill the analysis graph nodes levels
with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        node_and_level = line.split("::")
        level = node_and_level[len(node_and_level) - 1]
        nodes_levels[node_and_level[0]] = int(level)
        if node_and_level[0] in analysis_graph:
            analysis_graph[node_and_level[0]].level = level
        if int(level) > int(max_level):
            max_level = level

max_level = int(max_level) + 1

nodes_parts = {}
no_nodes_parts = False

cnt = 0
# fill parts
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' ')
        if len(splits) > 1:
          nodes_parts[splits[0]] = splits[1]
          cnt = cnt + 1
          if splits[0] in analysis_graph.keys():
              analysis_graph[splits[0]].part = splits[1]
# decleare and initialize levels set
levels = {}
for i in range(0, int(max_level)):
    levels[i] = LevelProps()

# fill start end end times of the levels
for node, node_properties in analysis_graph.items():
    node_level = int(node_properties.level)
    if node_level != -1:
        current_level = levels[node_level]
        current_level.num_nodes = current_level.num_nodes + 1
        current_level.dense_duration = current_level.dense_duration + node_properties.duration
        current_level.nodes[node] = node_properties
        if int(node_properties.end_time) > int(current_level.end_time):
            current_level.end_time = node_properties.end_time
        if int(node_properties.start_time) < int(current_level.start_time) or int(current_level.start_time) == -1:
            current_level.start_time = node_properties.start_time


# fill durations of levels:
for level, Level_properties in levels.items():
    Level_properties.duration = Level_properties.end_time - Level_properties.start_time

""" for level, Level_properties in levels.items():
    print(str(level) + " => " + str(Level_properties.duration)) """

offset_x = 20
offset_y = 20
graphics_window_width = 1300
graphics_window_height = 600
levels_to_represent = min(200, int(max_level))
start_from_level = 500
end_at_level = start_from_level + levels_to_represent
first_node_in_the_graph_to_show_start_time = 0
last_node_in_the_graph_to_show_end_time = 8

for i in range(start_from_level, end_at_level):
    if (levels[i].start_time < first_node_in_the_graph_to_show_start_time or first_node_in_the_graph_to_show_start_time == 0) and levels[i].start_time > 0:
        first_node_in_the_graph_to_show_start_time = levels[i].start_time
    if levels[i].end_time > last_node_in_the_graph_to_show_end_time:
        last_node_in_the_graph_to_show_end_time = levels[i].end_time

graph_duration_to_show = float(
    last_node_in_the_graph_to_show_end_time - first_node_in_the_graph_to_show_start_time)


''' win = GraphWin(width=graphics_window_width, height=graphics_window_height)
win2 = GraphWin(width=graphics_window_width, height=graphics_window_height)
win3 = GraphWin(width=graphics_window_width, height=graphics_window_height)
win4 = GraphWin(width=graphics_window_width, height=graphics_window_height)
win5 = GraphWin(width=graphics_window_width, height=graphics_window_height) '''

graphics_window_width = graphics_window_width - 2 * offset_x
graphics_window_height = graphics_window_height - 2 * offset_y

level_height = graphics_window_height / levels_to_represent

''' for i in range(start_from_level, end_at_level):
    rectangle_x = offset_x + graphics_window_width * \
        ((levels[i].start_time - first_node_in_the_graph_to_show_start_time) /
         graph_duration_to_show)
    rectangle_y = offset_y + (i - start_from_level) * level_height
    rectangle_end_x = rectangle_x + \
        (levels[i].duration / graph_duration_to_show) * graphics_window_width
    rectangle_end_y = rectangle_y + level_height
    pt = Point(rectangle_x, rectangle_end_y)
    rect = Rectangle(Point(rectangle_end_x, rectangle_y), pt)
    rect.draw(win)
    rect.setFill(color="#ff0000") '''

inc = graphics_window_width / 30
time_inc = int(round(graph_duration_to_show / (30000)))
line_position = 0
line_time_label = 0
''' for i in range(1, 31):
    line = Line(Point(line_position, 0), Point(
        line_position, graphics_window_height))
    line.setOutline(color='#cccccc')
    line.draw(win)
    lable = Text(Point(line_position, graphics_window_height +
                       1.5 * offset_y), str(line_time_label) + 'ms')
    line_position = line_position + inc
    line_time_label = line_time_label + time_inc
    lable.draw(win) '''

devices_colors = ['red', 'green', 'pink',
                  'yellow', 'black', 'blue', 'gray', 'orange']
devices = ['gpu1', 'gpu2', 'gpu3', 'gpu4', 'cpu', 'add', 'add', 'add']


levels_colors = ["#ff0000"] * int(max_level)
''' for node, node_properties in analysis_graph.items():
    node_level = int(node_properties.level)
    if node_level >= start_from_level and node_level < end_at_level:
        rectangle_x = offset_x + graphics_window_width * \
            ((node_properties.start_time -
              first_node_in_the_graph_to_show_start_time) / graph_duration_to_show)
        rectangle_y = offset_y + (node_level - start_from_level) * level_height
        rectangle_end_x = rectangle_x + \
            (node_properties.duration / graph_duration_to_show) * \
            graphics_window_width
        rectangle_end_y = rectangle_y + level_height
        pt = Point(rectangle_x, rectangle_y)
        rect = Rectangle(Point(rectangle_end_x, rectangle_end_y), pt)
        rect.setOutline(devices_colors[int(node_properties.part)%5])
        rect.draw(win2)
        rect.setFill(devices_colors[int(node_properties.part)%5])
        bottom_line = Line(Point(0, rectangle_end_y), Point(
            graphics_window_width, rectangle_end_y))
        bottom_line.setOutline(color='#cccccc')
        bottom_line.draw(win2)
        level_lable = Text(
            Point(offset_x, rectangle_end_y - level_height / 2), node_level)
        level_lable.draw(win2) '''

# drawing axess
inc = graphics_window_width / 22
time_inc = int(round(graph_duration_to_show / (22)))
line_position = 0
line_time_label = 0
''' for i in range(1, 31):
    line = Line(Point(line_position, 0), Point(
        line_position, graphics_window_height))
    line.setOutline(color='#cccccc')
    line.draw(win2)
    lable = Text(Point(line_position, graphics_window_height +
                       1.5 * offset_y), str(line_time_label) + 'us')
    line_position = line_position + inc
    line_time_label = line_time_label + time_inc
    lable.draw(win2) '''

''' color_legends_dim = 20
for i in range(0, 2 * len(devices_colors), 2):
    color_rect = Rectangle(Point(graphics_window_width, graphics_window_height / 2 + i * color_legends_dim), Point(graphics_window_width + color_legends_dim,
                                                                                                                   graphics_window_height / 2 + (i + 1) * color_legends_dim))
    color_rect.setFill(devices_colors[int(i / 2)%5])
    color_rect.draw(win2)
    lable = Text(Point(graphics_window_width, graphics_window_height /
                       2 + (i + 1.5) * color_legends_dim), devices[int(i / 2)])
    lable.draw(win2)
    color_sepaerator = Line(Point(graphics_window_width - 40, graphics_window_height / 2 + (i + 2) * color_legends_dim), Point(graphics_window_width + color_legends_dim,
                                                                                                                               graphics_window_height / 2 + (i + 2) * color_legends_dim))
    color_sepaerator.draw(win2) '''


levels_current_node_start_points = [0] * max_level
graph_duration_to_show = 0
max_nodes_per_level_to_show = 0
for i in range(start_from_level, end_at_level):
    if levels[i].dense_duration > graph_duration_to_show:
        graph_duration_to_show = levels[i].dense_duration
    if levels[i].num_nodes > max_nodes_per_level_to_show:
        max_nodes_per_level_to_show = levels[i].num_nodes

iii = 0
''' for node, node_properties in analysis_graph.items():
    iii = iii + 1
    node_level = int(node_properties.level)
    if node_level >= start_from_level and node_level < end_at_level:
        rectangle_x = offset_x + graphics_window_width * \
            ((levels_current_node_start_points[node_level]
              ) / graph_duration_to_show)
        rectangle_y = offset_y + (node_level - start_from_level) * level_height
        rectangle_end_x = rectangle_x + \
            (node_properties.duration / graph_duration_to_show) * \
            graphics_window_width
        rectangle_end_y = rectangle_y + level_height
        pt = Point(rectangle_x, rectangle_end_y)
        rect = Rectangle(Point(rectangle_end_x, rectangle_y), pt)
        if iii % 2 == 0:
            rect.setOutline('#ff0000')
            rect.setFill('#ff0000')
        else:
            rect.setOutline('#00ff00')
            rect.setFill('#00ff00')
        rect.draw(win3)
        # rect.setFill(devices_colors[int(node_properties.part)])
        levels_current_node_start_points[node_level] = levels_current_node_start_points[node_level] + \
            node_properties.duration '''


g_node_dim = min(graphics_window_width /
                 (max_nodes_per_level_to_show * 2), level_height)


class NodePlot:
    def __init__(self, node_key='', node_x=0, node_y=0, node_dim=g_node_dim):
        self.node_key = node_key
        self.node_x = node_x
        self.node_y = node_y
        self.node_dim = node_dim


print(g_node_dim)

# will contain the graph as an adgacency list
graph = {}
nodes_ids = {}
# initializing the nodes and adjacencies from the dot file
with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            if no_nodes_parts:
                nodes_parts[nodes[0]] = 0
                nodes_parts[nodes[1]] = 0
            if not nodes[0] in nodes_ids:
                nodes_ids[nodes[0]] = len(nodes_ids)
            if len(nodes) > 1 and not nodes[1] in nodes_ids:
                nodes_ids[nodes[1]] = len(nodes_ids)
            if len(nodes) > 1:
                if nodes[0] in graph:
                    graph[nodes[0]].append(nodes[1])
                else:
                    graph[nodes[0]] = [nodes[1]]


# decleare and initialize levels set
levels = {}
print(max_level)
for i in range(0, int(max_level)):
    levels[i] = LevelProps()

# fill start end end times of the levels
for node, adj in graph.items():
    node_level = int(nodes_levels[node])
    if node_level != -1:
        current_level = levels[node_level]
        current_level.num_nodes = current_level.num_nodes + 1
        current_level.nodes[node] = node_level
    for adj_node in adj:
        node_level = int(nodes_levels[adj_node])
        if node_level != -1:
            current_level = levels[node_level]
            current_level.num_nodes = current_level.num_nodes + 1
            current_level.nodes[adj_node] = node_level

nodes_to_be_plotted = {}
for i in range(start_from_level, end_at_level):
    level_distance_between_nodes = graphics_window_width / \
        (len(levels[i].nodes))
    start_point_x = offset_x + g_node_dim / 2
    for node, node_properties in levels[i].nodes.items():
        current_node_to_be_plotted = NodePlot()
        current_node_to_be_plotted.node_x = start_point_x
        current_node_to_be_plotted.node_y = (
            i + 0.5 - start_from_level) * level_height
        nodes_to_be_plotted[node] = current_node_to_be_plotted
        start_point_x = start_point_x + level_distance_between_nodes

''' for node, node_plot in nodes_to_be_plotted.items():
    cir = Circle(Point(node_plot.node_x, node_plot.node_y), g_node_dim / 4)
    if node in nodes_parts:
        cir.setFill(devices_colors[int(nodes_parts[node])%5])
    else:
        print("missing node")
    if not node in analysis_graph.keys():
        cir.setFill("#ffffff")
    cir.draw(win4) '''


""" nodes_memory = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        nodes_memory[splitted[0]] = splitted[1] """

nodes_res_memory = {}
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_res_memory[node_name] = int(splitted[1])

gvz = {}
devices_colors_gvz = ['red', 'green', 'pink',
                      'yellow', 'white', 'gray', 'blue', 'orange']

nodes_keys_mappig = {}

for i in range(start_from_level + 1, end_at_level):
    for node, node_properties in levels[i].nodes.items():
        for prev_layer_node, prev_layer_node_plot in nodes_to_be_plotted.items():
            if prev_layer_node in graph.keys() and node in graph[prev_layer_node]:
                edge = Line(Point(
                    nodes_to_be_plotted[prev_layer_node].node_x, nodes_to_be_plotted[prev_layer_node].node_y),
                    Point(nodes_to_be_plotted[node].node_x, nodes_to_be_plotted[node].node_y))
                color_indx = ''
                if prev_layer_node in nodes_parts:
                    color_indx = nodes_parts[prev_layer_node]
                elif prev_layer_node.lower() in nodes_parts:
                    color_indx = nodes_parts[prev_layer_node.lower()]
                else:
                    color_indx = 0
                edge.setOutline(devices_colors[int(color_indx)%5])
                #edge.draw(win4)
                from_node_dur = 0
                to_node_dur = 0
                if prev_layer_node in analysis_graph.keys():
                    from_node_dur = analysis_graph[prev_layer_node].duration
                if node in analysis_graph.keys():
                    to_node_dur = analysis_graph[node].duration

                src_key = str(from_node_dur) + '_' + str(
                    nodes_ids[prev_layer_node]) + '\\n' + str(nodes_levels[prev_layer_node]) #+ '\\n' + str(nodes_memory[prev_layer_node])
                nodes_keys_mappig[src_key] = prev_layer_node
                dst_key = str(to_node_dur) + '_' + \
                    str(nodes_ids[node]) + '\\n' + str(nodes_levels[node]) #+ '\\n' + str(nodes_memory[prev_layer_node])
                nodes_keys_mappig[dst_key] = node

                if src_key not in gvz.keys():
                    gvz[src_key] = [dst_key]
                else:
                    gvz[src_key].append(dst_key)


comm_transfer_rate = 1000000 / (130 * 1000 * 1000 * 1000)
comm_latency = 25
tensors_sizes = {}
# get tensors sizes
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensors_sizes[splitted[0]] = splitted[1]


with_tensors = True
with open(io_folder_path + '/vis/' + 'part_' + str(start_from_level) + '_' + str(end_at_level) + '.dot', 'w') as f:
    f.write("digraph{\n")
    for src, dst in gvz.items():
        tensor_size = str(int(int(tensors_sizes[nodes_keys_mappig[src]]) * comm_transfer_rate +
                              comm_latency)) if nodes_keys_mappig[src] in tensors_sizes else '?'
        fill_color = 'white'
        if src in nodes_keys_mappig and nodes_keys_mappig[src] in nodes_parts and int(nodes_parts[nodes_keys_mappig[src]]) >= 0:
          fill_color = (devices_colors_gvz[int(nodes_parts[nodes_keys_mappig[src]])%8])
        f.write(
            '"' + src + '" [style=filled, shape = circle, fillcolor = ' + fill_color + ' tooltip="' + nodes_keys_mappig[src] + '"]\n')
        for dst_item in dst:
          fill_color = 'white'
          print(dst_item)
          if dst_item in nodes_keys_mappig and nodes_keys_mappig[dst_item] in nodes_parts and int(nodes_parts[nodes_keys_mappig[dst_item]]) >= 0:
            fill_color = (devices_colors_gvz[int(nodes_parts[nodes_keys_mappig[dst_item]])%8])
            f.write('"' + str(src) + '"' + ' -> ' +
                    '"' + str(dst_item) + '"' + ('[ label="' + tensor_size + '" ]\n' if with_tensors else '\n'))
            f.write(
                '"' + dst_item + '" [style=filled, shape = circle, fillcolor = ' + fill_color + ' tooltip="' + nodes_keys_mappig[dst_item] + '"]\n')
    f.write("}")

utils.write_dot_of_nodes_at_levels(start_from_level, end_at_level, nodes_levels, in4,
                                   io_folder_path + '/vis/txt_part_' + str(start_from_level) + '_' + str(end_at_level) + '.dot')

#os.system("C:/Documents and Settings/flow_model/flow.exe")

#io_folder_path + '/vis/' + 'part_' + str(start_from_level) + '_' + str(end_at_level) + '.dot', 'w'