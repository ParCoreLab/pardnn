#include "include/node.h"

Node::Node() {}
Node::Node(string _name) {
    name = _name;
    duration = 1;
    memory = 1;
    level = -1;
    reversed_level = -1;
    top_level = -1;
    bottom_level = -1;
    weighted_level = -1;
    start_time = -1;
    end_time = -1;
    part = -1;
    type = NORM_OP;
}
Node::Node(string name, int _duration, long long _memory, int _level,
        int _reversed_level, int _top_level, int _bottom_level,
        int _weighted_level, int _start_time, int _end_time, int _part,
        NODE_TYPES _type) {
    duration = _duration;
    memory = _memory;
    level = _level;
    reversed_level = _reversed_level;
    top_level = _top_level;
    bottom_level = _bottom_level;
    weighted_level = _weighted_level;
    start_time = _start_time;
    end_time = _end_time;
    part = _part;
    type = _type;
}
string Node::get_name() const { return name; }
int Node::get_duration() const { return duration; }
long long Node::get_memory() const { return memory; }
int Node::get_level() const { return level; }
int Node::get_reversed_level() const { return reversed_level; }
int Node::get_weighted_level() const { return weighted_level; }
int Node::get_top_level() const { return top_level; }
int Node::get_bottom_level() const { return bottom_level; }
int Node::get_start_time() const { return start_time; }
int Node::get_end_time() const { return end_time; }
int Node::get_part() const { return part; }
Node::NODE_TYPES Node::get_type() const { return type; }
void Node::set_duration(int _duration) { duration = _duration; }
void Node::set_memory(long long _memory) { memory = _memory; }
void Node::set_level(int _level) { level = _level; }
void Node::set_reversed_level(int _reversed_level) {
    reversed_level = _reversed_level;
}
void Node::set_top_level(int _top_level) { top_level = _top_level; }
void Node::set_bottom_level(int _bottom_level) { bottom_level = _bottom_level; }
void Node::set_weighted_level(int _weighted_level) {
    weighted_level = _weighted_level;
}
void Node::set_start_time(int _start_time) {
    start_time = _start_time;
    end_time = start_time + duration;
}
void Node::set_part(int _part) { part = _part; }
void Node::set_type(NODE_TYPES _type) { type = _type; }

bool Node::operator==(const Node &node) const {
    return name == node.get_name();
}
