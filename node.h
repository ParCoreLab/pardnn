#include <string>
#include "utils.h"

using namespace std;

#ifndef NODE_H
#define NODE_H

class Node{
  public:
  enum NODE_TYPES {VAR, REF_OP, NORM_OP, NO_OP, CONT_DEP, PSEUDO};
  Node();
  Node(string);
  Node(string, int, long long, int, int, int, int,int, int, int, int, NODE_TYPES);
  string get_name() const;
  int get_duration() const;
  long long get_memory() const;
  int get_level() const;
  int get_reversed_level() const;
  int get_top_level() const;
  int get_bottom_level() const;
  int get_weighted_level() const;
  int get_start_time() const;
  int get_end_time() const;
  int get_part() const;
  NODE_TYPES get_type() const;
  void set_duration(int);
  void set_memory(long long);
  void set_level(int);
  void set_reversed_level(int);
  void set_top_level(int);
  void set_bottom_level(int);
  void set_weighted_level(int);
  void set_start_time(int);
  void set_part(int);
  void set_type(NODE_TYPES);
  bool operator==(const Node&) const;
  void print_node();

  private:
  int duration, level, reversed_level, start_time, end_time, part, top_level, bottom_level, weighted_level;
  long long memory;
  NODE_TYPES type;
  string name;
};

/* class NodeHashFunction { 
public:
    size_t operator()(const Node& node) const{ 
      hash<string> str_hash;
      return str_hash(node.get_name()); 
    } 
}; */

#endif