#include <string>
#include "node.h"

using namespace std;

class LinearCluster{
  private:
  Node *src_node, *snk_node;
  int src_level, snk_level;
  bool primary;
  int weight;
  pair<int, int> span;
  public:
  vector<Node*> nodes;
  LinearCluster();
  Node* operator[](int);
  void add_node(Node*);
  Node* get_src_node(); 
  Node* get_snk_node(); 
  int length();
  bool is_primary();
  int get_weight();
  void set_weight(int);
  pair<int, int> get_span();
  void set_span(pair<int, int>);
  void print_cluster();
  int get_indx(string&);
  bool operator < (LinearCluster&);
  double sorting_criteria_mirror();

  static bool start_end_sorting_key_mirror(LinearCluster&, LinearCluster&);
  static bool sorting_criteria_density(LinearCluster &, LinearCluster &);
  static bool sorting_criteria_weighted_level(LinearCluster &, LinearCluster &);
};