#include <iostream>
#include "graph.h"
#include <chrono> 
#include <fstream> 

using namespace std::chrono;

void print_levels(Graph *g){
  cout<<"start printing the graph levels:\n";
  vector<string> graph_keys = g->get_keys();
  for (string nd : graph_keys) {
    cout << nd << "::" << g->get_node_by_name(nd)->get_top_level() << "::" << g->get_node_by_name(nd)->get_bottom_level() <<"::" <<
     g->get_node_by_name(nd)->get_weighted_level() << "\n";
  }
  cout<<"done printing the graph levels\n";
}
int main(){
  Graph g = Graph::create_and_annotate_graph();
  vector<string> graph_keys = g.get_keys();

 /*  print_levels(g);
  
  for (string nd : graph_keys) {
    cout << nd<<" "<<(*g.get_node_by_name(nd)).get_duration()<<" "<<(*g.get_node_by_name(nd)).get_type()<<"::";
     for (string adj: g[nd]){
       cout<<adj<<" "<<g.get_edge_weight(nd, adj)<<" ";
     }
     cout<<"\n";
  }  */
  
  auto start = high_resolution_clock::now(); 
  g.obtain_linear_clusters();  
  g.obtain_mapped_clusters();
  auto stop = high_resolution_clock::now(); 
  auto duration = duration_cast<microseconds>(stop - start);
  cout << (double)duration.count()/1000000 << endl;

  cout<<g.get_makespan()<<" is the makespan before collocation\n";
  //g.handle_memory();
  g.place_cpu_nodes();
  if(BALANCE_MEMORY){
    g.handle_memory();
  }
  g.collocate();
  cout<<g.get_makespan()<<" is the makespan after collocation\n";

 /*  int summ = 0, summ2 = 0;
  for(string node_name : g.get_keys()){
    Node *node = g.get_node_by_name(node_name);
    if(node_name != g.snk_node_name && node->get_type() != Node::REF_OP && 
      node->get_type() != Node::VAR && node->get_type() != Node::NO_OP){
      bool _long = true;
      for(string adj_name : g[node_name]){
        Node *adj_node = g.get_node_by_name(adj_name);
        if(adj_node->get_top_level() - (node->get_top_level() + node->get_duration() + g.get_edge_weight(node_name, adj_name)) < 10000 ){
          _long = false;
          break;
        }
      }
      if(_long){
        summ += g.get_edge_weight(node_name, g[node_name][0]);
        node->set_part(-1);
      }
      else{
        summ2 += g.get_edge_weight(node_name, g[node_name][0]);
      }
    }
  }
  cout<<summ<<", "<<summ2<<" is sum \n"; */

  int counts[K]={0};
  for(string node : g.get_keys()){
    counts[g.get_node_by_name(node)->get_part()]++;
  }
  cout<<"after collocation\n";
  for(int i = 0; i < K; i++){
    cout<<counts[i]<<"\n";
  }

  /* int cnt = 0;
  for(auto &it : g.levels_nodes){
    for(Node *nd : it.second){
      string str1 = "gradients_1/addn";
      string str2 = "gradients/addn";
      if (nd->get_name().find(str1) != std::string::npos || nd->get_name().find(str2) != std::string::npos) {
        nd->set_part(cnt++ % K);
      }
    }
  } */
  ofstream out_file;
  out_file.open ("/home/nahmad/placement.place");
  for(string node : g.get_keys()){
    if(g.get_node_by_name(node)->get_type() != Node::NO_OP){
      out_file<<node<<" "<<g.get_node_by_name(node)->get_part()<<"\n";
    }
  }
	out_file.close();
}