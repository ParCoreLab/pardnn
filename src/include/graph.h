#include "linear_cluster.h"
#include "node.h"
#include "utils.h"
#include <map>
#include <queue>
#include <unordered_map>

using namespace std;

class Graph {
    private:
        Graph();
        unordered_map<string, vector<string>> adj_list;
        unordered_map<string, vector<string>> rev_adj_list;
        vector<string> keys;
        unordered_map<string, Node> nodes;
        map<int, map<int, int>> levels_weights;
        map<int, int> levels_weights_as_sums;
        map<int, int> levels_comms_as_sums;
        map<int, int> levels_indices;
        vector<int> ordered_levels;
        unordered_map<string, unordered_map<string, int>> edges;
        unordered_map<string, int> nodes_in_degrees;
        unordered_map<string, int> nodes_out_degrees;
        unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>>
            clusters_comms;
        unordered_map<LinearCluster *, int> clusters_comms_as_totals;
        unordered_map<LinearCluster *, int> clusters_comms_as_totals_with_primaries;
        unordered_map<LinearCluster *, pair<int, int>> clusters_spans;
        void fill_levels_nodes();

    public:
        long double DoP;
        long double CCR;
        long long int total_work = 0;
        long long int total_comm = 0;
        map<int, vector<Node *>> levels_nodes;
        vector<LinearCluster> primary_clusters, secondary_clusters;
        static Graph create_and_annotate_graph();
        string src_node_name = "src";
        string snk_node_name = "snk";
        vector<string> operator[](string &);
        const vector<string> &get_keys() const;
        void add_edge(string &, string &);
        Node *get_node_by_name(string &);
        void read_graph(string &);
        void read_weights(string &);
        void read_memories(string &);
        void read_edges_weights(string &);
        void read_var_nodes(string &);
        void read_ref_nodes(string &);
        void read_no_op_nodes(string &);
        void print_graph();
        void print_adjacents(const string &);
        int get_edge_weight(const string &, const string &);
        int get_max_rev_adj_level(const string &);
        void top_sort();
        void top_sort_reversed();
        void calc_nodes_top_levels(bool, bool);
        void calc_nodes_bottom_levels(bool, bool);
        void calc_nodes_weighted_levels(bool, bool);
        void calc_nodes_in_degrees();
        void calc_nodes_out_degrees();
        void calc_nodes_tmp_top_levels(const unordered_map<string, bool> &,
                unordered_map<string, int> &, vector<string> &,
                unordered_map<string, int>, bool);
        string calc_nodes_tmp_bottom_levels(const unordered_map<string, bool> &,
                unordered_map<string, int> &,
                vector<string> &,
                unordered_map<string, int>);
        string calc_nodes_tmp_weighted_level(const unordered_map<string, bool> &,
                unordered_map<string, int> &,
                vector<string> &, vector<string> &,
                unordered_map<string, int> &,
                unordered_map<string, int> &);
        void obtain_linear_clusters();
        void obtain_mapped_clusters();
        void obtain_clusters_comms(
                unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>> &,
                unordered_map<LinearCluster *, int> &,
                unordered_map<LinearCluster *, int> &, vector<LinearCluster> &, bool);
        void long_term_comms(map<int, long long int> &);
        void obtain_clusters_spans(unordered_map<LinearCluster *, pair<int, int>> &,
                vector<LinearCluster> &);
        void merge_clusters(
                LinearCluster *, LinearCluster *, bool, int *, int *,
                unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>> &,
                unordered_map<LinearCluster *, int> &,
                unordered_map<LinearCluster *, int> &);
        LinearCluster *max_communicating(
                LinearCluster *,
                unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>> &);
        int least_load_primary_in_span(int *[], pair<int, int>, int &);
        int most_load_primary_in_span(int *[], pair<int, int>);
        int unmapped_work_in_span(int *, int *[], pair<int, int>);
        int max_communicating_primary(
                LinearCluster *,
                unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>> &);
        void collocate();
        void place_cpu_nodes();
        int get_makespan();
        Node *get_earliest_snk_child(LinearCluster *);
        Node *get_latest_src_parent(LinearCluster *);
        map<int, int> comm_to_primary(Node *, LinearCluster *);
        void handle_memory();
        void assign_distant_vars_to_cpu(int);
        void assign_heavy_memory_consumers_to_cpu();
};
