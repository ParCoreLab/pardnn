#include "include/graph.h"
#include <bits/stdc++.h>
#include <chrono>
#include <math.h>
#include <queue>
#include <utility>

using namespace std;
using namespace std::chrono;
int mul = 1;
Graph Graph::create_and_annotate_graph() {
    Graph g;
    g.read_graph(folder_files.files[GRAPH_FILE]);
    g.read_weights(folder_files.files[NODES_WEIGHTS_FILE]);
    g.read_var_nodes(folder_files.files[VAR_NODES_FILE]);
    g.read_ref_nodes(folder_files.files[REF_NODES_FILE]);
    g.read_no_op_nodes(folder_files.files[NO_OPS_FILE]);
    g.read_edges_weights(folder_files.files[EDGES_COSTS_FILE]);
    if (BALANCE_MEMORY) {
        g.read_memories(folder_files.files[MEMORY_FILE_NAME]);
    }

    g.calc_nodes_in_degrees();
    g.calc_nodes_out_degrees();

    g.top_sort();
    g.top_sort_reversed();

    return g;
}

bool has_suffix(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
        str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

Graph::Graph() {}

vector<string> Graph::operator[](string &node_name) {
    if (adj_list.find(node_name) == adj_list.end()) {
        cout << node_name << " is Not found";
        return vector<string>();
    }
    return adj_list[node_name];
}

void Graph::add_edge(string &src_name, string &dst_name) {
    if (adj_list.find(src_name) == adj_list.end()) {
        adj_list[src_name] = vector<string>();
        keys.push_back(src_name);
        nodes[src_name] = Node(src_name);
    }
    if (adj_list.find(dst_name) == adj_list.end()) {
        keys.push_back(dst_name);
        nodes[dst_name] = Node(dst_name);
        adj_list[dst_name] = vector<string>();
    }

    if (rev_adj_list.find(dst_name) == rev_adj_list.end()) {
        rev_adj_list[dst_name] = vector<string>();
    }
    if (rev_adj_list.find(src_name) == rev_adj_list.end()) {
        rev_adj_list[src_name] = vector<string>();
    }

    adj_list[src_name].push_back(dst_name);
    rev_adj_list[dst_name].push_back(src_name);

    if (edges.find(src_name) == edges.end()) {
        edges[src_name] = unordered_map<string, int>();
    }
    if (src_name == src_node_name || dst_name == snk_node_name)
        edges[src_name][dst_name] = 0;
    else
        edges[src_name][dst_name] = (int)COMM_LATENCY;
}

Node *Graph::get_node_by_name(string &node_name) {
    Node *result = NULL;
    if (nodes.find(node_name) != nodes.end()) {
        result = &nodes[node_name];
    }
    return result;
}

void Graph::read_graph(string &file_name) {
    ifstream graph_file;
    string line;
    graph_file.open(file_name);

    if (!graph_file.good()) {
        throw GRAPH_FILE_NOT_FOUND;
    }

    while (getline(graph_file, line)) {
        clean_line(line);
        vector<string> splits = split(line, "->");
        if (splits.size() == 2) {
            add_edge(splits[0], splits[1]);
        }
    }
}

void Graph::read_weights(string &file_name) {
    ifstream weights_file;
    string line;
    weights_file.open(file_name);

    if (!weights_file.good()) {
        throw 3;
    }

    while (getline(weights_file, line)) {
        clean_line(line);
        vector<string> splits = split(line, "::");
        if (splits.size() == 2) {
            Node *nd = get_node_by_name(splits[0]);
            if (nd != NULL) {
                (*nd).set_duration(stoi(splits[1]));
                total_work += stoi(splits[1]);
            }
        }
    }
}

void Graph::read_memories(string &file_name) {
    ifstream memories_file;
    string line;
    memories_file.open(file_name);
    while (getline(memories_file, line)) {
        clean_line(line);
        vector<string> splits = split(line, "::");
        if (splits.size() == 2) {
            Node *nd = get_node_by_name(splits[0]);
            if (nd != NULL) {
                (*nd).set_memory(stoi(splits[1]));
            }
        }
    }
}

void Graph::read_edges_weights(string &file_name) {
    ifstream weights_file;
    string line;
    weights_file.open(file_name);

    if (!weights_file.good()) {
        throw EDGES_WEIGHTS_FILE_NOT_FOUND;
    }

    while (getline(weights_file, line)) {
        clean_line(line);
        vector<string> splits = split(line, "::");
        if (splits.size() == 2) {
            string node_name = splits[0];
            Node *nd = get_node_by_name(node_name);
            if (nd != NULL and nd->get_type() != Node::NO_OP) {
                for (auto &it : edges[node_name]) {
                    string adj_name = it.first;
                    if (Node::NO_OP != nodes[adj_name].get_type() &&
                            adj_name != snk_node_name) {
                        edges[node_name][adj_name] =
                            (int)((ceil(stod(splits[1]) * COMM_TRANSFER_RATE_RECIPROCAL)) +
                                    COMM_LATENCY);
                        total_comm += edges[node_name][adj_name];
                    } else {
                        edges[node_name][adj_name] = 0;
                    }
                }
            }
        }
    }
}

void Graph::read_var_nodes(string &file_name) {
    ifstream nodes_file;
    string line;
    nodes_file.open(file_name);

    if (!nodes_file.good()) {
        throw NODES_FILE_NOT_FOUND;
    }

    while (getline(nodes_file, line)) {
        clean_line(line);
        Node *nd = get_node_by_name(line);
        if (nd != NULL) {
            (*nd).set_type(Node::VAR);
        }
    }
}

void Graph::read_ref_nodes(string &file_name) {
    ifstream nodes_file;
    string line;
    nodes_file.open(file_name);
    
    if (!nodes_file.good()) {
        throw REF_NODES_FILE_NOT_FOUND;
    }

    while (getline(nodes_file, line)) {
        clean_line(line);
        Node *nd = get_node_by_name(line);
        if (nd != NULL) {
            (*nd).set_type(Node::REF_OP);
        }
    }
}

void Graph::read_no_op_nodes(string &file_name) {
    ifstream nodes_file;
    string line;
    nodes_file.open(file_name);

    if (!nodes_file.good()) {
        throw NO_OP_NODES_FILE_NOT_FOUND;
    }

    while (getline(nodes_file, line)) {
        clean_line(line);
        Node *nd = get_node_by_name(line);
        if (nd != NULL) {
            (*nd).set_type(Node::NO_OP);
        }
        for (string dst_node : adj_list[line]) {
            edges[line][dst_node] = COMM_LATENCY;
        }
        for (string src_node : rev_adj_list[line]) {
            edges[src_node][line] = COMM_LATENCY;
        }
    }
}

int Graph::get_edge_weight(const string &src, const string &dst) {
    return edges[src][dst];
}

const vector<string> &Graph::get_keys() const { return keys; }

void Graph::print_graph() {
    for (auto &it : adj_list) {
        cout << it.first << "->";
        print_adjacents(it.first);
    }
}

void Graph::print_adjacents(const string &node_name) {
    cout << "[ ";
    for (string adj : adj_list[node_name]) {
        cout << adj << " ";
    }
    cout << "]\n";
}

int Graph::get_max_rev_adj_level(const string &node_name) {
    int max_level = 0;
    for (string rev_adj : rev_adj_list[node_name]) {
        int current_level = get_node_by_name(rev_adj)->get_level();
        if (current_level > max_level) {
            max_level = current_level;
        }
    }
    return max_level;
}

void Graph::top_sort() {
    // This function assumes the graph is DAG
    queue<string> to_visit;
    get_node_by_name(src_node_name)->set_level(0);
    to_visit.push(src_node_name);
    unordered_map<string, int> tmp_nodes_in_degrees;
    copy_unordered_map(nodes_in_degrees, tmp_nodes_in_degrees);

    while (!to_visit.empty()) {
        string current_node = to_visit.front();
        int current_node_level = get_node_by_name(current_node)->get_level();
        to_visit.pop();
        for (string adj_node : adj_list[current_node]) {
            if (--tmp_nodes_in_degrees[adj_node] == 0) {
                to_visit.push(adj_node);
            }
            get_node_by_name(adj_node)->set_level(
                    max(current_node_level + 1, get_node_by_name(adj_node)->get_level()));
        }
    }
}

void Graph::top_sort_reversed() {
    // Both this and the top_sort() can be implemented with one function, however,
    // I see splitting better as it reduces branching.
    queue<string> to_visit;
    get_node_by_name(snk_node_name)->set_reversed_level(0);
    to_visit.push(snk_node_name);
    unordered_map<string, int> tmp_nodes_out_degrees;
    copy_unordered_map(nodes_out_degrees, tmp_nodes_out_degrees);

    while (!to_visit.empty()) {
        string current_node = to_visit.front();
        int current_node_level =
            get_node_by_name(current_node)->get_reversed_level();
        to_visit.pop();
        for (string adj_node : rev_adj_list[current_node]) {
            if (--tmp_nodes_out_degrees[adj_node] == 0) {
                to_visit.push(adj_node);
            }
            get_node_by_name(adj_node)->set_reversed_level(
                    max(current_node_level + 1,
                        get_node_by_name(adj_node)->get_reversed_level()));
        }
    }
}

void Graph::fill_levels_nodes() {
    levels_weights.clear();
    levels_nodes.clear();
    levels_weights_as_sums.clear();
    levels_comms_as_sums.clear();
    levels_indices.clear();
    ordered_levels.clear();
    for (int i = 0; i < K; i++) {
        levels_weights[i] = map<int, int>();
    }
    for (string node_name : keys) {
        Node *node = get_node_by_name(node_name);
        int node_level = node->get_top_level();
        if (levels_nodes.find(node_level) == levels_nodes.end()) {
            levels_nodes[node_level] = vector<Node *>();
            levels_weights_as_sums[node_level] = 0;
            for (int i = 0; i < K; i++) {
                levels_weights[i][node_level] = 0;
            }
        }
        levels_nodes[node_level].push_back(node);
        levels_weights_as_sums[node_level] += node->get_duration();
        levels_weights[node->get_part()][node_level] += node->get_duration();
    }

    int i = 0;
    for (auto &it : levels_weights_as_sums) {
        levels_indices[it.first] = i++;
        ordered_levels.push_back(it.first);
    }
    for (string node_name : keys) {
        Node *node = get_node_by_name(node_name);
        int node_level = node->get_top_level();
        if (levels_comms_as_sums.find(levels_indices[node_level]) ==
                levels_comms_as_sums.end()) {
            levels_comms_as_sums[levels_indices[node_level]] = 0;
        }
        int visited_parts[K] = {0};
        for (string adj_node_name : adj_list[node_name]) {
            Node *adj_node = get_node_by_name(adj_node_name);
            if (adj_node->get_part() < K && node->get_part() < K &&
                    visited_parts[adj_node->get_part()] == 0 &&
                    adj_node->get_part() != node->get_part()) {
                levels_comms_as_sums[levels_indices[node_level]] +=
                    edges[node_name][adj_node_name];
                visited_parts[adj_node->get_part()] = 1;
            }
        }
    }
}

void Graph::calc_nodes_top_levels(bool is_clustered, bool with_weights) {
    queue<string> to_visit;
    get_node_by_name(src_node_name)->set_top_level(0);
    to_visit.push(src_node_name);
    unordered_map<string, int> tmp_nodes_in_degrees;
    copy_unordered_map(nodes_in_degrees, tmp_nodes_in_degrees);

    while (!to_visit.empty()) {
        string current_node_name = to_visit.front();
        Node *current_node = get_node_by_name(current_node_name);
        int current_node_level =
            get_node_by_name(current_node_name)->get_top_level();
        to_visit.pop();
        for (string adj_node_name : adj_list[current_node_name]) {
            if (--tmp_nodes_in_degrees[adj_node_name] == 0) {
                to_visit.push(adj_node_name);
            }
            Node *adj_node = get_node_by_name(adj_node_name);
            int edge_weight =
                ((current_node->get_part() == -1 ||
                  current_node->get_part() != adj_node->get_part() || !is_clustered)
                 ? get_edge_weight(current_node_name, adj_node_name)
                 : 1);
            adj_node->set_top_level(max(
                        current_node->get_top_level() +
                        (with_weights ? current_node->get_duration() : 1) + edge_weight,
                        adj_node->get_top_level()));
        }
    }
}

void Graph::calc_nodes_bottom_levels(bool is_clustered, bool with_weights) {
    queue<string> to_visit;
    get_node_by_name(snk_node_name)->set_bottom_level(1);
    to_visit.push(snk_node_name);
    unordered_map<string, int> tmp_nodes_out_degrees;
    copy_unordered_map(nodes_out_degrees, tmp_nodes_out_degrees);

    while (!to_visit.empty()) {
        string current_node_name = to_visit.front();
        Node *current_node = get_node_by_name(current_node_name);
        int current_node_level =
            get_node_by_name(current_node_name)->get_bottom_level();
        to_visit.pop();
        for (string adj_node_name : rev_adj_list[current_node_name]) {
            if (--tmp_nodes_out_degrees[adj_node_name] == 0) {
                to_visit.push(adj_node_name);
            }
            Node *adj_node = get_node_by_name(adj_node_name);
            int edge_weight =
                ((current_node->get_part() == -1 ||
                  current_node->get_part() != adj_node->get_part() || !is_clustered)
                 ? get_edge_weight(adj_node_name, current_node_name)
                 : 1);
            adj_node->set_bottom_level(max(current_node->get_bottom_level() +
                        (adj_node->get_duration()) +
                        edge_weight,
                        adj_node->get_bottom_level()));
        }
    }
}

void Graph::calc_nodes_weighted_levels(bool clustered, bool with_weights) {
    for (string node_name : keys) {
        Node *node = get_node_by_name(node_name);
        node->set_bottom_level(node->get_duration());
        node->set_top_level(0);
    }
    calc_nodes_top_levels(clustered, with_weights);
    calc_nodes_bottom_levels(clustered, with_weights);
    for (string node_name : keys) {
        Node *node = get_node_by_name(node_name);
        node->set_weighted_level(node->get_top_level() + node->get_bottom_level());
    }
}

void Graph::calc_nodes_in_degrees() {
    for (string node_name : keys) {
        nodes_in_degrees[node_name] = rev_adj_list[node_name].size();
        // if (nodes_in_degrees[node_name] > 100)
        // cout << node_name << "\t" << nodes_in_degrees[node_name] << "\n";
    }
}

void Graph::calc_nodes_out_degrees() {
    for (string node_name : keys) {
        nodes_out_degrees[node_name] = adj_list[node_name].size();
    }
}

void Graph::calc_nodes_tmp_top_levels(
        const unordered_map<string, bool> &visited,
        unordered_map<string, int> &tmp_top_levels, vector<string> &free_nodes,
        unordered_map<string, int> tmp_nodes_in_degrees, bool clustered = false) {
    for (string free_node_name : free_nodes) {
        if (visited.find(free_node_name) != visited.end()) {
            continue;
        }
        tmp_top_levels[free_node_name] = 0;

        queue<string> to_visit;
        for (string node_name : adj_list[free_node_name]) {
            to_visit.push(node_name);
        }
        while (!to_visit.empty()) {
            string current_node_name = to_visit.front();
            Node *current_node = get_node_by_name(current_node_name);
            to_visit.pop();
            for (string adj_node : adj_list[current_node_name]) {
                if (visited.find(adj_node) == visited.end()) {
                    if (--tmp_nodes_in_degrees[adj_node] == 0) {
                        to_visit.push(adj_node);
                    }
                    int edge_weight =
                        (!clustered || current_node->get_part() !=
                         get_node_by_name(adj_node)->get_part())
                        ? get_edge_weight(current_node_name, adj_node)
                        : 1;
                    tmp_top_levels[adj_node] =
                        max(tmp_top_levels[current_node_name] +
                                current_node->get_duration() + edge_weight,
                                tmp_top_levels[adj_node]);
                }
            }
        }
    }
}

string Graph::calc_nodes_tmp_bottom_levels(
        const unordered_map<string, bool> &visited,
        unordered_map<string, int> &tmp_bottom_levels, vector<string> &free_nodes,
        unordered_map<string, int> tmp_nodes_out_degrees) {
    string max_weighted_level_node = free_nodes[0];
    int max_weighted_level = tmp_bottom_levels[max_weighted_level_node];
    for (string free_node_name : free_nodes) {
        if (visited.find(free_node_name) != visited.end()) {
            continue;
        }
        queue<string> to_visit;
        to_visit.push(free_node_name);
        while (!to_visit.empty()) {
            string current_node_name = to_visit.front();
            Node *current_node = get_node_by_name(current_node_name);
            to_visit.pop();
            for (string adj_node : rev_adj_list[current_node_name]) {
                if (visited.find(adj_node) == visited.end()) {
                    tmp_bottom_levels[adj_node] =
                        max(tmp_bottom_levels[current_node_name] +
                                get_node_by_name(adj_node)->get_duration() +
                                get_edge_weight(adj_node, current_node_name),
                                tmp_bottom_levels[adj_node]);

                    if (--tmp_nodes_out_degrees[adj_node] == 0) {
                        to_visit.push(adj_node);
                        if (visited.find(adj_node) == visited.end() &&
                                tmp_bottom_levels[adj_node] > max_weighted_level) {
                            max_weighted_level = tmp_bottom_levels[adj_node];
                            max_weighted_level_node = adj_node;
                        }
                    }
                }
            }
        }
    }
    return max_weighted_level_node;
}

string Graph::calc_nodes_tmp_weighted_level(
        const unordered_map<string, bool> &visited,
        unordered_map<string, int> &tmp_weighted_levels, vector<string> &free_nodes,
        vector<string> &rev_free_nodes,
        unordered_map<string, int> &tmp_nodes_in_degrees,
        unordered_map<string, int> &tmp_nodes_out_degrees) {
    unordered_map<string, int> tmp_top_levels;
    unordered_map<string, int> tmp_bottom_levels;
    calc_nodes_tmp_top_levels(visited, tmp_top_levels, free_nodes,
            tmp_nodes_in_degrees);
    calc_nodes_tmp_bottom_levels(visited, tmp_bottom_levels, rev_free_nodes,
            tmp_nodes_out_degrees);
    int max_weighted_level = -1;
    string max_weighted_level_node = "";
    for (string node : keys) {
        tmp_weighted_levels[node] = tmp_top_levels[node] + tmp_bottom_levels[node];
        if (visited.find(node) == visited.end() &&
                tmp_weighted_levels[node] > max_weighted_level) {
            max_weighted_level = tmp_weighted_levels[node];
            max_weighted_level_node = node;
        }
    }

    return max_weighted_level_node;
}

// Comparator function to sort pairs
// according to second value
bool cmp(pair<string, int> &a, pair<string, int> &b) {
    return a.second < b.second;
}

void Graph::obtain_linear_clusters() {
    vector<string> rev_free_nodes;
    rev_free_nodes.push_back(snk_node_name);
    unordered_map<string, bool> visited;
    unordered_map<string, int> tmp_bottom_levels;
    unordered_map<string, int> tmp_weighted_levels;
    unordered_map<string, int> tmp_nodes_in_degrees;
    unordered_map<string, int> tmp_nodes_out_degrees;
    int part = 0;

    copy_unordered_map(nodes_in_degrees, tmp_nodes_in_degrees);
    copy_unordered_map(nodes_out_degrees, tmp_nodes_out_degrees);
    calc_nodes_bottom_levels(false, true);

    for (string node_name : keys) {
        tmp_bottom_levels[node_name] =
            get_node_by_name(node_name)->get_bottom_level();
    }

    priority_queue<pair<double, string>> free_nodes_pq;
    free_nodes_pq.push(make_pair(
                get_node_by_name(src_node_name)->get_bottom_level(), src_node_name));

    LinearCluster current_cluster = LinearCluster();
    string current_node = src_node_name;

    while (!free_nodes_pq.empty()) {
        current_cluster = LinearCluster();

        if (primary_clusters.size() == K ||
                visited.find(current_node) != visited.end()) {
            current_node = free_nodes_pq.top().second;
            while (!free_nodes_pq.empty() &&
                    visited.find(current_node) != visited.end()) {
                free_nodes_pq.pop();
                current_node = free_nodes_pq.top().second;
            }
        }

        if (part == K) {
            calc_nodes_weighted_levels(true, true);
            for (string node_name : keys) {
                if (visited.find(node_name) == visited.end()) {
                    free_nodes_pq.push(
                            make_pair(get_node_by_name(node_name)->get_weighted_level() +
                                get_node_by_name(node_name)->get_reversed_level() /
                                (double)(get_node_by_name(src_node_name)
                                    ->get_reversed_level()),
                                node_name));
                }
                tmp_bottom_levels[node_name] =
                    get_node_by_name(node_name)->get_bottom_level();
            }
        }

        while (visited.find(current_node) == visited.end()) {
            visited[current_node] = true;
            if (current_node == "" || tmp_bottom_levels[current_node] == 0) {
                break;
            }
            get_node_by_name(current_node)->set_part(part);
            current_cluster.add_node(get_node_by_name(current_node));
            // for(string adj : adj_list[current_node])
            //{if(--tmp_nodes_in_degrees[adj] == 0){free_nodes_pq.push(make_pair(
            //tmp_weighted_levels[adj], adj) );}}
            for (string adj : rev_adj_list[current_node]) {
                if (--tmp_nodes_out_degrees[adj] == 0) {
                    rev_free_nodes.push_back(adj);
                }
            }

            int max_level = 0;
            string next_node = "";
            for (string adj : adj_list[current_node]) {
                if ((part >= K || visited.find(adj) == visited.end()) &&
                        tmp_bottom_levels[adj] > max_level) {
                    max_level = tmp_bottom_levels[adj];
                    next_node = adj;
                }
            }

            current_node = next_node;
        }

        if (current_cluster.length() > 0) {
            if (part < K) {
                for (string node_name : keys) {
                    tmp_bottom_levels[node_name] =
                        get_node_by_name(node_name)->get_duration();
                }
                current_node = calc_nodes_tmp_bottom_levels(
                        visited, tmp_bottom_levels, rev_free_nodes, tmp_nodes_out_degrees);
                if (part == 0) {
                    DoP = ((double)total_work) / current_cluster.get_weight();
                    CCR = ((double)total_comm) / total_work;
                    cout << "DoP = " << DoP << "\n";
                    cout << "CCR = " << CCR << "\n";
                    cout << "Total work = " << (double)total_work / 1000000.0 << "\n";
                    cout << endl;
                }
                primary_clusters.push_back(current_cluster);
            } else {
                secondary_clusters.push_back(current_cluster);
            }
            part++;
        }
    }
}

void Graph::obtain_mapped_clusters() {
    mul = 100;
    calc_nodes_weighted_levels(true, true);

    std::sort(secondary_clusters.begin(), secondary_clusters.end(),
            LinearCluster::sorting_criteria_weighted_level);
    vector<pair<int, int>> clusters_weights;
    for (int i = 0; i < secondary_clusters.size(); i++) {
        LinearCluster *lc_ptr = &secondary_clusters[i];
        clusters_weights.push_back(make_pair(lc_ptr->get_weight(), i));
    }
    // std::sort(clusters_weights.rbegin(), clusters_weights.rend());

    int median_path_weighted_level =
        secondary_clusters[3 * secondary_clusters.size() / 4]
        .get_src_node()
        ->get_weighted_level();
    int max_weighted_level =
        secondary_clusters[0].get_src_node()->get_weighted_level();
    int median_cluster_weight =
        clusters_weights[clusters_weights.size() / 4].first;
    int max_cluster_weight = clusters_weights[0].first;
    for (int i = 0; i < secondary_clusters.size(); i++) {
        LinearCluster *lc_ptr = &secondary_clusters[i];
        for (Node *node : lc_ptr->nodes) {
            node->set_part(K + i);
        }
    }

    fill_levels_nodes();

    int *BITree =
        constructBITree(levels_weights_as_sums, levels_weights_as_sums.size());
    int *BITree_comms =
        constructBITree(levels_comms_as_sums, levels_comms_as_sums.size());
    int *primary_trees[K];
    for (int i = 0; i < K; i++) {
        primary_trees[i] =
            constructBITree(levels_weights[i], levels_weights[i].size());
    }

    unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>>
        clusters_comms;
    unordered_map<LinearCluster *, int> clusters_comms_as_totals;
    unordered_map<LinearCluster *, int> clusters_comms_as_totals_with_primaries;
    static unordered_map<LinearCluster *, pair<int, int>> clusters_spans;

    obtain_clusters_comms(clusters_comms, clusters_comms_as_totals,
            clusters_comms_as_totals_with_primaries,
            secondary_clusters, true);
    obtain_clusters_spans(clusters_spans, secondary_clusters);

    unordered_map<LinearCluster *, LinearCluster *> merged_clusters;
    unordered_map<LinearCluster *, LinearCluster *> initially_merged_clusters;
    string str = "generator_2/e3d-lstm_12/e3d3/hidden_1/batchnorm/mul_1";

    int cntts[5] = {0};
    int cntt = 1;
    int weighted_level_threshold = median_path_weighted_level;
    int weight_threshold = max_cluster_weight;
    int weight_threshold_step = (max_cluster_weight - median_cluster_weight) / 2;
    int i = 0, ii = 0;

    map<LinearCluster *, string> unmerged_clusters;

    int counts[K] = {0};
    for (string node : keys) {
        if (get_node_by_name(node)->get_part() < K)
            counts[get_node_by_name(node)->get_part()]++;
    }
    // cout << "before mapping\n";
    // for (int i = 0; i < K; i++)
    // {
    //   cout << counts[i] << "\n";
    // }

    for (int itrr = 0; itrr < 2; itrr++) {
        cntt = 1;
        while (cntt > 0) {
            cntt = 0;
            for (int main_step = 5; main_step <= 10; main_step++) {
                vector<pair<int, int>> clusters_weights_map;
                i = 0;
                while (i < main_step * secondary_clusters.size() / 10) {
                    LinearCluster *lc_ptr = &secondary_clusters[i];
                    if (lc_ptr->get_src_node()->get_level() > 5 || itrr == 1) {
                        clusters_weights_map.push_back(make_pair(
                                    clusters_spans[lc_ptr].second - clusters_spans[lc_ptr].first,
                                    i));
                    }
                    i++;
                }
                std::sort(clusters_weights_map.begin(), clusters_weights_map.end());
                // cout << clusters_weights_map.size() << " is mapped_clusters size\n";
                i = 0;
                while (i < clusters_weights_map.size()) {
                    LinearCluster *lc_ptr =
                        &secondary_clusters[clusters_weights_map[i].second];
                    i++;
                    /* if(lc_ptr->get_indx(str) != -1)
                       {
                       for(int ik = 0 ; ik < K; ik++){
                       cout<<clusters_comms[lc_ptr][&primary_clusters[ik]]<<"\t";
                       }
                       lc_ptr->print_cluster();
                       } */
                    if (merged_clusters.find(lc_ptr) != merged_clusters.end()) {
                        continue;
                    }
                    int primary_indx = max_communicating_primary(lc_ptr, clusters_comms);
                    LinearCluster *max_lc_prt = &primary_clusters[primary_indx];
                    int total_primaries_comm =
                        clusters_comms_as_totals_with_primaries[lc_ptr];
                    if (primary_indx != -1 &&
                            clusters_comms_as_totals[lc_ptr] <=
                            clusters_comms[lc_ptr][max_lc_prt] &&
                            clusters_comms[lc_ptr][max_lc_prt] > COMM_LATENCY) {
                        pair<int, int> targeted_span =
                            make_pair(levels_indices[clusters_spans[lc_ptr].first],
                                    levels_indices[clusters_spans[lc_ptr].second]);
                        int least_loaded_lc;
                        int least_load = least_load_primary_in_span(
                                primary_trees, targeted_span, least_loaded_lc);
                        // cout<<least_loaded_lc<<"\n";
                        least_loaded_lc = primary_indx;
                        int most_load =
                            most_load_primary_in_span(primary_trees, targeted_span);
                        int unmapped_work =
                            unmapped_work_in_span(BITree, primary_trees, targeted_span);
                        int all_work_in_the_span = getSum(BITree, targeted_span.second) -
                            getSum(BITree, targeted_span.first);
                        int comm_in_the_span =
                            0; // getSum(BITree_comms, targeted_span.second) -
                               // getSum(BITree_comms, targeted_span.first);
                        int the_work_of_the_target_primary_if_merged =
                            lc_ptr->get_weight() +
                            getSum(primary_trees[primary_indx], targeted_span.second) -
                            getSum(primary_trees[primary_indx], targeted_span.first);
                        int the_work_of_the_target_primary =
                            getSum(primary_trees[primary_indx], targeted_span.second) -
                            getSum(primary_trees[primary_indx], targeted_span.first);

                        int latest_parent_level =
                            get_latest_src_parent(lc_ptr)->get_top_level();
                        int avg_delay =
                            getSum(
                                    primary_trees[least_loaded_lc],
                                    levels_indices[lc_ptr->get_src_node()->get_top_level()]) -
                            getSum(primary_trees[least_loaded_lc],
                                    levels_indices[latest_parent_level]);

                        if (all_work_in_the_span / K >=
                                the_work_of_the_target_primary_if_merged ||
                                max(most_load, least_load + lc_ptr->get_weight() +
                                    clusters_comms[lc_ptr][max_lc_prt] +
                                    avg_delay + unmapped_work / K) >=
                                the_work_of_the_target_primary_if_merged ||
                                the_work_of_the_target_primary_if_merged <=
                                lc_ptr->get_weight() + clusters_comms[lc_ptr][max_lc_prt] ||
                                clusters_comms[lc_ptr][max_lc_prt] >= most_load - least_load) {
                            vector<LinearCluster *> merge_destination_ptr;
                            merge_destination_ptr.push_back(max_lc_prt);
                            merge_clusters(lc_ptr, merge_destination_ptr[0], true,
                                    primary_trees[primary_indx], BITree_comms,
                                    clusters_comms, clusters_comms_as_totals,
                                    clusters_comms_as_totals_with_primaries);
                            merged_clusters[lc_ptr] = merge_destination_ptr[0];
                            cntt++;
                            cntts[0]++;
                            cntts[1]++;
                            if (unmerged_clusters.find(lc_ptr) != unmerged_clusters.end()) {
                                unmerged_clusters[lc_ptr] = "";
                            }
                        } else if (lc_ptr->get_indx(str) != -1) {
                            string str2 = "generator_1/e3d-lstm_12/e3d3/conv3d/biasadd";
                            string str3 = "generator_1/e3d-lstm_12/e3d3/conv3d/biasadd";
                            cout << least_load << "\t" << most_load << "\t" << unmapped_work
                                << "\t" << all_work_in_the_span << "\t"
                                << clusters_comms_as_totals[lc_ptr] << "\t"
                                << clusters_spans[lc_ptr].first << "\t"
                                << clusters_spans[lc_ptr].second << "\t"
                                << the_work_of_the_target_primary_if_merged << "\t"
                                << get_node_by_name(str2)->get_top_level() << "\t"
                                << get_node_by_name(str3)->get_top_level() << "\n";
                            // lc_ptr->print_cluster();
                            cout << "\n-----------------1--------------------\n";
                            for (int ik = 0; ik < K; ik++) {
                                int current_sum =
                                    getSum(primary_trees[ik], targeted_span.second) -
                                    getSum(primary_trees[ik], targeted_span.first);
                                cout << current_sum << "\t";
                            }
                            cout << "\n------------------1-------------------\n";
                        }
                    }
                }

                i = 0;
                if (K == REPLICATION_FACTOR || ceil(CCR) >= max(10, 2 * (int)DoP)) {
                    while (i < clusters_weights_map.size()) {
                        LinearCluster *lc_ptr =
                            &secondary_clusters[clusters_weights_map[i].second];
                        i++;
                        if (merged_clusters.find(lc_ptr) != merged_clusters.end()) {
                            continue;
                        }

                        int primary_indx =
                            max_communicating_primary(lc_ptr, clusters_comms);
                        LinearCluster *max_lc_prt = &primary_clusters[primary_indx];

                        int total_primaries_comm =
                            clusters_comms_as_totals_with_primaries[lc_ptr];
                        if (total_primaries_comm <= clusters_comms[lc_ptr][max_lc_prt] &&
                                clusters_comms[lc_ptr][max_lc_prt] > COMM_LATENCY) {
                            if (primary_indx != -1) {
                                pair<int, int> targeted_span =
                                    make_pair(levels_indices[clusters_spans[lc_ptr].first],
                                            levels_indices[clusters_spans[lc_ptr].second]);
                                int least_loaded_lc;
                                int least_load = least_load_primary_in_span(
                                        primary_trees, targeted_span, least_loaded_lc);
                                // cout<<least_loaded_lc<<"\n";
                                least_loaded_lc = primary_indx;
                                int most_load =
                                    most_load_primary_in_span(primary_trees, targeted_span);
                                int unmapped_work =
                                    unmapped_work_in_span(BITree, primary_trees, targeted_span);
                                int all_work_in_the_span =
                                    getSum(BITree, targeted_span.second) -
                                    getSum(BITree, targeted_span.first);
                                int comm_in_the_span =
                                    0; // getSum(BITree_comms, targeted_span.second) -
                                       // getSum(BITree_comms, targeted_span.first);
                                int the_work_of_the_target_primary_if_merged =
                                    lc_ptr->get_weight() +
                                    getSum(primary_trees[primary_indx], targeted_span.second) -
                                    getSum(primary_trees[primary_indx], targeted_span.first);
                                int the_work_of_the_target_primary =
                                    getSum(primary_trees[primary_indx], targeted_span.second) -
                                    getSum(primary_trees[primary_indx], targeted_span.first);

                                int latest_parent_level =
                                    get_latest_src_parent(lc_ptr)->get_top_level();
                                int avg_delay = getSum(primary_trees[least_loaded_lc],
                                        levels_indices[lc_ptr->get_src_node()
                                        ->get_top_level()]) -
                                    getSum(primary_trees[least_loaded_lc],
                                            levels_indices[latest_parent_level]);

                                if (all_work_in_the_span / K >=
                                        the_work_of_the_target_primary_if_merged ||
                                        max(most_load, least_load + lc_ptr->get_weight() +
                                            clusters_comms[lc_ptr][max_lc_prt]) +
                                        avg_delay + unmapped_work / K >=
                                        the_work_of_the_target_primary_if_merged ||
                                        the_work_of_the_target_primary_if_merged <=
                                        lc_ptr->get_weight() +
                                        clusters_comms[lc_ptr][max_lc_prt]) {
                                    vector<LinearCluster *> merge_destination_ptr;
                                    merge_destination_ptr.push_back(max_lc_prt);
                                    merge_clusters(lc_ptr, merge_destination_ptr[0], true,
                                            primary_trees[primary_indx], BITree_comms,
                                            clusters_comms, clusters_comms_as_totals,
                                            clusters_comms_as_totals_with_primaries);
                                    merged_clusters[lc_ptr] = merge_destination_ptr[0];
                                    cntt++;
                                    cntts[0]++;
                                    cntts[2]++;
                                    if (unmerged_clusters.find(lc_ptr) !=
                                            unmerged_clusters.end()) {
                                        unmerged_clusters[lc_ptr] = "";
                                    }
                                } else if (itrr == 0 && lc_ptr->get_indx(str) != -1) {
                                    string str2 = "generator_2/e3d-lstm_10/e3d2/inputs_1/moments/"
                                        "stopgradient";
                                    string str3 = "generator_3/e3d-lstm_10/e3d2/inputs_1/moments/"
                                        "stopgradient";
                                    cout << least_load << "\t" << most_load << "\t"
                                        << unmapped_work << "\t" << all_work_in_the_span << "\t"
                                        << clusters_comms_as_totals[lc_ptr] << "\t"
                                        << clusters_spans[lc_ptr].first << "\t"
                                        << clusters_spans[lc_ptr].second << "\t"
                                        << the_work_of_the_target_primary_if_merged << "\t"
                                        << get_node_by_name(str2)->get_top_level() << "\t"
                                        << get_node_by_name(str3)->get_top_level() << "\n";
                                    // lc_ptr->print_cluster();
                                    cout << "\n-----------------2--------------------\n";
                                    for (int ik = 0; ik < K; ik++) {
                                        int current_sum =
                                            getSum(primary_trees[ik], targeted_span.second) -
                                            getSum(primary_trees[ik], targeted_span.first);
                                        cout << current_sum << "\t";
                                    }
                                    cout << "\n------------------2-------------------\n";
                                }
                            }
                        }
                    }
                }
            }
            if (cntt > 0 || itrr == 0) {
                // calc_nodes_weighted_levels(true, true);

                /* std::sort(secondary_clusters.begin(), secondary_clusters.end(),
                   LinearCluster::sorting_criteria_weighted_level); for (int i = 0; i <
                   secondary_clusters.size(); i++)
                   {
                   LinearCluster *lc_ptr = &secondary_clusters[i];
                   for (Node *node : lc_ptr->nodes)
                   {
                   node->set_part(K + i);
                   }
                   } */

                /* fill_levels_nodes();

                   BITree = constructBITree(levels_weights_as_sums,
                   levels_weights_as_sums.size()); BITree_comms =
                   constructBITree(levels_comms_as_sums, levels_comms_as_sums.size()); for
                   (int i = 0; i < K; i++)
                   {
                   primary_trees[i] = constructBITree(levels_weights[i],
                   levels_weights[i].size());
                   }*/

                if (itrr == 1 &&
                        (REPLICATION_FACTOR == K || ceil(CCR) >= max(10, 2 * (int)DoP))) {
                    obtain_clusters_comms(clusters_comms, clusters_comms_as_totals,
                            clusters_comms_as_totals_with_primaries,
                            secondary_clusters, false);
                } else {
                    obtain_clusters_comms(clusters_comms, clusters_comms_as_totals,
                            clusters_comms_as_totals_with_primaries,
                            secondary_clusters, true);
                }
                // obtain_clusters_spans(clusters_spans, secondary_clusters);
            } else {
                //________________________________________
                if (K == REPLICATION_FACTOR || ceil(CCR) >= max(10, 2 * (int)DoP)) {
                    for (int main_step = 5; main_step <= 10; main_step++) {
                        vector<pair<int, int>> clusters_weights_map;
                        i = 0;
                        while (i < main_step * secondary_clusters.size() / 10) {
                            LinearCluster *lc_ptr = &secondary_clusters[i];
                            if (lc_ptr->get_src_node()->get_level() > 5 || main_step == 9) {
                                clusters_weights_map.push_back(
                                        make_pair(get_earliest_snk_child(lc_ptr)->get_level() -
                                            lc_ptr->get_src_node()->get_level(),
                                            i));
                            }
                            i++;
                        }
                        // cout << clusters_weights_map.size() << " is mapped_clusters
                        // size\n";
                        for (int step = 1; step <= 20; step++) {
                            cntt = 1;
                            while (cntt > 0) {
                                cntt = 0;
                                i = 0;
                                while (i < clusters_weights_map.size()) {
                                    LinearCluster *lc_ptr =
                                        &secondary_clusters[clusters_weights_map[i].second];
                                    i++;
                                    if (merged_clusters.find(lc_ptr) != merged_clusters.end()) {
                                        continue;
                                    }
                                    int primary_indx =
                                        max_communicating_primary(lc_ptr, clusters_comms);
                                    LinearCluster *max_lc_prt = &primary_clusters[primary_indx];
                                    bool prnt = false;
                                    int total_primaries_comm =
                                        clusters_comms_as_totals_with_primaries[lc_ptr];
                                    if (primary_indx != -1 &&
                                            clusters_comms[lc_ptr][max_lc_prt] > COMM_LATENCY &&
                                            clusters_comms[lc_ptr][max_lc_prt] >
                                            max(total_primaries_comm / 2,
                                                clusters_comms_as_totals[lc_ptr] / K)) {
                                        pair<int, int> targeted_span = make_pair(
                                                levels_indices[clusters_spans[lc_ptr].first],
                                                levels_indices[clusters_spans[lc_ptr].second]);
                                        int least_loaded_lc;
                                        int least_load = least_load_primary_in_span(
                                                primary_trees, targeted_span, least_loaded_lc);
                                        // cout<<least_loaded_lc<<"\n";
                                        least_loaded_lc = primary_indx;
                                        int most_load =
                                            most_load_primary_in_span(primary_trees, targeted_span);
                                        int unmapped_work = unmapped_work_in_span(
                                                BITree, primary_trees, targeted_span);
                                        int all_work_in_the_span =
                                            getSum(BITree, targeted_span.second) -
                                            getSum(BITree, targeted_span.first);
                                        int the_work_of_the_target_primary_if_merged =
                                            lc_ptr->get_weight() +
                                            getSum(primary_trees[primary_indx],
                                                    targeted_span.second) -
                                            getSum(primary_trees[primary_indx],
                                                    targeted_span.first);
                                        int the_work_of_the_target_primary =
                                            getSum(primary_trees[primary_indx],
                                                    targeted_span.second) -
                                            getSum(primary_trees[primary_indx],
                                                    targeted_span.first);
                                        int comm_in_the_span =
                                            0; // getSum(BITree_comms, targeted_span.second) -
                                               // getSum(BITree_comms, targeted_span.first);

                                        int latest_parent_level =
                                            get_latest_src_parent(lc_ptr)->get_top_level();
                                        int avg_delay =
                                            getSum(primary_trees[least_loaded_lc],
                                                    levels_indices[lc_ptr->get_src_node()
                                                    ->get_top_level()]) -
                                            getSum(primary_trees[least_loaded_lc],
                                                    levels_indices[latest_parent_level]);

                                        if (all_work_in_the_span / K >=
                                                the_work_of_the_target_primary_if_merged ||
                                                min(most_load, least_load + lc_ptr->get_weight() +
                                                    clusters_comms[lc_ptr][max_lc_prt] -
                                                    total_primaries_comm / K +
                                                    avg_delay) >=
                                                the_work_of_the_target_primary_if_merged ||
                                                comm_in_the_span + clusters_comms[lc_ptr][max_lc_prt] >=
                                                max(most_load,
                                                    the_work_of_the_target_primary_if_merged)) {
                                            vector<LinearCluster *> merge_destination_ptr;
                                            merge_destination_ptr.push_back(max_lc_prt);
                                            merge_clusters(lc_ptr, merge_destination_ptr[0], true,
                                                    primary_trees[primary_indx], BITree_comms,
                                                    clusters_comms, clusters_comms_as_totals,
                                                    clusters_comms_as_totals_with_primaries);
                                            merged_clusters[lc_ptr] = merge_destination_ptr[0];
                                            cntt++;
                                            cntts[0]++;
                                            cntts[3]++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } // end else
        }   // end while
            // cout << secondary_clusters.size() << "\t" << cntts[0] << "\t" << cntts[1]
            // << "\t" << cntts[2] << "\t" << cntts[3] << "\t" << cntts[4] << " are
            // cntts \n";
    } // end main for
      //____________________________________________________________________________
    cntt = 1;
    i = 0;
    if (REPLICATION_FACTOR == K || ceil(CCR) >= max(10, 2 * (int)DoP)) {
        while (ii < clusters_weights.size()) {
            LinearCluster *lc_ptr = &secondary_clusters[ii];
            ii++;
            if (merged_clusters.find(lc_ptr) != merged_clusters.end()) {
                continue;
            }
            vector<pair<double, int>> primaries_coms_map;
            pair<int, int> targeted_span =
                make_pair(levels_indices[clusters_spans[lc_ptr].first],
                        levels_indices[clusters_spans[lc_ptr].second]);
            int most_load = most_load_primary_in_span(primary_trees, targeted_span);
            int least_loaded_lc;
            int least_load = least_load_primary_in_span(primary_trees, targeted_span,
                    least_loaded_lc);
            int all_work_in_the_span = getSum(BITree, targeted_span.second) -
                getSum(BITree, targeted_span.first);
            for (int i = 0; i < K; i++) {
                LinearCluster *primary_ptr = &primary_clusters[i];
                primaries_coms_map.push_back(make_pair(
                            clusters_comms[lc_ptr][primary_ptr] +
                            1.0 / (double)(getSum(primary_trees[i], targeted_span.second) -
                                getSum(primary_trees[i], targeted_span.first) +
                                1.0),
                            i));
            }
            primaries_coms_map.push_back(
                    make_pair(-1, primaries_coms_map[K - 1].first));
            std::sort(primaries_coms_map.rbegin(), primaries_coms_map.rend());
            int max_comm =
                clusters_comms[lc_ptr]
                [&primary_clusters[primaries_coms_map[0].second]];
            int comm_in_the_span = 0; // getSum(BITree_comms, targeted_span.second) -
                                      // getSum(BITree_comms, targeted_span.first);
            int j = 0;
            while (clusters_comms[lc_ptr]
                    [&primary_clusters[primaries_coms_map[j].second]] >
                    COMM_LATENCY) {
                int the_work_of_the_target_primary_if_merged =
                    lc_ptr->get_weight() +
                    getSum(primary_trees[i], targeted_span.second) -
                    getSum(primary_trees[i], targeted_span.first);
                int i = primaries_coms_map[j].second;
                j++;
                int curret_comm = clusters_comms[lc_ptr][&primary_clusters[i]];
                if (min(most_load - comm_in_the_span,
                            least_load + lc_ptr->get_weight() + curret_comm +
                            (curret_comm - max_comm)) >=
                        the_work_of_the_target_primary_if_merged ||
                        the_work_of_the_target_primary_if_merged + comm_in_the_span <=
                        most_load ||
                        the_work_of_the_target_primary_if_merged <=
                        lc_ptr->get_weight() + curret_comm ||
                        comm_in_the_span + curret_comm >=
                        max(most_load, the_work_of_the_target_primary_if_merged)) {
                    vector<LinearCluster *> merge_destination_ptr;
                    merge_destination_ptr.push_back(&primary_clusters[i]);
                    merge_clusters(lc_ptr, merge_destination_ptr[0], true,
                            primary_trees[i], BITree_comms, clusters_comms,
                            clusters_comms_as_totals,
                            clusters_comms_as_totals_with_primaries);
                    merged_clusters[lc_ptr] = merge_destination_ptr[0];
                    cntt++;
                    cntts[0]++;
                    cntts[4]++;
                    break;
                }
            }
        }
    }
    // cout << secondary_clusters.size() << "\t" << cntts[0] << "\t" << cntts[1]
    // << "\t" << cntts[2] << "\t" << cntts[3] << "\t" << cntts[4] << " are cntts
    // \n";
    //________________________________________

    obtain_clusters_comms(clusters_comms, clusters_comms_as_totals,
            clusters_comms_as_totals_with_primaries,
            secondary_clusters, false);
    ii = 0;
    while (ii <
            clusters_weights.size()) // && clusters_weights[ii].first >=
                                     // weight_threshold - weight_threshold_step)
    {
        LinearCluster *lc_ptr = &secondary_clusters[clusters_weights[ii].second];
        ii++;
        if (merged_clusters.find(lc_ptr) != merged_clusters.end()) {
            continue;
        }
        int min_cost =
            getSum(primary_trees[0],
                    levels_indices[clusters_spans[lc_ptr].second]) -
            getSum(primary_trees[0], levels_indices[clusters_spans[lc_ptr].first]) +
            -clusters_comms[lc_ptr][&primary_clusters[0]];
        int min_indx = 0;
        for (int j = 1; j < primary_clusters.size(); j++) {
            int current_cost = getSum(primary_trees[j],
                    levels_indices[clusters_spans[lc_ptr].second]) -
                getSum(primary_trees[j],
                        levels_indices[clusters_spans[lc_ptr].first]) +
                -clusters_comms[lc_ptr][&primary_clusters[j]];
            if (current_cost < min_cost) {
                min_cost = current_cost;
                min_indx = j;
            }
        }
        vector<LinearCluster *> merge_destination_ptr;
        merge_destination_ptr.push_back(&primary_clusters[min_indx]);
        merge_clusters(lc_ptr, merge_destination_ptr[0], true,
                primary_trees[min_indx], BITree_comms, clusters_comms,
                clusters_comms_as_totals,
                clusters_comms_as_totals_with_primaries);
        merged_clusters[lc_ptr] = merge_destination_ptr[0];
    }
    counts[K] = {0};
    for (string node : keys) {
        if (get_node_by_name(node)->get_part() < K)
            counts[get_node_by_name(node)->get_part()]++;
    }
    // cout << "mapping is done\n";
    // for (int i = 0; i < K; i++)
    // {
    //   cout << counts[i] << "\n";
    // }
}

void Graph::merge_clusters(
        LinearCluster *src, LinearCluster *dst, bool merge_work, int *BITree,
        int *BITree_comms,
        unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>>
        &clusters_comms,
        unordered_map<LinearCluster *, int> &clusters_comms_as_totals,
        unordered_map<LinearCluster *, int>
        &clusters_comms_as_totals_with_primaries) {
    int dst_part = dst->nodes[0]->get_part();
    int old_length = dst->length();
    int new_weight = dst->get_weight();

    for (Node *node : src->nodes) {
        dst->add_node(node);
        string node_name = node->get_name();
        int tmp_comm = 0;
        node->set_part(dst_part);
        new_weight += node->get_duration();
    }
    for (Node *node : src->nodes) {
        if (merge_work) {
            updateBIT(BITree, levels_weights_as_sums.size(),
                    levels_indices[node->get_top_level()], node->get_duration());
        }

        /* string node_name = node->get_name();
           map<LinearCluster *, int> visited_parts;
           for(string adj_name: adj_list[node_name]){
           Node *adj_node = get_node_by_name(adj_name);
           int adj_node_part = adj_node->get_part();
           LinearCluster *adj_lc = adj_node_part < K ?
           &primary_clusters[adj_node_part] : &secondary_clusters[adj_node_part - K];
           if (visited_parts.find(adj_lc) == visited_parts.end())
           {
           visited_parts[adj_lc] = 0;
           }
           visited_parts[adj_lc] = max(visited_parts[adj_lc],
           edges[node_name][adj_name]);
           }
           for(string rev_adj_name: rev_adj_list[node_name]){
           Node *rev_adj_node = get_node_by_name(rev_adj_name);
           int adj_node_part = rev_adj_node->get_part();
           LinearCluster *adj_lc = adj_node_part < K ?
           &primary_clusters[adj_node_part] : &secondary_clusters[adj_node_part - K];
           if (visited_parts.find(adj_lc) == visited_parts.end())
           {
           visited_parts[adj_lc] = 0;
           }
           visited_parts[adj_lc] = max(visited_parts[adj_lc],
           edges[node_name][rev_adj_name]);
           }
           for(auto &it : visited_parts){
           clusters_comms[it.first][dst] += it.second;
           clusters_comms[dst][it.first] += it.second;
           clusters_comms[it.first][src] = 0;
           clusters_comms_as_totals_with_primaries[it.first] += it.second;
           } */
    }
    dst->set_weight(new_weight);

    if (dst->nodes[old_length]->get_top_level() <
            dst->nodes[0]->get_top_level()) {
        Node *tmp = dst->nodes[0];
        dst->nodes[0] = dst->nodes[old_length];
        dst->nodes[old_length] = tmp;
    }
    if (dst->nodes[old_length - 1]->get_top_level() >
            dst->nodes[dst->length() - 1]->get_top_level()) {
        Node *tmp = dst->nodes[dst->length() - 1];
        dst->nodes[dst->length() - 1] = dst->nodes[old_length - 1];
        dst->nodes[old_length - 1] = tmp;
    }
}

LinearCluster *Graph::max_communicating(
        LinearCluster *lc,
        unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>>
        &clusters_comms) {
    int max_comm = -1;
    LinearCluster *max_communicating;
    for (auto &it : clusters_comms[lc]) {
        if (it.second > max_comm) {
            max_communicating = it.first;
            max_comm = it.second;
        }
    }
    return max_communicating;
}

void Graph::obtain_clusters_comms(
        unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>>
        &clusters_comms,
        unordered_map<LinearCluster *, int> &clusters_comms_as_totals,
        unordered_map<LinearCluster *, int>
        &clusters_comms_as_totals_with_primaries,
        vector<LinearCluster> &target_clusters, bool is_critical) {
    clusters_comms.clear();
    clusters_comms_as_totals.clear();
    clusters_comms_as_totals_with_primaries.clear();
    map<Node *, map<int, int>> nodes_primaries_comms;
    for (string node_name : keys) {
        Node *node = get_node_by_name(node_name);
        if (nodes_primaries_comms.find(node) == nodes_primaries_comms.end()) {
            nodes_primaries_comms[node] = map<int, int>();
            for (int i = 0; i < K; i++) {
                nodes_primaries_comms[node][i] = 0;
            }
        }
        /* for (string adj_node_name : adj_list[node_name])
           {
           int adj_part = get_node_by_name(adj_node_name)->get_part();
           if (adj_part < K)
           {
           nodes_primaries_comms[node][adj_part] =
           max(nodes_primaries_comms[node][adj_part], edges[node_name][adj_node_name]);
           }
           } */
    }

    for (int i = 0; i < target_clusters.size(); i++) {
        LinearCluster *lc_ptr = &target_clusters[i];
        if (lc_ptr->get_src_node()->get_part() < K) {
            continue; // already merged cluster
        }
        clusters_comms_as_totals[lc_ptr] = 0;
        clusters_comms_as_totals_with_primaries[lc_ptr] = 0;
        clusters_comms[lc_ptr] = unordered_map<LinearCluster *, int>();
        for (int k = 0; k < K; k++) {
            clusters_comms[lc_ptr][&primary_clusters[k]] = 0;
            clusters_comms[&primary_clusters[k]][lc_ptr] = 0;
        }
    }

    for (int i = 0; i < target_clusters.size(); i++) {
        LinearCluster *lc_ptr = &target_clusters[i];
        for (Node *node : lc_ptr->nodes) {
            if (is_critical && node->get_level() < 5) {
                continue;
            }
            string node_name = node->get_name();
            map<LinearCluster *, int> visited_parts;
            for (string adj_name : adj_list[node_name]) {
                Node *adj = get_node_by_name(adj_name);
                int adj_part = adj->get_part();
                if (adj_part != node->get_part()) {
                    LinearCluster *target_cluster =
                        adj_part < K ? &primary_clusters[adj_part]
                        : (&secondary_clusters[adj_part - K]);

                    if (clusters_comms[lc_ptr].find(target_cluster) ==
                            clusters_comms[lc_ptr].end()) {
                        clusters_comms[lc_ptr][target_cluster] = 0;
                    }
                    if (clusters_comms[target_cluster].find(lc_ptr) ==
                            clusters_comms[target_cluster].end()) {
                        clusters_comms[target_cluster][lc_ptr] = 0;
                    }

                    int comm = edges[node_name][adj_name];
                    if (visited_parts.find(target_cluster) == visited_parts.end()) {
                        visited_parts[target_cluster] = 0;
                    }
                    visited_parts[target_cluster] =
                        max(visited_parts[target_cluster], comm);
                }
            }

            for (auto &it : visited_parts) {
                clusters_comms[lc_ptr][it.first] += it.second;
                clusters_comms[it.first][lc_ptr] += it.second;
                clusters_comms_as_totals[lc_ptr] += it.second;
                clusters_comms_as_totals[it.first] += it.second;
                for (int k = 0; k < K; k++) {
                    if (&primary_clusters[k] == it.first) {
                        clusters_comms_as_totals_with_primaries[lc_ptr] += it.second;
                    }
                }
            }
        }
    }

    for (int i = 0; i < primary_clusters.size(); i++) {
        LinearCluster *lc_ptr = &primary_clusters[i];
        for (Node *node : lc_ptr->nodes) {
            if (is_critical && node->get_level() < 5)
                continue;
            string node_name = node->get_name();
            map<LinearCluster *, int> visited_parts;
            for (string adj_name : adj_list[node_name]) {
                Node *adj = get_node_by_name(adj_name);
                int adj_part = adj->get_part();
                if (adj_part >= K) {
                    LinearCluster *target_cluster = &secondary_clusters[adj_part - K];
                    int comm = edges[node_name][adj_name];
                    if (visited_parts.find(target_cluster) == visited_parts.end()) {
                        visited_parts[target_cluster] = 0;
                    }
                    visited_parts[target_cluster] =
                        max(visited_parts[target_cluster], comm);
                }
            }

            for (auto &it : visited_parts) {
                clusters_comms[lc_ptr][it.first] += it.second;
                clusters_comms[it.first][lc_ptr] += it.second;
                clusters_comms_as_totals[lc_ptr] += it.second;
                clusters_comms_as_totals[it.first] += it.second;
                clusters_comms_as_totals_with_primaries[it.first] += it.second;
            }
        }
    }
}

void Graph::long_term_comms(map<int, long long int> &long_term_comms_map) {
    long long int net_comm_to_add = 0l;
    map<int, long long int> subtract_at_level;
    for (auto it : levels_nodes) {
        int level = it.first;
        long_term_comms_map[level] = 0l;
        if (subtract_at_level.find(level) == subtract_at_level.end()) {
            subtract_at_level[level] = 0l;
        }
        vector<Node *> nodes = it.second;
        for (Node *node : nodes) {
            map<int, bool> visited_parts;
            string node_name = node->get_name();
            for (string adj_node_name : adj_list[node_name]) {
                Node *adj_node = get_node_by_name(adj_node_name);
                int adj_node_part = adj_node->get_part();
                if (node->get_part() == adj_node_part || visited_parts[adj_node_part]) {
                    continue;
                }
                visited_parts[adj_node_part] = true;
                net_comm_to_add += edges[node_name][adj_node_name];
                int adj_level = adj_node->get_top_level();
                if (subtract_at_level.find(adj_level) == subtract_at_level.end()) {
                    subtract_at_level[adj_level] = 0l;
                }
                subtract_at_level[adj_level] += edges[node_name][adj_node_name];
            }
        }
        net_comm_to_add -= subtract_at_level[level];
        long_term_comms_map[level] = max((long long int)0l, net_comm_to_add);
        net_comm_to_add -= levels_weights_as_sums[level];
    }
}

void Graph::obtain_clusters_spans(
        unordered_map<LinearCluster *, pair<int, int>> &clusters_spans,
        vector<LinearCluster> &target_clusters) {
    for (int i = 0; i < target_clusters.size(); i++) {
        LinearCluster *lc_ptr = &target_clusters[i];
        int latest_parent_level = lc_ptr->get_src_node()->get_top_level(),
            earliest_child_level = get_node_by_name(snk_node_name)->get_top_level();

        for (string adj_name : adj_list[lc_ptr->get_snk_node()->get_name()]) {
            int adj_level = get_node_by_name(adj_name)->get_top_level();
            if (adj_level < earliest_child_level) {
                earliest_child_level = adj_level;
            }
        }

        for (string adj_name : rev_adj_list[lc_ptr->get_src_node()->get_name()]) {
            if (adj_name == src_node_name) {
                continue;
            }
            int rev_adj_earliest_possible_child =
                get_node_by_name(adj_name)->get_top_level() +
                get_node_by_name(adj_name)->get_duration();
            int start_indx =
                levels_indices[get_node_by_name(adj_name)->get_top_level()];
            bool found = false;
            while (ordered_levels[start_indx] <= earliest_child_level) {
                if (ordered_levels[start_indx] >= rev_adj_earliest_possible_child) {
                    rev_adj_earliest_possible_child = ordered_levels[start_indx];
                    break;
                }
                start_indx++;
            }
            if (levels_indices.find(rev_adj_earliest_possible_child) !=
                    levels_indices.end() &&
                    rev_adj_earliest_possible_child < latest_parent_level) {
                latest_parent_level = rev_adj_earliest_possible_child;
            }
        }
        lc_ptr->set_span(make_pair(latest_parent_level, earliest_child_level));
        clusters_spans[lc_ptr] =
            make_pair(latest_parent_level, earliest_child_level);
    }
}

void Graph::collocate() {
    std::map<string, bool> changed;
    int iii = 0;
    /* for (string node_name : keys)
       {
       if (get_node_by_name(node_name)->get_part() == -1 ||
       (get_node_by_name(node_name)->get_type() != Node::VAR &&
       !has_suffix(node_name, "/read")) )
       {
       continue;
       }
       get_node_by_name(node_name)->set_part(max(1, iii%K));
       } */
    for (string node_name : keys) {
        if (get_node_by_name(node_name)->get_type() != Node::VAR ||
                changed.find(node_name) != changed.end()) {
            continue;
        }

        int parts_freqs[K] = {0};
        int max_freq = 0;
        int node_part = get_node_by_name(node_name)->get_part();
        bool assigned_to_cpu = node_part == -1;
        /* if(!assigned_to_cpu){
           for (string adj_node_name : adj_list[node_name])
           {
           int adj_part;
           if (get_node_by_name(adj_node_name)->get_type() == Node::REF_OP)
           {
           adj_part = get_node_by_name(adj_node_name)->get_part();
           parts_freqs[adj_part] +=
           get_node_by_name(adj_node_name)->get_weighted_level(); if
           (parts_freqs[adj_part] > max_freq)
           {
           max_freq = parts_freqs[adj_part];
           node_part = adj_part;
           }
           }
           }
           } */
        string current_node_name;
        queue<string> to_visit;
        get_node_by_name(node_name)->set_part(node_part);
        to_visit.push(node_name);
        while (!to_visit.empty()) {
            current_node_name = to_visit.front();
            changed[current_node_name] = true;
            to_visit.pop();
            Node *node = get_node_by_name(current_node_name);
            for (string adj : adj_list[current_node_name]) {
                Node *adj_node = get_node_by_name(adj);
                if (adj_node->get_type() == Node::REF_OP) {
                    adj_node->set_part(node_part);
                    for (string rev_adj : rev_adj_list[adj]) {
                        Node *rev_adj_node = get_node_by_name(rev_adj);
                        if (changed.find(rev_adj) == changed.end() &&
                                rev_adj_node->get_type() == Node::VAR) {
                            rev_adj_node->set_part(node_part);
                            to_visit.push(rev_adj);
                            changed[rev_adj] = true;
                        }
                    }
                }
            }
        }
    }
    /* for (string node_name : keys)
       {
       if (get_node_by_name(node_name)->get_type() != Node::VAR)
       {
       continue;
       }
       cout<<get_node_by_name(node_name)->get_part()<<"\n";
       } */
}

int Graph::most_load_primary_in_span(int *primary_trees[],
        pair<int, int> span) {
    int max_load = 0;
    int lc_indx;
    for (int i = 0; i < K; i++) {
        int current_sum = getSum(primary_trees[i], span.second) -
            getSum(primary_trees[i], span.first);
        if (current_sum > max_load) {
            max_load = current_sum;
            lc_indx = i;
        }
    }
    return max_load;
}

int Graph::least_load_primary_in_span(int *primary_trees[], pair<int, int> span,
        int &lc) {
    int min_load = getSum(primary_trees[0], span.second) -
        getSum(primary_trees[0], span.first);
    int lc_indx = 0;
    for (int i = 1; i < K; i++) {
        int current_sum = getSum(primary_trees[i], span.second) -
            getSum(primary_trees[i], span.first);
        if (current_sum < min_load) {
            min_load = current_sum;
            lc_indx = i;
        }
    }
    lc = lc_indx;
    return min_load;
}

int Graph::unmapped_work_in_span(int *BITree, int *primary_trees[],
        pair<int, int> span) {
    int mapped_sum = 0;
    for (int i = 0; i < K; i++) {
        mapped_sum += getSum(primary_trees[i], span.second) -
            getSum(primary_trees[i], span.first);
    }
    return getSum(BITree, span.second) - getSum(BITree, span.first) - mapped_sum;
}

void Graph::place_cpu_nodes() {
    ifstream my_file;
    string line;
    my_file.open(folder_files.files[VANILLA_PLACEMENT_FILE]);
    while (getline(my_file, line)) {
        clean_line_keep_spaces(line);
        vector<string> splits = split(line, " ");
        if (splits.size() == 2 && splits[1] == "-1" && get_node_by_name(splits[0])
                //&& ( (get_node_by_name(splits[0])->get_level() < 500 ||
                //get_node_by_name(splits[0])->get_level() > 2000) || splits[1] == "-1")
           ) {
            get_node_by_name(splits[0])->set_part(stoi(splits[1]));
        }
    }
}

int Graph::max_communicating_primary(
        LinearCluster *lc_ptr,
        unordered_map<LinearCluster *, unordered_map<LinearCluster *, int>>
        &clusters_comms) {
    int primary_indx = -1;
    int max_comm = 0;
    for (int j = 0; j < K; j++) {
        if (clusters_comms[lc_ptr][&primary_clusters[j]] > max_comm) {
            primary_indx = j;
            max_comm = clusters_comms[lc_ptr][&primary_clusters[j]];
        }
    }
    return primary_indx;
}

int Graph::get_makespan() {
    // double test_ratio = COMM_TRANSFER_RATE_RECIPROCAL / 130.0;
    int parts_ready_times[K] = {0};
    queue<string> to_visit;
    get_node_by_name(src_node_name)->set_top_level(0);
    to_visit.push(src_node_name);
    unordered_map<string, int> tmp_nodes_in_degrees;
    copy_unordered_map(nodes_in_degrees, tmp_nodes_in_degrees);
    for (string node_name : keys) {
        Node *node = get_node_by_name(node_name);
        node->set_top_level(0);
    }
    while (!to_visit.empty()) {
        string current_node_name = to_visit.front();
        Node *current_node = get_node_by_name(current_node_name);
        to_visit.pop();
        for (string adj_node_name : adj_list[current_node_name]) {
            Node *adj_node = get_node_by_name(adj_node_name);
            int edge_weight = current_node->get_part() != adj_node->get_part()
                ? get_edge_weight(current_node_name, adj_node_name)
                : 1;
            adj_node->set_top_level(
                    max(parts_ready_times[current_node->get_part()] + edge_weight,
                        parts_ready_times[adj_node->get_part()]));
            if (--tmp_nodes_in_degrees[adj_node_name] == 0) {
                to_visit.push(adj_node_name);
                parts_ready_times[adj_node->get_part()] =
                    adj_node->get_top_level() + adj_node->get_duration();
            }
        }
    }
    return get_node_by_name(snk_node_name)->get_top_level();
}

Node *Graph::get_earliest_snk_child(LinearCluster *lc_ptr) {
    Node *result = get_node_by_name(snk_node_name);
    int min_level = result->get_level();
    Node *snk_node = lc_ptr->get_snk_node();
    for (string adj : adj_list[snk_node->get_name()]) {
        Node *nd = get_node_by_name(adj);
        if (nd->get_level() < min_level) {
            min_level = nd->get_level();
            result = nd;
        }
    }
    return result;
}

Node *Graph::get_latest_src_parent(LinearCluster *lc_ptr) {
    Node *result = get_node_by_name(src_node_name);
    int max_level = result->get_top_level();
    Node *src_node = lc_ptr->get_src_node();
    for (string rev_adj : rev_adj_list[src_node->get_name()]) {
        Node *nd = get_node_by_name(rev_adj);
        if (nd->get_top_level() > max_level) {
            max_level = nd->get_top_level();
            result = nd;
        }
    }
    return result;
}

map<int, int> Graph::comm_to_primary(Node *nd, LinearCluster *lc) {
    map<int, int> comms;
    comms[levels_indices[nd->get_top_level()]] = 0;
    for (string adj_node : adj_list[nd->get_name()]) {
        if (&primary_clusters[get_node_by_name(adj_node)->get_part()] == lc) {
            comms[levels_indices[nd->get_top_level()]] =
                max(comms[levels_indices[nd->get_top_level()]],
                        edges[nd->get_name()][adj_node]);
        }
    }
    for (string rev_adj_node : rev_adj_list[nd->get_name()]) {
        if (comms.find(
                    levels_indices[get_node_by_name(rev_adj_node)->get_level()]) ==
                comms.end()) {
            comms[levels_indices
                [levels_indices[get_node_by_name(rev_adj_node)->get_level()]]] =
                0;
        }
        if (&primary_clusters[get_node_by_name(rev_adj_node)->get_part()] == lc) {
            comms[levels_indices[levels_indices[get_node_by_name(rev_adj_node)
                ->get_level()]]] +=
                comms[levels_indices[levels_indices[get_node_by_name(rev_adj_node)
                ->get_level()]]];
        }
    }
    return comms;
}

void Graph::assign_distant_vars_to_cpu(int heavy_comm_last_level) {
    vector<pair<int, Node *>> vars_earliest_needed_levels;
    int count_assigned_to_cpu = 0;
    long long int mem_assigned_to_cpu = 0l;
    for (string node_name : keys) {
        if (get_node_by_name(node_name)->get_type() != Node::VAR) {
            continue;
        }
        int min_level_child = get_node_by_name(snk_node_name)->get_top_level();
        for (string adj_node_name : adj_list[node_name]) {
            Node *adj_node = get_node_by_name(adj_node_name);
            int adj_level = adj_node->get_top_level();
            if (has_suffix(adj_node_name, "read")) {
                for (string read_adj_node_name : adj_list[adj_node_name]) {
                    adj_node = get_node_by_name(read_adj_node_name);
                    adj_level = adj_node->get_top_level();
                    if (adj_level < min_level_child) {
                        min_level_child = adj_level;
                    }
                }
            } else {
                if (adj_level < min_level_child) {
                    min_level_child = adj_level;
                }
            }
        }
        vars_earliest_needed_levels.push_back(
                make_pair(min_level_child, get_node_by_name(node_name)));
    }
    std::sort(vars_earliest_needed_levels.begin(),
            vars_earliest_needed_levels.end());
    int total_comm_cost_host_devices = 0;
    for (int i = 0; i < vars_earliest_needed_levels.size(); i++) {
        Node *node = vars_earliest_needed_levels[i].second;
        int comm = edges[node->get_name()][adj_list[node->get_name()][0]];
        if (vars_earliest_needed_levels[i].first - heavy_comm_last_level >
                (total_comm_cost_host_devices + comm) / K) {
            total_comm_cost_host_devices += comm;
            node->set_part(-1);
            count_assigned_to_cpu++;
            mem_assigned_to_cpu += node->get_memory();
        }
    }
    cout << count_assigned_to_cpu << "\t are assigned to cpu with memory \t"
        << mem_assigned_to_cpu << "\n";
}

void Graph::assign_heavy_memory_consumers_to_cpu() {
    for (string node_name : keys) {
        long long int reserved_for = 0l;
        Node *node = get_node_by_name(node_name);
        int node_level = node->get_level();
        int earliest_child_level = node_level;
        for (string rev_adj_name : rev_adj_list[node_name]) {
            Node *rev_adj = get_node_by_name(rev_adj_name);
            reserved_for += rev_adj->get_memory();
            earliest_child_level = min(earliest_child_level, rev_adj->get_level());
        }
        if (node_level - earliest_child_level >=
                get_node_by_name(snk_node_name)->get_level() / 2 &&
                reserved_for >= (0.1 * DEVICE_MEM_CAPACITY)) {
            cout << node_name << " is a mem hungry\n";
            node->set_part(-1);
        }
    }
}

void Graph::handle_memory() {
    long double total_intermediate_memory = 0.0, total_var_memory = 0.0;
    long double total_intermediate_memories[K] = {0.0};
    long double total_var_memories[K] = {0.0};
    long double intermediate_memroy_ratios[K] = {0.0};
    long double var_memroy_ratios[K] = {0.0};
    vector<pair<double, int>> parts_ratios;
    map<int, priority_queue<pair<long long, Node *>>> parts_var_nodes;

    map<int, long long int> long_term_comms_map;
    long_term_comms(long_term_comms_map);
    int heavy_comm_last_level;
    for (auto it : long_term_comms_map) {
        if (it.second <= 0) {
            break;
            heavy_comm_last_level = it.first;
        }
    }
    for (string node_name : keys) {
        Node *node = get_node_by_name(node_name);
        int node_part = node->get_part();
        long long node_memory = node->get_memory();
        if (node->get_type() == Node::VAR) {
            total_var_memory += node_memory;
            total_var_memories[node_part] += node_memory;
            parts_var_nodes[node_part].push(make_pair(node->get_memory(), node));
        } else if (node->get_type() != Node::REF_OP) {
            total_intermediate_memory += node_memory;
            total_intermediate_memories[node_part] += node_memory;
        }
    }
    assign_heavy_memory_consumers_to_cpu();

    for (int i = 0; i < K; i++) {
        intermediate_memroy_ratios[i] =
            total_intermediate_memories[i] / total_intermediate_memory;
        var_memroy_ratios[i] = total_var_memories[i] / total_var_memory;
    }
    for (int i = 0; i < K; i++) {
        parts_ratios.push_back(make_pair(
                    K * (intermediate_memroy_ratios[i] + var_memroy_ratios[i]) / 2.0, i));
    }

    cout << "ratios before\t";
    for (int i = 0; i < K; i++) {
        cout << parts_ratios[i].first << ","
            << var_memroy_ratios[parts_ratios[i].second] << ","
            << intermediate_memroy_ratios[parts_ratios[i].second] << ","
            << parts_ratios[i].second << "\t";
    }
    cout << "\n";

    std::sort(parts_ratios.begin(), parts_ratios.end());
    int most_loaded_part = K - 1;
    // cout<<parts_var_nodes[0].size()<<" is it\n";
    // parts_var_nodes[0].pop();
    // cout<<parts_var_nodes[0].size()<<" is it\n";

    // return;
    for (int i = 0; i < most_loaded_part; i++) {
        int part_indx = parts_ratios[i].second;
        int most_loaded_part_index = parts_ratios[most_loaded_part].second;
        while (parts_ratios[i].first < 1 && most_loaded_part > i &&
                parts_ratios[i].first < parts_ratios[most_loaded_part].first) {
            Node *node_to_move = parts_var_nodes[most_loaded_part_index].top().second;
            parts_var_nodes[most_loaded_part_index].pop();
            node_to_move->set_part(part_indx);
            var_memroy_ratios[part_indx] +=
                node_to_move->get_memory() / total_var_memory;
            var_memroy_ratios[most_loaded_part_index] -=
                node_to_move->get_memory() / total_var_memory;
            parts_ratios[i].first = K *
                (intermediate_memroy_ratios[part_indx] +
                 var_memroy_ratios[part_indx]) /
                2.0;
            parts_ratios[most_loaded_part].first =
                K *
                (intermediate_memroy_ratios[most_loaded_part_index] +
                 var_memroy_ratios[most_loaded_part_index]) /
                2.0;
            while ((parts_ratios[most_loaded_part].first <= 1 ||
                        parts_var_nodes[most_loaded_part_index].size() == 0) &&
                    most_loaded_part > i) {
                most_loaded_part--;
                most_loaded_part_index = parts_ratios[most_loaded_part].second;
            }
        }
    }

    cout << "ratios after\t";
    for (int i = 0; i < K; i++) {
        cout << parts_ratios[i].first << ","
            << var_memroy_ratios[parts_ratios[i].second] << ","
            << intermediate_memroy_ratios[parts_ratios[i].second] << ","
            << parts_ratios[i].second << "\t";
    }
    cout << "\n";
    // assign_distant_vars_to_cpu(0);
}
