#include "include/linear_cluster.h"

LinearCluster::LinearCluster() {
    nodes = vector<Node *>();
    weight = -1;
    primary = false;
}

Node *LinearCluster::operator[](int indx) { return nodes[indx]; }

void LinearCluster::add_node(Node *node) { nodes.push_back(node); }

Node *LinearCluster::get_src_node() { return nodes[0]; }
Node *LinearCluster::get_snk_node() { return nodes[nodes.size() - 1]; }
int LinearCluster::length() { return nodes.size(); }
bool LinearCluster::is_primary() { return primary; }
int LinearCluster::get_weight() {
    if (weight == -1) {
        weight = 0;
        for (Node *node : nodes) {
            weight += node->get_duration();
        }
    }
    return weight;
}
void LinearCluster::set_weight(int _weight) { weight = _weight; }

void LinearCluster::print_cluster() {
    cout << "[";
    for (Node *node : nodes) {
        cout << node->get_name() << " :: ";
    }
    cout << "]\n";
}

int LinearCluster::get_indx(string &node_name) {
    int indx = -1;
    for (Node *node : nodes) {
        indx += 1;
        if (node->get_name() == node_name) {
            return indx;
        }
    }
    return -1;
}

pair<int, int> LinearCluster::get_span() { return span; }

void LinearCluster::set_span(pair<int, int> _span) { span = _span; }

bool LinearCluster::operator<(LinearCluster &lc) {
    return (weight < lc.get_weight());
}

double LinearCluster::sorting_criteria_mirror() {
    return get_src_node()->get_level() +
        double(get_snk_node()->get_level() / 1000000);
}

bool LinearCluster::start_end_sorting_key_mirror(LinearCluster &lc1,
        LinearCluster &lc2) {
    return lc1.sorting_criteria_mirror() < lc2.sorting_criteria_mirror();
}

bool LinearCluster::sorting_criteria_density(LinearCluster &lc1,
        LinearCluster &lc2) {
    return (double)(lc1.get_snk_node()->get_top_level() +
            lc1.get_snk_node()->get_duration() -
            lc1.get_src_node()->get_top_level()) /
        (double)(lc1.get_span().second - lc1.get_span().first) >
        (double)(lc2.get_snk_node()->get_top_level() +
                lc2.get_snk_node()->get_duration() -
                lc2.get_src_node()->get_top_level()) /
        (double)(lc2.get_span().second - lc2.get_span().first);
}

bool LinearCluster::sorting_criteria_weighted_level(LinearCluster &lc1,
        LinearCluster &lc2) {
    return lc1.get_src_node()->get_weighted_level() >
        lc2.get_src_node()->get_weighted_level();
}
