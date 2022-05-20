#include "include/utils.h"
#include "include/json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using json = nlohmann::json;

Folder_Files folder_files;

int K = 2;
bool BALANCE_MEMORY = false;
int REPLICATION_FACTOR = 0;
long long int DEVICE_MEM_CAPACITY = 32000000000l;
double COMM_LATENCY = 25.0;
double COMM_TRANSFER_RATE_RECIPROCAL = 1.0 / 130000;

void clean_line(string &line) {
    string result = "";
    for (size_t i = 0; i < line.size(); i++) {
        if (line[i] != '\n' and line[i] != '\r' and line[i] != '\t' and
                line[i] != ' ' and line[i] != '"') {
            result += line[i];
        }
    }
    line = result;
}

void clean_line_keep_spaces(string &line) {
    string result = "";
    for (size_t i = 0; i < line.size(); i++) {
        if (line[i] != '\n' and line[i] != '\r' and line[i] != '\t' and
                line[i] != '"') {
            result += line[i];
        }
    }
    line = result;
}

void print_vector(vector<string> vect) {
    cout << "[ ";
    for (int i = 0; i < vect.size(); i++) {
        cout << vect[i] << " ";
    }
    cout << "]\n";
}

vector<string> split(const string &str, string delimiter) {
    vector<string> tokens;
    string token;
    int indx = 0;
    size_t i = 0;
    string maybe = "";
    while (i < str.size()) {
        while (str[i] == delimiter[indx]) {
            maybe += str[i];
            indx++;
            i++;
            if (indx == delimiter.size()) {
                maybe = "";
                tokens.push_back(token);
                token = "";
                break;
            }
        }
        indx = 0;
        if (maybe != "") {
            token += maybe;
            maybe = "";
        }
        token += str[i];
        i++;
    }
    if (token != "") {
        tokens.push_back(token);
    }
    return tokens;
}

void copy_unordered_map(unordered_map<string, int> &src,
        unordered_map<string, int> &dst) {
    for (auto &it : src) {
        dst[it.first] = it.second;
    }
}

void read_settings() {
    ifstream settings_file;
    settings_file.open("settings.json");

    if (!settings_file.good()) {
        throw SETTINGS_FILE_NOT_FOUND;
    }

    json settings;

    try {
        settings_file >> settings;
    } catch (nlohmann::detail::parse_error) {
        throw SETTINGS_FILE_ERROR;
    }

    if (!settings.contains("folder")) {
        throw NO_FOLDER_SUPPLIED;
    }

    folder_files.folder = settings["folder"];
    folder_files.files = {
        { GRAPH_FILE, settings.value("graph", folder_files.folder + "/crn_src_sink_low.dot") },
        { NODES_WEIGHTS_FILE, settings.value("nodes_weights", folder_files.folder + "/nodes_average_durations_fixed.txt") },
        { EDGES_COSTS_FILE, settings.value("edges_costs", folder_files.folder + "/tensors_sz_32_low.txt") },
        { VAR_NODES_FILE, settings.value("var_nodes", folder_files.folder + "/var_nodes.txt") },
        { REF_NODES_FILE, settings.value("ref_nodes", folder_files.folder + "/ref_nodes.txt") },
        { NO_OPS_FILE, settings.value("no_ops", folder_files.folder + "/no_ops.txt") },
        { CPU_NODES_FILE, settings.value("cpu_nodes", folder_files.folder + "/cpu_nodes.txt") },
        { VANILLA_PLACEMENT_FILE, settings.value( "vanilla_cleaned", folder_files.folder + "/vanilla_cleaned.place")},
        { MEMORY_FILE_NAME, settings.value("memory_file", folder_files.folder + "/memory.txt") }
    };

    K = settings.value("K", 2);
    COMM_TRANSFER_RATE_RECIPROCAL =
        1.0 / (settings.value("COMM_TRANSFER_RATE", 140) * 1000);
    COMM_LATENCY = settings.value("COMM_LATENCY", 25);
    BALANCE_MEMORY = settings.value("BALANCE_MEMORY", false);
    REPLICATION_FACTOR = settings.value("REPLICATION_FACTOR", 1);
    DEVICE_MEM_CAPACITY = (long)settings.value("DEVICE_MEM_CAPACITY", 800000000l);
}

/*         n --> No. of elements present in input array.
           BITree[0..n] --> Array that represents Binary Indexed Tree.
           arr[0..n-1] --> Input array for which prefix sum is evaluated. */

// Returns sum of arr[0..index]. This function assumes
// that the array is preprocessed and partial sums of
// array elements are stored in BITree[].
int getSum(int BITree[], int index) {
    int sum = 0; // Iniialize result

    // index in BITree[] is 1 more than the index in arr[]
    index = index + 1;

    // Traverse ancestors of BITree[index]
    while (index > 0) {
        // Add current element of BITree to sum
        sum += BITree[index];

        // Move index to parent node in getSum View
        index -= index & (-index);
    }
    return sum;
}

// Updates a node in Binary Index Tree (BITree) at given index
// in BITree. The given value 'val' is added to BITree[i] and
// all of its ancestors in tree.
void updateBIT(int BITree[], int n, int index, int val) {
    // index in BITree[] is 1 more than the index in arr[]
    index = index + 1;

    // Traverse all ancestors and add 'val'
    while (index <= n) {
        // Add 'val' to current node of BI Tree
        BITree[index] += val;

        // Update index to that of parent in update View
        index += index & (-index);
    }
}

// Constructs and returns a Binary Indexed Tree for given
// array of size n.
int *constructBITree(int arr[], int n) {
    // Create and initialize BITree[] as 0
    int *BITree = new int[n + 1];
    for (int i = 1; i <= n; i++)
        BITree[i] = 0;

    // Store the actual values in BITree[] using update()
    for (int i = 0; i < n; i++)
        updateBIT(BITree, n, i, arr[i]);

    // Uncomment below lines to see contents of BITree[]
    // for (int i=1; i<=n; i++)
    //     cout << BITree[i] << " ";

    return BITree;
}

// Constructs and returns a Binary Indexed Tree for given
// array of size n.
int *constructBITree(map<int, int> &mp, int n) {
    // Create and initialize BITree[] as 0
    int *BITree = new int[n + 1];
    for (int i = 1; i <= n; i++)
        BITree[i] = 0;

    // Store the actual values in BITree[] using update()
    int i = 0;
    for (auto &it : mp) {
        updateBIT(BITree, n, i, it.second);
        i++;
    }
    // Uncomment below lines to see contents of BITree[]
    // for (int i=1; i<=n; i++)
    //     cout << BITree[i] << " ";

    return BITree;
}
