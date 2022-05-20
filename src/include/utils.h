#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

#ifndef UTILS_H
#define UTILS_H

extern int K;
extern bool BALANCE_MEMORY;
extern int REPLICATION_FACTOR;
extern long long int DEVICE_MEM_CAPACITY;
extern double COMM_LATENCY;
extern double COMM_TRANSFER_RATE_RECIPROCAL; // Reciprocal of the transfer rate
                                             // eqivalent to 140 GB/s in us/byte

enum ERRORS {
    SETTINGS_FILE_NOT_FOUND,
    GRAPH_FILE_NOT_FOUND,
    WEIGHTS_FILE_NOT_FOUND,
    NODES_FILE_NOT_FOUND,
    REF_NODES_FILE_NOT_FOUND,
    NO_OP_NODES_FILE_NOT_FOUND,
    EDGES_WEIGHTS_FILE_NOT_FOUND,
    SETTINGS_FILE_ERROR,
    NO_FOLDER_SUPPLIED,
};

enum FILES {
    GRAPH_FILE,
    NODES_WEIGHTS_FILE,
    EDGES_COSTS_FILE,
    VAR_NODES_FILE,
    REF_NODES_FILE,
    NO_OPS_FILE,
    CPU_NODES_FILE,
    VANILLA_PLACEMENT_FILE,
    MEMORY_FILE_NAME
};

struct Folder_Files {
    string folder;
    unordered_map<FILES, string> files;
};

extern Folder_Files folder_files;

void clean_line(string &);

void clean_line_keep_spaces(string &);

void print_vector(vector<string> vect);

vector<string> split(const string &str, string delimiter);

void copy_unordered_map(unordered_map<string, int> &,
        unordered_map<string, int> &);

void read_settings();

int getSum(int[], int);
void updateBIT(int[], int, int, int);
int *constructBITree(int[], int);
int *constructBITree(map<int, int> &, int);
int *constructBITree(map<int, vector<int>> &, int);

#endif
